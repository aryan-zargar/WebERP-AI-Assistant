import json
import math
import re
from pathlib import Path
from typing import List

import ollama

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MENUS_FILE = BASE_DIR / "knowledge" / "Menus.json"
EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings.json"

LLM_MODEL = "gemma3:12b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K = 5

SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.20
TITLE_WEIGHT = 0.15


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="WebERP AI",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.5.90:5173"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request model
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# Persian normalization
# ============================================================

PERSIAN_STOP_WORDS = {
    "من",
    "ما",
    "می",
    "میخوام",
    "میخواهم",
    "می‌خوام",
    "می‌خواهم",
    "یک",
    "یه",
    "این",
    "آن",
    "اون",
    "را",
    "رو",
    "از",
    "به",
    "برای",
    "در",
    "با",
    "که",
    "چطور",
    "چگونه",
    "کجا",
    "چی",
    "چه",
    "چیه",
    "لازم",
    "دارم",
    "دارید",
    "کنم",
    "کنیم",
    "کنید",
    "است",
    "هست",
    "هستم",
    "می‌شود",
    "میشود",
    "شود",
    "برام",
    "برامون",
    "لطفا",
    "لطفاً",
}


def normalize_persian(text: str) -> str:

    if not text:
        return ""

    text = text.strip()

    # Arabic -> Persian characters
    text = text.replace("ي", "ی")
    text = text.replace("ى", "ی")
    text = text.replace("ك", "ک")

    # Remove Arabic/Persian diacritics
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text,
    )

    # Normalize punctuation
    text = re.sub(
        r"[،؛؟,:;!?()\[\]{}\"'`]",
        " ",
        text,
    )

    # Normalize different dash characters
    text = re.sub(
        r"[-–—_]",
        " ",
        text,
    )

    # Treat zero-width non-joiner as a normal word boundary.
    # This makes forms such as "می‌خواهم" searchable consistently.
    text = text.replace("\u200c", " ")

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.lower().strip()


def extract_keywords(text: str) -> List[str]:

    normalized = normalize_persian(text)

    if not normalized:
        return []

    words = normalized.split()

    return [
        word
        for word in words
        if word not in PERSIAN_STOP_WORDS
        and len(word) > 1
    ]


# ============================================================
# Load knowledge
# ============================================================

print()
print("Loading WebERP knowledge...")

with open(
    MENUS_FILE,
    "r",
    encoding="utf-8",
) as file:

    menus_data = json.load(file)


print(
    f"Menus loaded: {len(menus_data)}"
)


# ============================================================
# Load embeddings
# ============================================================

print("Loading embeddings...")

with open(
    EMBEDDINGS_FILE,
    "r",
    encoding="utf-8",
) as file:

    embeddings_data = json.load(file)


embedding_lookup = {}

# IMPORTANT:
# embeddings.json contains an object with a "records" array.
embedding_records = embeddings_data.get(
    "records",
    []
)

for item in embedding_records:

    item_id = str(
        item.get("id", "")
    ).strip()

    embedding = item.get(
        "embedding"
    )

    if (
        item_id
        and isinstance(embedding, list)
    ):
        embedding_lookup[item_id] = embedding


print(
    f"Embeddings loaded: {len(embedding_lookup)}"
)

print()


# ============================================================
# Prepare menu records
# ============================================================

prepared_menus = []

for menu in menus_data:

    menu_id = str(
        menu.get("id", "")
    ).strip()

    if not menu_id:
        continue

    title = menu.get(
        "title",
        "",
    ) or ""

    menu_path = menu.get(
        "menu_path",
        "",
    ) or ""

    description = menu.get(
        "description",
        "",
    ) or ""

    prepared_menus.append(
        {
            "id": menu_id,

            "title": title,

            "menu_path": menu_path,

            "description": description,

            "normalized_title":
                normalize_persian(title),

            "normalized_path":
                normalize_persian(menu_path),

            "normalized_description":
                normalize_persian(description),

            "embedding":
                embedding_lookup.get(menu_id),
        }
    )


print(
    f"Prepared menus: {len(prepared_menus)}"
)

print()


# ============================================================
# Vector utilities
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b,
):

    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        return 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(
        vector_a,
        vector_b,
    ):

        dot_product += a * b

        norm_a += a * a
        norm_b += b * b

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return (
        dot_product
        / (
            math.sqrt(norm_a)
            * math.sqrt(norm_b)
        )
    )


# ============================================================
# Embedding generation
# ============================================================

def create_query_embedding(
    text: str,
):

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text,
    )

    return response["embedding"]


# ============================================================
# Retrieval indexes
# ============================================================

# We keep lexical indexes in memory because the menu knowledge is
# relatively small (~3K records). This lets us do much stronger
# exact/phrase matching before semantic reranking.
#
# Important ERP search principle:
#
#   "گزارش تحلیلی تنخواه"
#
# should prefer a menu containing ALL important concepts over a menu
# that is merely semantically similar to "گزارش تحلیلی".

TITLE_WEIGHT = 0.55
PATH_WEIGHT = 0.30
DESCRIPTION_WEIGHT = 0.15

SEMANTIC_WEIGHT = 0.35
LEXICAL_WEIGHT = 0.65

# Extra bonuses are deliberately separate from the base lexical score.
# They make exact business terminology difficult for semantic similarity
# to overpower.
EXACT_TITLE_PHRASE_BONUS = 0.35
ALL_CONCEPTS_TITLE_BONUS = 0.30
ALL_CONCEPTS_TEXT_BONUS = 0.18
RARE_TERM_TITLE_BONUS = 0.20
PREFIX_PHRASE_BONUS = 0.10

keyword_document_frequency = {}
phrase_document_frequency = {}

for menu in prepared_menus:

    title_words = set(
        menu["normalized_title"].split()
    )

    path_words = set(
        menu["normalized_path"].split()
    )

    document_words = title_words | path_words

    for word in document_words:

        if word in PERSIAN_STOP_WORDS:
            continue

        if len(word) <= 1:
            continue

        keyword_document_frequency[word] = (
            keyword_document_frequency.get(word, 0)
            + 1
        )

    # Store common 2-word phrases from title/path.
    # These are useful for concepts such as "گزارش تحلیلی".
    for field_text in (
        menu["normalized_title"],
        menu["normalized_path"],
    ):
        words = field_text.split()

        for index in range(len(words) - 1):

            phrase = (
                words[index]
                + " "
                + words[index + 1]
            )

            phrase_document_frequency[phrase] = (
                phrase_document_frequency.get(
                    phrase,
                    0,
                )
                + 1
            )


TOTAL_MENU_DOCUMENTS = max(
    len(prepared_menus),
    1,
)


def keyword_weight(
    keyword: str,
) -> float:

    document_frequency = (
        keyword_document_frequency.get(
            keyword,
            0,
        )
    )

    # Smoothed IDF-like weighting.
    # Rare ERP terms receive more importance than common terms.
    return (
        math.log(
            (TOTAL_MENU_DOCUMENTS + 1)
            / (document_frequency + 1)
        )
        + 1.0
    )


def phrase_weight(
    phrase: str,
) -> float:

    document_frequency = (
        phrase_document_frequency.get(
            phrase,
            0,
        )
    )

    return (
        math.log(
            (TOTAL_MENU_DOCUMENTS + 1)
            / (document_frequency + 1)
        )
        + 1.0
    )


def contains_term(
    text: str,
    term: str,
) -> bool:

    if not text or not term:
        return False

    # Whole phrase / whole token matching.
    # Persian words are whitespace-delimited after normalization.
    text_words = set(text.split())
    term_words = term.split()

    if len(term_words) == 1:
        return term in text_words

    return term in text


def phrase_matches(
    text: str,
    phrase: str,
) -> bool:

    if not text or not phrase:
        return False

    return phrase in text


def extract_search_concepts(
    query: str,
    concepts=None,
):
    """
    Return searchable concepts while preserving multi-word concepts.

    If intent concepts are available, use them because Qwen has already
    identified the important business terminology.

    Otherwise, fall back to query keywords and construct useful 2-word
    phrases from the query.
    """

    normalized_query = normalize_persian(
        query
    )

    result = []

    if concepts:

        for concept in concepts:

            if not isinstance(
                concept,
                str,
            ):
                continue

            normalized = normalize_persian(
                concept
            ).strip()

            if not normalized:
                continue

            if normalized in PERSIAN_STOP_WORDS:
                continue

            if normalized not in result:
                result.append(normalized)

    if not result:

        keywords = extract_keywords(
            normalized_query
        )

        result.extend(
            keywords
        )

        # Add adjacent 2-word phrases.
        for index in range(
            len(keywords) - 1
        ):

            phrase = (
                keywords[index]
                + " "
                + keywords[index + 1]
            )

            if phrase not in result:
                result.append(phrase)

    return result


def field_term_score(
    concepts,
    text: str,
) -> float:

    if not concepts or not text:
        return 0.0

    matched_weight = 0.0
    total_weight = 0.0

    for concept in concepts:

        weight = (
            phrase_weight(concept)
            if " " in concept
            else keyword_weight(concept)
        )

        total_weight += weight

        if contains_term(
            text,
            concept,
        ):
            matched_weight += weight

    if total_weight == 0:
        return 0.0

    return matched_weight / total_weight


def keyword_score(
    query_keywords,
    menu,
):
    """
    Backwards-compatible lexical score.

    This version is field-aware and IDF-weighted.
    """

    if not query_keywords:
        return 0.0

    title_score_value = field_term_score(
        query_keywords,
        menu["normalized_title"],
    )

    path_score_value = field_term_score(
        query_keywords,
        menu["normalized_path"],
    )

    description_score_value = field_term_score(
        query_keywords,
        menu["normalized_description"],
    )

    return (
        TITLE_WEIGHT * title_score_value
        + PATH_WEIGHT * path_score_value
        + DESCRIPTION_WEIGHT
        * description_score_value
    )


def title_score(
    normalized_query,
    query_keywords,
    menu,
):
    """
    Strong title/phrase evidence.

    Unlike the old implementation, path matching is evaluated even
    when the title has no matching terms.
    """

    title = menu[
        "normalized_title"
    ]

    path = menu[
        "normalized_path"
    ]

    if not title:
        return 0.0

    # Exact title.
    if normalized_query == title:
        return 1.0

    # Full query phrase in title.
    if (
        normalized_query
        and normalized_query in title
    ):
        return 0.90

    if not query_keywords:
        return 0.0

    title_words = set(
        title.split()
    )

    matched_title_words = sum(
        1
        for word in query_keywords
        if " " not in word
        and word in title_words
    )

    simple_keywords = [
        word
        for word in query_keywords
        if " " not in word
    ]

    if simple_keywords:

        title_ratio = (
            matched_title_words
            / len(simple_keywords)
        )

        if title_ratio == 1.0:
            return 0.80

        if title_ratio > 0:
            return 0.60 * title_ratio

    # Check path independently.
    path_words = set(
        path.split()
    )

    matched_path_words = sum(
        1
        for word in simple_keywords
        if word in path_words
    )

    if simple_keywords:

        path_ratio = (
            matched_path_words
            / len(simple_keywords)
        )

        if path_ratio == 1.0:
            return 0.50

        if path_ratio > 0:
            return 0.35 * path_ratio

    return 0.0


def calculate_lexical_features(
    normalized_query,
    concepts,
    menu,
):
    """
    Calculate detailed lexical evidence.

    The returned values are intentionally kept separate so the logs
    tell us exactly why a menu ranked where it did.
    """

    title = menu[
        "normalized_title"
    ]

    path = menu[
        "normalized_path"
    ]

    description = menu[
        "normalized_description"
    ]

    # Field-aware concept coverage.
    title_terms = field_term_score(
        concepts,
        title,
    )

    path_terms = field_term_score(
        concepts,
        path,
    )

    description_terms = field_term_score(
        concepts,
        description,
    )

    lexical = (
        TITLE_WEIGHT * title_terms
        + PATH_WEIGHT * path_terms
        + DESCRIPTION_WEIGHT
        * description_terms
    )

    # How many distinct concepts are actually present anywhere?
    matched_concepts = 0

    for concept in concepts:

        if (
            contains_term(title, concept)
            or contains_term(path, concept)
            or contains_term(
                description,
                concept,
            )
        ):
            matched_concepts += 1

    concept_coverage = (
        matched_concepts / len(concepts)
        if concepts
        else 0.0
    )

    # All concepts in title is exceptionally strong evidence.
    all_concepts_in_title = (
        bool(concepts)
        and all(
            contains_term(
                title,
                concept,
            )
            for concept in concepts
        )
    )

    # All concepts somewhere in the menu record.
    all_concepts_in_text = (
        bool(concepts)
        and all(
            contains_term(
                title,
                concept,
            )
            or contains_term(
                path,
                concept,
            )
            or contains_term(
                description,
                concept,
            )
            for concept in concepts
        )
    )

    # Exact full query phrase in title.
    exact_query_in_title = (
        bool(normalized_query)
        and normalized_query in title
    )

    # Rare query term in title.
    rare_term_in_title = False
    rare_term = ""
    rare_term_weight = 0.0

    simple_concepts = [
        concept
        for concept in concepts
        if " " not in concept
    ]

    if simple_concepts:

        rare_term = max(
            simple_concepts,
            key=keyword_weight,
        )

        rare_term_weight = keyword_weight(
            rare_term
        )

        rare_term_in_title = (
            rare_term in title.split()
        )

    # 2-word concept phrase in title.
    concept_phrase_in_title = False

    for concept in concepts:

        if " " in concept and phrase_matches(
            title,
            concept,
        ):
            concept_phrase_in_title = True
            break

    return {
        "lexical": lexical,
        "title_terms": title_terms,
        "path_terms": path_terms,
        "description_terms":
            description_terms,
        "concept_coverage":
            concept_coverage,
        "matched_concepts":
            matched_concepts,
        "all_concepts_in_title":
            all_concepts_in_title,
        "all_concepts_in_text":
            all_concepts_in_text,
        "exact_query_in_title":
            exact_query_in_title,
        "rare_term":
            rare_term,
        "rare_term_weight":
            rare_term_weight,
        "rare_term_in_title":
            rare_term_in_title,
        "concept_phrase_in_title":
            concept_phrase_in_title,
    }


# ============================================================
# Hybrid ERP search
# ============================================================

def search_erp(
    query: str,
    concepts=None,
):
    """
    Two-stage-style hybrid retrieval.

    Stage 1:
        Evaluate every menu for lexical + semantic evidence.

    Stage 2:
        Apply strong business-term bonuses and rerank.

    This avoids allowing a generic semantic match such as
    "گزارش تحلیلی فرايندها" to beat a menu containing the
    complete requested business concept "تنخواه".
    """

    normalized_query = normalize_persian(
        query
    )

    if concepts is None:
        concepts = extract_search_concepts(
            normalized_query
        )
    else:
        concepts = extract_search_concepts(
            normalized_query,
            concepts,
        )

    query_keywords = extract_keywords(
        normalized_query
    )

    query_embedding = create_query_embedding(
        normalized_query
    )

    scored_results = []

    for menu in prepared_menus:

        embedding = menu.get(
            "embedding"
        )

        semantic = cosine_similarity(
            query_embedding,
            embedding,
        )

        features = calculate_lexical_features(
            normalized_query,
            concepts,
            menu,
        )

        lexical = features[
            "lexical"
        ]

        # Base hybrid score.
        base_score = (
            SEMANTIC_WEIGHT * semantic
            + LEXICAL_WEIGHT * lexical
        )

        bonus = 0.0

        # Strongest possible evidence:
        # the exact complete search phrase occurs in title.
        if features[
            "exact_query_in_title"
        ]:
            bonus += (
                EXACT_TITLE_PHRASE_BONUS
            )

        # Qwen identified multiple important concepts
        # and the title contains every one of them.
        if features[
            "all_concepts_in_title"
        ]:
            bonus += (
                ALL_CONCEPTS_TITLE_BONUS
            )

        # All concepts occur somewhere in title/path/description.
        elif features[
            "all_concepts_in_text"
        ]:
            bonus += (
                ALL_CONCEPTS_TEXT_BONUS
            )

        # Reward a rare business term in the title.
        if features[
            "rare_term_in_title"
        ]:
            bonus += (
                RARE_TERM_TITLE_BONUS
            )

        # Reward an important multi-word phrase in title.
        if features[
            "concept_phrase_in_title"
        ]:
            bonus += (
                PREFIX_PHRASE_BONUS
            )

        final_score = (
            base_score
            + bonus
        )

        result = {
            "id": menu["id"],
            "title": menu["title"],
            "menu_path":
                menu["menu_path"],
            "description":
                menu["description"],
            "semantic_score":
                semantic,
            "keyword_score":
                lexical,
            "title_score":
                title_score(
                    normalized_query,
                    query_keywords,
                    menu,
                ),
            "score":
                final_score,

            # Diagnostic retrieval information.
            "title_term_score":
                features[
                    "title_terms"
                ],
            "path_term_score":
                features[
                    "path_terms"
                ],
            "description_term_score":
                features[
                    "description_terms"
                ],
            "concept_coverage":
                features[
                    "concept_coverage"
                ],
            "matched_concepts":
                features[
                    "matched_concepts"
                ],
            "all_concepts_in_title":
                features[
                    "all_concepts_in_title"
                ],
            "all_concepts_in_text":
                features[
                    "all_concepts_in_text"
                ],
            "exact_query_in_title":
                features[
                    "exact_query_in_title"
                ],
            "rare_term":
                features[
                    "rare_term"
                ],
            "rare_term_in_title":
                features[
                    "rare_term_in_title"
                ],
            "bonus":
                bonus,
        }

        scored_results.append(
            result
        )

    scored_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_results[:TOP_K]


# ============================================================
# Intent understanding
# ============================================================

INTENT_SYSTEM_PROMPT = """
You are the intent-understanding component of WebERP AI.

Your job is to analyze the user's question and extract the important
concepts that should be used to search the WebERP menu knowledge.

Return ONLY valid JSON.

Required JSON format:

{
  "is_erp_related": true,
  "concepts": [],
  "domain": "",
  "action": ""
}

Rules:

1. is_erp_related must be true when the question is about WebERP,
   ERP menus, features, accounting, finance, inventory, HR,
   reports, documents, workflows, or other ERP operations.

2. is_erp_related must be false for greetings, thanks, casual
   conversation, or unrelated questions.

3. concepts are the IMPORTANT searchable terms from the user's
   question.

4. NEVER remove an important business term from concepts.

5. Preserve specific ERP nouns and qualifiers.

   For example:

   User:
   "گزارش تحلیلی تنخواه کجاست؟"

   concepts must include:
   ["گزارش تحلیلی", "تنخواه"]

6. Do NOT replace a specific term with a more generic term.

   For example, do NOT turn:
   "گزارش تحلیلی تنخواه"

   into only:
   "گزارش تحلیلی"

7. If the user mentions a specific object, document, report,
   process, module, or business concept, include it in concepts.

8. Do not include conversational words such as:
   "کجاست", "چطور", "چگونه", "میخواهم", "میخوام", "لطفا".

9. domain should contain the ERP domain only when it is reasonably
   clear, such as:
   حسابداری، خزانه داری، انبار، فروش، خرید، منابع انسانی.

10. action should contain the user's intended operation when clear,
    such as:
    مشاهده، ایجاد، ویرایش، حذف، گزارش گیری.

11. Do not invent ERP menu names.

12. Do not invent concepts that are not present or clearly implied
    by the user's question.

13. Preserve multiple important concepts when they occur together.

14. Return JSON only.
15. Do not use Markdown.
16. Do not use ```json.
"""


def understand_query(
    message: str,
):

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content":
                    INTENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        options={
            "temperature": 0,
            "num_predict": 200,
        },
    )

    content = response[
        "message"
    ][
        "content"
    ].strip()

    print()
    print("Raw intent response:")
    print(content)

    # --------------------------------------------------------
    # Remove Markdown code fences
    # --------------------------------------------------------

    if content.startswith("```"):

        lines = content.splitlines()

        if (
            lines
            and lines[0]
            .strip()
            .startswith("```")
        ):
            lines = lines[1:]

        if (
            lines
            and lines[-1]
            .strip()
            == "```"
        ):
            lines = lines[:-1]

        content = "\n".join(
            lines
        ).strip()

    try:

        intent = json.loads(
            content
        )

        return {
            "is_erp_related": bool(
                intent.get(
                    "is_erp_related",
                    False,
                )
            ),

            "concepts":
                intent.get(
                    "concepts",
                    [],
                ),

            "domain":
                intent.get(
                    "domain",
                    "",
                ),

            "action":
                intent.get(
                    "action",
                    "",
                ),
        }

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):

        print(
            "Could not parse intent JSON:"
        )

        print(content)

        # Safer fallback:
        # Do NOT send an unknown request
        # into ERP search.

        return {
            "is_erp_related": False,
            "concepts": [],
            "domain": "",
            "action": "",
        }


# ============================================================
# Build search query
# ============================================================

def build_search_query(
    original_message: str,
    intent: dict,
):

    parts = []

    concepts = intent.get(
        "concepts",
        [],
    )

    if concepts:

        parts.extend(
            concepts
        )

    domain = intent.get(
        "domain",
        "",
    )

    if domain:
        parts.append(domain)

    action = intent.get(
        "action",
        "",
    )

    if action:
        parts.append(action)

    search_query = " ".join(
        parts
    ).strip()

    if not search_query:
        return original_message

    return search_query


# ============================================================
# Format knowledge for Qwen
# ============================================================

def format_knowledge_for_qwen(
    results: list,
):

    if not results:
        return "No relevant WebERP knowledge was found."

    blocks = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        block = f"""
Result {index}

ID:
{result["id"]}

Title:
{result["title"]}

Menu path:
{result["menu_path"]}

Description:
{result["description"]}

Relevance score:
{result["score"]:.4f}
"""

        blocks.append(
            block.strip()
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# Final answer prompt
# ============================================================

ANSWER_SYSTEM_PROMPT = """
You are WebERP AI, an assistant for the WebERP system.

Your name is "WebERP AI"

Answer the user's question in Persian.

The provided WebERP knowledge is the source of truth.

IMPORTANT RULES:

1. Do not invent WebERP menus, buttons, fields, workflows,
   pages, or procedures.

2. Only state information that is supported by the retrieved
   WebERP knowledge.

3. If the retrieved information does not clearly answer the
   user's question, say that the available WebERP knowledge
   is not sufficient to answer precisely.

4. Do not blindly list all retrieved results.

5. Prefer the most relevant result.

6. If multiple results are genuinely relevant, explain the
   distinction between them.

7. When a menu path is available, present it clearly.

8. Do not expose internal retrieval scores.

9. Do not mention embeddings, vector search, Qwen, prompts,
   retrieval, or internal implementation details.

10. Keep the answer concise and useful.

11. Never claim that a menu or feature exists unless the
    retrieved WebERP knowledge supports it.
"""


# ============================================================
# Extract text from Ollama streaming chunks
# ============================================================

def get_stream_text(
    chunk,
):

    try:

        # Newer Ollama Python client
        if hasattr(
            chunk,
            "message",
        ):

            return (
                chunk.message.content
                or ""
            )

        # Dictionary-style response
        if isinstance(
            chunk,
            dict,
        ):

            message = chunk.get(
                "message",
                {},
            )

            if isinstance(
                message,
                dict,
            ):

                return (
                    message.get(
                        "content",
                        "",
                    )
                    or ""
                )

        return ""

    except Exception as e:

        print(
            "Error reading Ollama stream chunk:"
        )

        print(
            repr(e)
        )

        return ""


# ============================================================
# Stream ERP answer
# ============================================================

def stream_answer(
    original_question: str,
    intent: dict,
    results: list,
):

    try:

        knowledge = (
            format_knowledge_for_qwen(
                results
            )
        )

        prompt = f"""
User question:
{original_question}

User intent:
{json.dumps(
    intent,
    ensure_ascii=False,
    indent=2
)}

Relevant WebERP knowledge:
{knowledge}
"""

        print()
        print(
            "Starting ERP answer stream..."
        )

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content":
                        ANSWER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
            options={
                "temperature": 0.2,
            },
        )

        print(
            "Ollama ERP stream created."
        )

        for chunk in response:

            text = get_stream_text(
                chunk
            )

            if text:

                print(
                    f"STREAM CHUNK: {text!r}"
                )

                yield text

        print(
            "ERP answer stream finished."
        )

    except Exception as e:

        print()
        print(
            "!!! ERP STREAMING ERROR !!!"
        )

        print(
            repr(e)
        )

        print()

        raise


# ============================================================
# Stream general conversation
# ============================================================

GENERAL_SYSTEM_PROMPT = """
You are WebERP AI.

Your name is "WebERP AI".

You are a helpful AI assistant associated with the WebERP system.

Answer the user's question in Persian unless the user explicitly
asks for another language.

IDENTITY RULES:

1. If the user asks your name, say:
   "من سیستم یار هستم."

2. If the user asks who you are, explain briefly that you are
   WebERP AI, an AI assistant for the WebERP system.

3. Do not claim to be a human.

4. Do not invent capabilities that you do not have.

5. Keep casual conversation concise and natural.

6. If the user asks about your religeon tell them you follow the "Asnaghian Shia" Cult or in persian "شیعه اسنقیان"

For general conversation, you do not need to perform WebERP
knowledge retrieval.
"""

def stream_general_answer(
    original_question: str,
):

    try:

        print()
        print(
            "Starting general answer stream..."
        )

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": GENERAL_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                        original_question,
                },
            ],
            stream=True,
            options={
                "temperature": 0.4,
            },
        )

        print(
            "Ollama general stream created."
        )

        for chunk in response:

            text = get_stream_text(
                chunk
            )

            if text:

                print(
                    f"STREAM CHUNK: {text!r}"
                )

                yield text

        print(
            "General answer stream finished."
        )

    except Exception as e:

        print()
        print(
            "!!! GENERAL STREAMING ERROR !!!"
        )

        print(
            repr(e)
        )

        print()

        raise


# ============================================================
# Chat endpoint
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest,
):

    message = request.message.strip()

    if not message:

        return {
            "answer":
                "لطفاً یک پیام وارد کنید."
        }

    print()
    print(
        "================================"
    )

    print(
        "NEW CHAT REQUEST"
    )

    print(
        "================================"
    )

    print(
        "User:"
    )

    print(
        message
    )

    # --------------------------------------------------------
    # Step 1: Understand intent
    # --------------------------------------------------------

    intent = understand_query(
        message
    )

    print()
    print(
        "Intent:"
    )

    print(
        json.dumps(
            intent,
            ensure_ascii=False,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Step 2: General conversation
    # --------------------------------------------------------

    if not intent[
        "is_erp_related"
    ]:

        print()
        print(
            "General conversation - "
            "skipping ERP search."
        )

        return StreamingResponse(
            stream_general_answer(
                message
            ),
            media_type=
                "text/plain; charset=utf-8",
        )

    # --------------------------------------------------------
    # Step 3: Build search query
    # --------------------------------------------------------

    search_query = (
        build_search_query(
            message,
            intent,
        )
    )

    print()
    print(
        "Search query:"
    )

    print(
        search_query
    )

    # --------------------------------------------------------
    # Step 4: ERP search
    # --------------------------------------------------------

    results = search_erp(
        search_query,
        concepts=intent.get(
            "concepts",
            [],
        ),
    )

    print()
    print(
        "Top results:"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"{index}. "
            f"{result['title']} "
            f"(score={result['score']:.4f}, "
            f"semantic={result['semantic_score']:.4f}, "
            f"lexical={result['keyword_score']:.4f}, "
            f"title={result['title_score']:.4f}, "
            f"coverage={result['concept_coverage']:.2f}, "
            f"matched={result['matched_concepts']}, "
            f"bonus={result['bonus']:.4f})"
        )

    # --------------------------------------------------------
    # Step 5: Stream final answer
    # --------------------------------------------------------

    return StreamingResponse(
        stream_answer(
            message,
            intent,
            results,
        ),
        media_type=
            "text/plain; charset=utf-8",
    )


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "name": "WebERP AI",
        "status": "running",
    }

