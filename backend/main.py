
import json
import math
import re
from pathlib import Path
from typing import Any
from fastapi.middleware.cors import CORSMiddleware
import ollama
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

LLM_MODEL = "qwen-local:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"

MENUS_FILE = "knowledge/Menus.json"
EMBEDDINGS_FILE = "data/embeddings.json"

TOP_K = 5

SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.20
TITLE_WEIGHT = 0.15


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="WebERP AI Assistant",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# Request models
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# Persian text normalization
# ============================================================

ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def normalize_text(text: str) -> str:
    """
    Normalize Persian/Arabic text so that search is not affected
    by common Unicode differences.
    """

    if not text:
        return ""

    text = str(text).strip()

    # Arabic/Persian character normalization
    text = text.replace("ي", "ی")
    text = text.replace("ى", "ی")
    text = text.replace("ك", "ک")

    # Remove Arabic diacritics
    text = ARABIC_DIACRITICS.sub("", text)

    # Normalize punctuation
    punctuation_map = {
        "،": " ",
        "؛": " ",
        "؟": " ",
        ",": " ",
        ";": " ",
        "?": " ",
        "!": " ",
        ":": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "/": " ",
        "\\": " ",
        "-": " ",
        "_": " ",
    }

    for old, new in punctuation_map.items():
        text = text.replace(old, new)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


# ============================================================
# Stop words
# ============================================================

STOP_WORDS = {
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


def extract_keywords(text: str) -> list[str]:
    """
    Extract useful search keywords from Persian text.
    """

    normalized = normalize_text(text)

    words = normalized.split()

    result = []

    for word in words:
        if not word:
            continue

        if word in STOP_WORDS:
            continue

        # Ignore very short tokens
        if len(word) <= 1:
            continue

        result.append(word)

    return result


# ============================================================
# Load menu data
# ============================================================

def load_json_file(path: str) -> Any:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path.resolve()}"
        )

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


menus = load_json_file(MENUS_FILE)
embeddings_data = load_json_file(EMBEDDINGS_FILE)


# ============================================================
# Prepare embeddings lookup
# ============================================================
embedding_lookup: dict[str, list[float]] = {}

embedding_records = embeddings_data.get("records", [])

for item in embedding_records:
    item_id = str(item.get("id", "")).strip()
    embedding = item.get("embedding")

    if item_id and isinstance(embedding, list):
        embedding_lookup[item_id] = embedding


# ============================================================
# Prepare menu records
# ============================================================

prepared_menus = []

for menu in menus:
    menu_id = str(menu.get("id", "")).strip()

    if not menu_id:
        continue

    title = str(menu.get("title") or "")
    menu_path = str(menu.get("menu_path") or "")
    description = str(menu.get("description") or "")

    normalized_title = normalize_text(title)
    normalized_path = normalize_text(menu_path)
    normalized_description = normalize_text(description)

    searchable_text = " ".join(
        part
        for part in [
            normalized_title,
            normalized_path,
            normalized_description,
        ]
        if part
    )

    prepared_menus.append(
        {
            "id": menu_id,
            "title": title,
            "menu_path": menu_path,
            "description": description,
            "parent_id": menu.get("parent_id"),
            "action": menu.get("action"),
            "is_summary": menu.get("is_summary"),
            "is_readonly": menu.get("is_readonly"),
            "window_id": menu.get("window_id"),
            "process_id": menu.get("process_id"),
            "form_id": menu.get("form_id"),
            "entity_type": menu.get("entity_type"),
            "is_transaction": menu.get("is_transaction"),
            "seqno": menu.get("seqno"),
            "normalized_title": normalized_title,
            "normalized_path": normalized_path,
            "normalized_description": normalized_description,
            "searchable_text": searchable_text,
            "embedding": embedding_lookup.get(menu_id),
        }
    )


print("================================")
print("WebERP AI started successfully.")
print(f"Menus loaded: {len(prepared_menus)}")
print(f"Embeddings loaded: {len(embedding_lookup)}")
print(f"LLM model: {LLM_MODEL}")
print(f"Embedding model: {EMBEDDING_MODEL}")
print("================================")


# ============================================================
# Vector utilities
# ============================================================

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        return 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (
        math.sqrt(norm_a) * math.sqrt(norm_b)
    )


# ============================================================
# Embedding
# ============================================================

def create_query_embedding(text: str) -> list[float]:
    """
    Create an embedding for a query using Ollama.
    """

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )

    return response["embedding"]


# ============================================================
# Search scoring
# ============================================================

def calculate_keyword_score(
    query_keywords: list[str],
    menu: dict[str, Any]
) -> float:

    if not query_keywords:
        return 0.0

    searchable_text = menu["searchable_text"]

    matches = 0

    for keyword in query_keywords:
        if keyword in searchable_text:
            matches += 1

    return matches / len(query_keywords)


def calculate_title_score(
    normalized_query: str,
    query_keywords: list[str],
    menu: dict[str, Any]
) -> float:

    title = menu["normalized_title"]
    path = menu["normalized_path"]

    if not title:
        return 0.0

    # Exact title match
    if normalized_query == title:
        return 1.0

    # Whole query appears in title
    if normalized_query and normalized_query in title:
        return 0.85

    if query_keywords:
        title_matches = sum(
            1
            for keyword in query_keywords
            if keyword in title
        )

        title_ratio = title_matches / len(query_keywords)

        if title_ratio == 1.0:
            return 0.75

        if title_ratio > 0:
            return 0.55 * title_ratio

    # Query words appear in menu path
    if query_keywords:
        path_matches = sum(
            1
            for keyword in query_keywords
            if keyword in path
        )

        if path_matches == len(query_keywords):
            return 0.45

    return 0.0


# ============================================================
# ERP Search
# ============================================================

def search_erp(
    query: str,
    top_k: int = TOP_K
) -> list[dict[str, Any]]:
    """
    Hybrid ERP search.

    Uses:
    - semantic similarity
    - keyword matching
    - title/path matching
    """

    normalized_query = normalize_text(query)

    query_keywords = extract_keywords(query)

    query_embedding = create_query_embedding(query)

    results = []

    for menu in prepared_menus:

        embedding = menu.get("embedding")

        if embedding:
            semantic_score = cosine_similarity(
                query_embedding,
                embedding
            )
        else:
            semantic_score = 0.0

        keyword_score = calculate_keyword_score(
            query_keywords,
            menu
        )

        title_score = calculate_title_score(
            normalized_query,
            query_keywords,
            menu
        )

        final_score = (
            SEMANTIC_WEIGHT * semantic_score
            + KEYWORD_WEIGHT * keyword_score
            + TITLE_WEIGHT * title_score
        )

        results.append(
            {
                "id": menu["id"],
                "title": menu["title"],
                "menu_path": menu["menu_path"],
                "description": menu["description"],
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "title_score": title_score,
                "score": final_score,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# Qwen - Intent Understanding
# ============================================================

INTENT_SYSTEM_PROMPT = """
You are the query-understanding component of a Persian WebERP AI assistant.

Your job is ONLY to understand the user's message and classify whether it is
related to the WebERP ERP system.

Return ONLY valid JSON.

The JSON must have exactly these fields:

{
  "is_erp_related": true,
  "concepts": [],
  "domain": "",
  "action": ""
}

Rules:

1. Set "is_erp_related" to true when the user is asking about WebERP,
   ERP menus, ERP features, accounting, inventory, HR, finance,
   reports, documents, workflows, or how to perform an ERP operation.

2. Set "is_erp_related" to false for normal conversation such as:
   greetings, thanks, casual conversation, general questions,
   or messages unrelated to WebERP.

3. "concepts" should contain the important concepts from the user's
   question that can help search the ERP knowledge base.

4. Do NOT invent specific WebERP menu names.

5. "domain" should contain a domain only when it is reasonably clear,
   such as:
   حسابداری
   مالی
   انبار
   منابع انسانی
   فروش
   خرید

6. "action" should describe the requested operation when clear,
   such as:
   ایجاد
   ثبت
   ویرایش
   حذف
   مشاهده
   جستجو
   گزارش

7. If the message is a greeting or casual conversation, return:
   {
     "is_erp_related": false,
     "concepts": [],
     "domain": "",
     "action": ""
   }

Examples:

User:
سلام

Output:
{
  "is_erp_related": false,
  "concepts": [],
  "domain": "",
  "action": ""
}

User:
ممنون

Output:
{
  "is_erp_related": false,
  "concepts": [],
  "domain": "",
  "action": ""
}

User:
سند حسابداری کجاست؟

Output:
{
  "is_erp_related": true,
  "concepts": ["سند حسابداری"],
  "domain": "حسابداری",
  "action": "مشاهده"
}

User:
چطور یک سند حسابداری جدید ایجاد کنم؟

Output:
{
  "is_erp_related": true,
  "concepts": ["سند حسابداری", "سند جدید"],
  "domain": "حسابداری",
  "action": "ایجاد"
}
"""


def understand_query(message: str):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": INTENT_SYSTEM_PROMPT,
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

    content = response["message"]["content"].strip()

    try:
        intent = json.loads(content)

        return {
            "is_erp_related": bool(
                intent.get("is_erp_related", True)
            ),
            "concepts": intent.get("concepts", []),
            "domain": intent.get("domain", ""),
            "action": intent.get("action", ""),
        }

    except (json.JSONDecodeError, TypeError, ValueError):
        print("Could not parse intent JSON:")
        print(content)

        # Safe fallback:
        # if Qwen fails to classify, assume ERP-related so
        # we don't silently ignore a potentially useful ERP question.
        return {
            "is_erp_related": True,
            "concepts": [],
            "domain": "",
            "action": "",
        }


# ============================================================
# Build search query from intent
# ============================================================

def build_search_query(
    original_message: str,
    intent: dict[str, Any]
) -> str:
    """
    Convert Qwen's structured intent into a search query.

    The original message is included as a fallback/context term
    so we do not completely lose information during intent
    extraction.
    """

    parts: list[str] = []

    concepts = intent.get("concepts", [])

    if isinstance(concepts, list):
        parts.extend(
            str(item).strip()
            for item in concepts
            if str(item).strip()
        )

    domain = str(
        intent.get("domain") or ""
    ).strip()

    action = str(
        intent.get("action") or ""
    ).strip()

    if domain:
        parts.append(domain)

    if action:
        parts.append(action)

    # If Qwen produced nothing useful, fall back to original text.
    if not parts:
        return original_message

    # Remove duplicates while preserving order.
    unique_parts = []

    for part in parts:
        normalized_part = normalize_text(part)

        if not normalized_part:
            continue

        already_exists = any(
            normalize_text(existing) == normalized_part
            for existing in unique_parts
        )

        if not already_exists:
            unique_parts.append(part)

    return " ".join(unique_parts)


# ============================================================
# Qwen - Final Answer
# ============================================================

ANSWER_SYSTEM_PROMPT = """
You are a Persian AI assistant for the WebERP system.

You receive:

1. The user's original question.
2. A structured interpretation of the question.
3. Retrieved WebERP knowledge.

Your job is to answer the user's question using the
retrieved WebERP knowledge.

IMPORTANT RULES:

- The retrieved ERP knowledge is the source of truth.
- Do NOT invent WebERP menus.
- Do NOT invent buttons.
- Do NOT invent fields.
- Do NOT invent screens.
- Do NOT invent workflows.
- Do NOT invent reports.
- Do NOT invent capabilities.

The structured interpretation is only an aid for understanding
the user's intent. It is NOT proof that a particular ERP feature
exists.

Only claim something about WebERP when it is supported by the
retrieved ERP knowledge.

If the retrieved information is insufficient, clearly say that
the available ERP information is not enough to determine the
exact answer.

If several retrieved menus could reasonably match the question,
mention the relevant possibilities and ask the user which one
they mean.

When a menu path is available, use the exact Persian menu path.

Do not invent additional steps after opening a menu unless the
retrieved information actually contains those steps.

Answer naturally and concisely in Persian.

Do not mention internal implementation details such as:
- embeddings
- cosine similarity
- vector search
- retrieval
- prompts
- Qwen
- ranking

unless the user explicitly asks about the AI system itself.
"""


def format_knowledge_for_qwen(
    results: list[dict[str, Any]]
) -> str:
    """
    Convert retrieved ERP records into a clean context for Qwen.
    """

    if not results:
        return "هیچ اطلاعات مرتبطی از WebERP پیدا نشد."

    sections = []

    for index, result in enumerate(results, start=1):

        section = [
            f"مورد {index}",
            f"شناسه: {result.get('id', '')}",
            f"عنوان: {result.get('title', '')}",
            f"مسیر منو: {result.get('menu_path', '')}",
        ]

        description = result.get("description")

        if description:
            section.append(
                f"توضیحات: {description}"
            )

        sections.append(
            "\n".join(section)
        )

    return "\n\n".join(sections)


def generate_answer(
    original_question: str,
    intent: dict[str, Any],
    results: list[dict[str, Any]]
) -> str:
    """
    Ask Qwen for the final Persian answer using retrieved ERP data.
    """

    knowledge = format_knowledge_for_qwen(results)

    intent_json = json.dumps(
        intent,
        ensure_ascii=False,
        indent=2
    )

    user_prompt = f"""
سؤال کاربر:

{original_question}


برداشت ساختاری از سؤال:

{intent_json}


اطلاعات موجود در WebERP:

{knowledge}


اکنون پاسخ مناسب را به کاربر بده.
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": ANSWER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0.2,
                "num_predict": 500,
            },
        )

        return response["message"]["content"].strip()

    except Exception as exc:
        print(
            f"[Answer] Failed to generate answer: {exc}"
        )

        return (
            "در حال حاضر امکان تولید پاسخ وجود ندارد. "
            "لطفاً دوباره تلاش کنید."
        )


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {
        "name": "WebERP AI Assistant",
        "status": "running",
        "menus": len(prepared_menus),
        "model": LLM_MODEL,
    }


@app.get("/menus")
def get_menus():
    """
    Return all loaded ERP menus.
    """

    return {
        "count": len(prepared_menus),
        "menus": [
            {
                "id": menu["id"],
                "title": menu["title"],
                "menu_path": menu["menu_path"],
            }
            for menu in prepared_menus
        ],
    }


@app.get("/search")
def search(
    q: str
):
    """
    Direct ERP search endpoint.

    This remains available for testing the retrieval system
    independently from Qwen.
    """

    if not q.strip():
        return {
            "query": q,
            "results": [],
        }

    results = search_erp(q)

    return {
        "query": q,
        "results": results,
    }


@app.post("/chat")
def chat(
    request: ChatRequest
):
    """
    Main AI pipeline:

        User question
              ↓
        Qwen intent understanding
              ↓
        Build search query
              ↓
        ERP retrieval
              ↓
        Qwen final answer
    """

    message = request.message.strip()

    if not message:
        return {
            "answer": "لطفاً سؤال خود را وارد کنید.",
            "intent": {
                "concepts": [],
                "domain": "",
                "action": "",
            },
            "search_query": "",
            "results": [],
        }

    print()
    print("================================")
    print("NEW CHAT REQUEST")
    print("================================")
    print("User:")
    print(message)

    # --------------------------------------------------------
    # Step 1: Understand the user's intention
    # --------------------------------------------------------

    intent = understand_query(message)

    print()
    print("Intent:")
    print(
        json.dumps(
            intent,
            ensure_ascii=False,
            indent=2
        )
    )

    # --------------------------------------------------------
    # Step 2: Build a smarter ERP search query
    # --------------------------------------------------------

    search_query = build_search_query(
        message,
        intent
    )

    print()
    print("Search query:")
    print(search_query)

    # --------------------------------------------------------
    # Step 3: Search ERP knowledge
    # --------------------------------------------------------

    results = search_erp(search_query)

    print()
    print("Top results:")

    for index, result in enumerate(results, start=1):
        print(
            f"{index}. "
            f"{result['title']} "
            f"({result['score']:.4f})"
        )

    # --------------------------------------------------------
    # Step 4: Ask Qwen for final answer
    # --------------------------------------------------------

    answer = generate_answer(
        original_question=message,
        intent=intent,
        results=results
    )

    print()
    print("Answer:")
    print(answer)
    print("================================")
    print()

    # --------------------------------------------------------
    # Development response
    #
    # We intentionally expose intent/search/results for now.
    # Later we can simplify this to only return "answer".
    # --------------------------------------------------------

    return {
        "answer": answer,
        "intent": intent,
        "search_query": search_query,
        "results": results,
    }

