import json
import math
import ollama


# ------------------------------------------------
# Load ERP knowledge
# ------------------------------------------------

with open("knowledge/menus.json", "r", encoding="utf-8") as file:
    menus = json.load(file)


# ------------------------------------------------
# Test questions
# ------------------------------------------------

queries = [
    "میخوام یک عملیات مالی جدید انجام بدم",
    "چطور یک سند جدید ایجاد کنم؟",
    "صورت وضعیت حسابداری رو لازم دارم",
    "وضعیت صندوق های شرکت رو میخوام ببینم",
    "یک گزارش کامل از تنخواه لازم دارم",
    "گزارش مربوط به عملکرد تنخواه رو میخوام"
]


# ------------------------------------------------
# Convert ERP record to embedding text
# ------------------------------------------------

def menu_to_text(menu):

    keywords = "، ".join(menu["keywords"])

    return f"""
عنوان: {menu["title"]}
ماژول: {menu["module"]}
مسیر: {menu["menu_path"]}
توضیحات: {menu["description"]}
کلمات مرتبط: {keywords}
""".strip()


# ------------------------------------------------
# Cosine similarity
# ------------------------------------------------

def cosine_similarity(a, b):

    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (magnitude_a * magnitude_b)


# ------------------------------------------------
# Keyword score
# ------------------------------------------------

def keyword_score(query, menu):

    stop_words = {
        "من",
        "می",
        "میخوام",
        "میخواهم",
        "می‌خوام",
        "می‌خواهم",
        "یک",
        "یه",
        "را",
        "رو",
        "از",
        "به",
        "برای",
        "که",
        "چطور",
        "چگونه",
        "کجا",
        "لازم",
        "دارم",
        "کنم",
        "کنم؟",
        "است",
        "هست",
        "هستم",
        "می‌شود",
        "میشود"
    }

    query_words = [
        word.strip("؟،,.! ")
        for word in query.split()
        if word.strip("؟،,.! ") not in stop_words
    ]

    searchable_text = " ".join([
        menu["title"],
        menu["module"],
        menu["menu_path"],
        menu["description"],
        " ".join(menu["keywords"])
    ])

    if not query_words:
        return 0

    matched_words = 0

    for word in query_words:

        if word in searchable_text:
            matched_words += 1

    return matched_words / len(query_words)


# ------------------------------------------------
# Create ERP embeddings
# ------------------------------------------------

print("Creating ERP embeddings...")

menu_embeddings = []

for menu in menus:

    text = menu_to_text(menu)

    embedding = ollama.embed(
        model="nomic-embed-text:latest",
        input=text
    ).embeddings[0]

    menu_embeddings.append({
        "menu": menu,
        "embedding": embedding
    })


# ------------------------------------------------
# Test hybrid search
# ------------------------------------------------

for query in queries:

    print()
    print("=" * 70)
    print("Query:")
    print(query)
    print("-" * 70)

    query_embedding = ollama.embed(
        model="nomic-embed-text:latest",
        input=query
    ).embeddings[0]

    results = []

    for item in menu_embeddings:

        menu = item["menu"]

        semantic = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        keyword = keyword_score(
            query,
            menu
        )

        # Hybrid score
        final_score = (
            0.7 * semantic
            + 0.3 * keyword
        )

        results.append({
            "title": menu["title"],
            "semantic": semantic,
            "keyword": keyword,
            "final": final_score
        })

    # Highest hybrid score first
    results.sort(
        key=lambda x: x["final"],
        reverse=True
    )

    for result in results:

        print(
            f'{result["final"]:.4f}  '
            f'(semantic={result["semantic"]:.4f}, '
            f'keyword={result["keyword"]:.4f})  '
            f'{result["title"]}'
        )