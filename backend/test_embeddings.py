import json
import math
import ollama


# Load ERP knowledge
with open("knowledge/menus.json", "r", encoding="utf-8") as file:
    menus = json.load(file)


# Test questions
queries = [
    "میخوام یک عملیات مالی جدید انجام بدم",
    "چطور یک سند جدید ایجاد کنم؟",
    "صورت وضعیت حسابداری رو لازم دارم",
    "وضعیت صندوق های شرکت رو میخوام ببینم",
    "یک گزارش کامل از تنخواه لازم دارم",
    "گزارش مربوط به عملکرد تنخواه رو میخوام"
]


# Convert an ERP record into searchable text
def menu_to_text(menu):

    keywords = "، ".join(menu["keywords"])

    return f"""
عنوان: {menu["title"]}
ماژول: {menu["module"]}
مسیر: {menu["menu_path"]}
توضیحات: {menu["description"]}
کلمات مرتبط: {keywords}
""".strip()


# Calculate cosine similarity
def cosine_similarity(a, b):

    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (magnitude_a * magnitude_b)


# ------------------------------------------------
# Create embeddings for ERP records
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
        "title": menu["title"],
        "embedding": embedding
    })


# ------------------------------------------------
# Test every query
# ------------------------------------------------

for query in queries:

    print()
    print("=" * 60)
    print("Query:")
    print(query)
    print("-" * 60)

    # Embed the question
    query_embedding = ollama.embed(
        model="nomic-embed-text:latest",
        input=query
    ).embeddings[0]

    results = []

    # Compare question with every ERP record
    for menu in menu_embeddings:

        score = cosine_similarity(
            query_embedding,
            menu["embedding"]
        )

        results.append({
            "title": menu["title"],
            "score": score
        })

    # Highest similarity first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Display results
    for result in results:

        print(
            f'{result["score"]:.4f}  {result["title"]}'
        )