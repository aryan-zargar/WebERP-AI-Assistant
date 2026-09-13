
import json
import os

import ollama


# ============================================================
# Configuration
# ============================================================

MENUS_FILE = "knowledge/Menus.json"
EMBEDDINGS_FILE = "data/embeddings.json"

EMBEDDING_MODEL = "nomic-embed-text:latest"


# ============================================================
# Persian text normalization
# ============================================================

def normalize_text(text: str) -> str:

    if not text:
        return ""

    text = str(text).strip()

    # Arabic -> Persian characters

    text = text.replace("ي", "ی")
    text = text.replace("ى", "ی")
    text = text.replace("ك", "ک")

    # Normalize Arabic/Persian punctuation

    text = text.replace("،", " ")
    text = text.replace("؛", " ")
    text = text.replace("؟", " ")

    return text


# ============================================================
# Create embedding text
# ============================================================

def menu_to_embedding_text(menu: dict):

    title = normalize_text(
        menu.get("title", "")
    )

    menu_path = normalize_text(
        menu.get("menu_path", "")
    )

    description = normalize_text(
        menu.get("description") or ""
    )


    parts = [
        f"عنوان: {title}",
        f"مسیر منو: {menu_path}"
    ]


    if description:

        parts.append(
            f"توضیحات: {description}"
        )


    return "\n".join(parts)


# ============================================================
# Main
# ============================================================

def main():

    print("Loading ERP menu knowledge...")


    if not os.path.exists(MENUS_FILE):

        raise RuntimeError(
            f"""
Menus file not found:

{MENUS_FILE}
"""
        )


    with open(
        MENUS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        menus = json.load(file)


    if not isinstance(menus, list):

        raise RuntimeError(
            "Menus.json must contain a JSON array."
        )


    print(
        f"Found {len(menus)} ERP menu records."
    )


    os.makedirs(
        "data",
        exist_ok=True
    )


    embeddings = []


    print()
    print("Creating embeddings...")
    print()


    for index, menu in enumerate(
        menus,
        start=1
    ):

        menu_id = str(
            menu["id"]
        )

        title = normalize_text(
            menu.get("title", "")
        )


        print(
            f"[{index}/{len(menus)}] "
            f"{menu_id} - {title}"
        )


        text = menu_to_embedding_text(
            menu
        )


        result = ollama.embed(
            model=EMBEDDING_MODEL,
            input=text
        )


        embedding = result.embeddings[0]


        embeddings.append(
            {
                "id": menu_id,
                "embedding": embedding
            }
        )


    output = {
        "model": EMBEDDING_MODEL,
        "source": MENUS_FILE,
        "records": embeddings
    }


    with open(
        EMBEDDINGS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False
        )


    print()
    print("================================")
    print("Embeddings created successfully.")
    print(
        f"Records: {len(embeddings)}"
    )
    print(
        f"Model: {EMBEDDING_MODEL}"
    )
    print(
        f"Saved to: {EMBEDDINGS_FILE}"
    )
    print("================================")


if __name__ == "__main__":
    main()

