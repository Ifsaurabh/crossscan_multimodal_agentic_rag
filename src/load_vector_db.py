import json
from pathlib import Path

from db import get_connection, SCHEMA_NAME
from embedding_config import EMBEDDING_VERSION

CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
TEXT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "embeddings" / "text"
IMAGE_EMBEDDINGS_PATH = Path(__file__).parent.parent / "data" / "embeddings" / "image_embeddings.json"
DOMAIN_MAP_PATH = Path(__file__).parent.parent / "data" / "domain_classification.json"


def load_domain_map():
    return {d["source_pdf"]: d["domain"] for d in json.loads(DOMAIN_MAP_PATH.read_text(encoding="utf-8"))}


def load_text(conn, domain_map):
    chunk_files = sorted(CHUNKS_DIR.glob("*.json"))
    parent_count = 0
    chunk_count = 0

    for chunk_path in chunk_files:
        doc = json.loads(chunk_path.read_text(encoding="utf-8"))
        source_pdf = doc["source_pdf"]
        domain = domain_map.get(source_pdf, "unclassified")

        embedding_path = TEXT_EMBEDDINGS_DIR / chunk_path.name
        embeddings_by_chunk_id = {}
        if embedding_path.exists():
            emb_doc = json.loads(embedding_path.read_text(encoding="utf-8"))
            embeddings_by_chunk_id = {r["chunk_id"]: r["embedding"] for r in emb_doc["records"]}

        for parent in doc["parents"]:
            conn.execute(
                f"""INSERT INTO {SCHEMA_NAME}.text_parents
                    (parent_id, source_pdf, section, page_start, page_end, domain, text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (parent_id) DO UPDATE SET
                        section = EXCLUDED.section, domain = EXCLUDED.domain, text = EXCLUDED.text""",
                (parent["chunk_id"], source_pdf, parent["section"], parent["page_start"],
                 parent["page_end"], domain, parent["text"]),
            )
            parent_count += 1

            for child in parent["children"]:
                embedding = embeddings_by_chunk_id.get(child["chunk_id"])
                if embedding is None:
                    continue

                conn.execute(
                    f"""INSERT INTO {SCHEMA_NAME}.text_chunks
                        (chunk_id, parent_id, source_pdf, section, page_start, page_end,
                         domain, text, embedding_version, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            domain = EXCLUDED.domain, text = EXCLUDED.text,
                            embedding_version = EXCLUDED.embedding_version, embedding = EXCLUDED.embedding""",
                    (child["chunk_id"], child["parent_id"], source_pdf, child["section"],
                     child["page_start"], child["page_end"], domain, child["text"],
                     EMBEDDING_VERSION, embedding),
                )
                chunk_count += 1

    return parent_count, chunk_count


def load_images(conn, domain_map):
    if not IMAGE_EMBEDDINGS_PATH.exists():
        return 0

    data = json.loads(IMAGE_EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    count = 0

    for record in data["records"]:
        domain = domain_map.get(record["source_pdf"], "unclassified")
        conn.execute(
            f"""INSERT INTO {SCHEMA_NAME}.images
                (image_file, source_pdf, page, domain, embedding_version, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (image_file) DO UPDATE SET
                    domain = EXCLUDED.domain, embedding_version = EXCLUDED.embedding_version,
                    embedding = EXCLUDED.embedding""",
            (record["image_file"], record["source_pdf"], record["page"], domain,
             EMBEDDING_VERSION, record["embedding"]),
        )
        count += 1

    return count


def load_vector_db():
    domain_map = load_domain_map()
    conn = get_connection()

    parent_count, chunk_count = load_text(conn, domain_map)
    image_count = load_images(conn, domain_map)

    conn.commit()
    conn.close()

    print(f"Loaded {parent_count} parents, {chunk_count} text chunks, {image_count} images into pgvector.")


if __name__ == "__main__":
    load_vector_db()
