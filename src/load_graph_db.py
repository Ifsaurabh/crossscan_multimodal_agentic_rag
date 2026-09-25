import json
from pathlib import Path

from graph_db import close_driver, get_driver

DOMAIN_MAP_PATH = Path(__file__).parent.parent / "data" / "domain_classification.json"
ENTITIES_PATH = Path(__file__).parent.parent / "data" / "entities.json"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
IMAGES_METADATA_PATH = Path(__file__).parent.parent / "data" / "images" / "metadata.json"
GUARDED_DIR = Path(__file__).parent.parent / "data" / "guarded"


def load_papers(session, domain_map):
    for source_pdf, domain in domain_map.items():
        session.run(
            "MERGE (p:Paper {source_pdf: $source_pdf}) SET p.domain = $domain",
            source_pdf=source_pdf, domain=domain,
        )
    return len(domain_map)


def load_entities(session, entities):
    method_count = dataset_count = metric_count = baseline_count = 0

    for source_pdf, data in entities.items():
        verified = data["verified"]

        for method in verified.get("methods", []):
            session.run(
                """MERGE (m:Method {name: $name})
                   WITH m
                   MATCH (p:Paper {source_pdf: $source_pdf})
                   MERGE (p)-[:USES_METHOD]->(m)""",
                name=method, source_pdf=source_pdf,
            )
            method_count += 1

        for dataset in verified.get("datasets", []):
            session.run(
                """MERGE (d:Dataset {name: $name})
                   WITH d
                   MATCH (p:Paper {source_pdf: $source_pdf})
                   MERGE (p)-[:USES_DATASET]->(d)""",
                name=dataset, source_pdf=source_pdf,
            )
            dataset_count += 1

        for metric in verified.get("metrics", []):
            session.run(
                """MERGE (m:Metric {name: $name})
                   WITH m
                   MATCH (p:Paper {source_pdf: $source_pdf})
                   MERGE (p)-[:EVALUATED_WITH]->(m)""",
                name=metric, source_pdf=source_pdf,
            )
            metric_count += 1

        for baseline in verified.get("baselines", []):
            session.run(
                """MERGE (b:Baseline {name: $name})
                   WITH b
                   MATCH (p:Paper {source_pdf: $source_pdf})
                   MERGE (p)-[:COMPARED_TO]->(b)""",
                name=baseline, source_pdf=source_pdf,
            )
            baseline_count += 1

    return method_count, dataset_count, metric_count, baseline_count


def load_sections(session):
    section_count = 0
    all_sections = []

    for chunk_path in sorted(CHUNKS_DIR.glob("*.json")):
        doc = json.loads(chunk_path.read_text(encoding="utf-8"))
        source_pdf = doc["source_pdf"]

        for parent in doc["parents"]:
            session.run(
                """MERGE (s:Section {chunk_id: $chunk_id})
                   SET s.heading = $heading, s.page_start = $page_start, s.page_end = $page_end
                   WITH s
                   MATCH (p:Paper {source_pdf: $source_pdf})
                   MERGE (p)-[:HAS_SECTION]->(s)""",
                chunk_id=parent["chunk_id"], heading=parent["section"],
                page_start=parent["page_start"], page_end=parent["page_end"],
                source_pdf=source_pdf,
            )
            section_count += 1
            all_sections.append({
                "chunk_id": parent["chunk_id"], "source_pdf": source_pdf,
                "page_start": parent["page_start"], "page_end": parent["page_end"],
            })

    return section_count, all_sections


def load_images(session, all_sections):
    metadata = json.loads(IMAGES_METADATA_PATH.read_text(encoding="utf-8"))
    image_count = 0
    linked_count = 0

    sections_by_pdf = {}
    for s in all_sections:
        sections_by_pdf.setdefault(s["source_pdf"], []).append(s)

    for item in metadata:
        source_pdf = item["source_pdf"]
        page = item["page"]

        session.run(
            """MERGE (i:Image {image_file: $image_file})
               SET i.page = $page
               WITH i
               MATCH (p:Paper {source_pdf: $source_pdf})
               MERGE (i)-[:APPEARS_IN]->(p)""",
            image_file=item["image_file"], page=page, source_pdf=source_pdf,
        )
        image_count += 1

        matching_section = next(
            (s for s in sections_by_pdf.get(source_pdf, [])
             if s["page_start"] is not None and s["page_start"] <= page <= s["page_end"]),
            None,
        )
        if matching_section:
            session.run(
                """MATCH (i:Image {image_file: $image_file})
                   MATCH (s:Section {chunk_id: $chunk_id})
                   MERGE (i)-[:NEAR_SECTION]->(s)""",
                image_file=item["image_file"], chunk_id=matching_section["chunk_id"],
            )
            linked_count += 1

    return image_count, linked_count


def load_tables(session, all_sections):
    """Tables have no ID from earlier stages (unlike sections/images) - assign
    one here, scoped to the document, stable across re-runs since a document's
    table order doesn't change between runs of the same data."""
    table_count = 0
    linked_count = 0

    sections_by_pdf = {}
    for s in all_sections:
        sections_by_pdf.setdefault(s["source_pdf"], []).append(s)

    for guarded_path in sorted(GUARDED_DIR.glob("*.json")):
        doc = json.loads(guarded_path.read_text(encoding="utf-8"))
        source_pdf = doc["source_pdf"]

        for i, table in enumerate(doc.get("tables", []), start=1):
            table_id = f"{source_pdf}_table{i}"
            page = table["page"]

            # Table nodes store their own text (unlike Section/Image, which
            # point back to Postgres/disk) - tables have no other home.
            session.run(
                """MERGE (t:Table {table_id: $table_id})
                   SET t.text = $text, t.page = $page
                   WITH t
                   MATCH (p:Paper {source_pdf: $source_pdf})
                   MERGE (t)-[:APPEARS_IN]->(p)""",
                table_id=table_id, text=table["text"], page=page, source_pdf=source_pdf,
            )
            table_count += 1

            matching_section = next(
                (s for s in sections_by_pdf.get(source_pdf, [])
                 if s["page_start"] is not None and page is not None
                 and s["page_start"] <= page <= s["page_end"]),
                None,
            )
            if matching_section:
                session.run(
                    """MATCH (t:Table {table_id: $table_id})
                       MATCH (s:Section {chunk_id: $chunk_id})
                       MERGE (t)-[:NEAR_SECTION]->(s)""",
                    table_id=table_id, chunk_id=matching_section["chunk_id"],
                )
                linked_count += 1

    return table_count, linked_count


def load_graph_db():
    domain_map = {d["source_pdf"]: d["domain"] for d in json.loads(DOMAIN_MAP_PATH.read_text(encoding="utf-8"))}
    entities = json.loads(ENTITIES_PATH.read_text(encoding="utf-8"))

    driver = get_driver()
    with driver.session() as session:
        paper_count = load_papers(session, domain_map)
        print(f"Loaded {paper_count} Paper nodes")

        method_count, dataset_count, metric_count, baseline_count = load_entities(session, entities)
        print(f"Loaded {method_count} USES_METHOD, {dataset_count} USES_DATASET, "
              f"{metric_count} EVALUATED_WITH, {baseline_count} COMPARED_TO relationships")

        section_count, all_sections = load_sections(session)
        print(f"Loaded {section_count} Section nodes")

        image_count, image_linked_count = load_images(session, all_sections)
        print(f"Loaded {image_count} Image nodes, {image_linked_count} linked to a Section via page overlap")

        table_count, table_linked_count = load_tables(session, all_sections)
        print(f"Loaded {table_count} Table nodes, {table_linked_count} linked to a Section via page overlap")

    close_driver()
    print("\nGraph load complete.")


if __name__ == "__main__":
    load_graph_db()
