from graph_db import close_driver, get_driver

CONSTRAINTS = [
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.source_pdf IS UNIQUE",
    "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT metric_name IF NOT EXISTS FOR (m:Metric) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT image_file IF NOT EXISTS FOR (i:Image) REQUIRE i.image_file IS UNIQUE",
    "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT table_id IF NOT EXISTS FOR (t:Table) REQUIRE t.table_id IS UNIQUE",
    "CREATE CONSTRAINT baseline_name IF NOT EXISTS FOR (b:Baseline) REQUIRE b.name IS UNIQUE",
]


def setup_graph_db():
    driver = get_driver()
    with driver.session() as session:
        for constraint in CONSTRAINTS:
            session.run(constraint)
    close_driver()
    print(f"Applied {len(CONSTRAINTS)} constraints (Paper, Method, Dataset, Metric, Image, Section, Table, Baseline).")


if __name__ == "__main__":
    setup_graph_db()
