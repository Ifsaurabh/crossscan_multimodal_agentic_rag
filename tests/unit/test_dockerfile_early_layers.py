"""The model-baking layers of the Dockerfiles copy only a few source files before the rest of `src/`, so the slow layer stays cached.
Those few files must not import another module of ours that is not copied with them: the build would fail with an ImportError
that no unit test sees (it happened: query_guardrail started importing langfuse_client)."""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OUR_PACKAGES = ("shared", "retrieval", "ingestion", "evaluation")


def early_files(dockerfile: str) -> set:
    """The source files copied one by one (COPY src/... ./src/...) before the first copy of a whole folder."""
    files = set()
    for line in (ROOT / dockerfile).read_text(encoding="utf-8").splitlines():
        match = re.match(r"COPY\s+(.+?)\s+\./src/", line.strip())
        if not match:
            continue
        sources = match.group(1).split()
        if any(not source.endswith(".py") for source in sources):
            break  # a whole folder: the early layers are over
        files.update(source.removeprefix("src/") for source in sources)
    return files


def our_top_level_imports(path: Path) -> set:
    """The modules of ours that a file imports when it is imported (not the ones imported lazily inside a function)."""
    found = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            package, _, rest = node.module.partition(".")
            if package in OUR_PACKAGES:
                if rest:
                    found.add(f"{package}/{rest.replace('.', '/')}.py")
                else:
                    found.update(f"{package}/{alias.name}.py" for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                package, _, rest = alias.name.partition(".")
                if package in OUR_PACKAGES and rest:
                    found.add(f"{package}/{rest.replace('.', '/')}.py")
    return found


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.worker"])
def test_every_file_copied_early_finds_what_it_imports_in_the_same_layer(dockerfile):
    copied = early_files(dockerfile)
    assert copied, "the early COPY lines were not found"

    for name in sorted(copied):
        if name.endswith("__init__.py"):
            continue
        missing = our_top_level_imports(ROOT / "src" / name) - copied
        assert not missing, f"{dockerfile}: src/{name} imports {sorted(missing)}, which the early layer does not copy"


def test_the_parser_sees_the_imports_that_broke_the_build():
    assert "shared/langfuse_client.py" in our_top_level_imports(ROOT / "src" / "shared" / "query_guardrail.py")
    assert "shared/db.py" in our_top_level_imports(ROOT / "src" / "retrieval" / "retrieval_config.py")
