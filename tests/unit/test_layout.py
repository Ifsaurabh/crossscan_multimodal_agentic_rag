"""The layout of src/: four folders, and which folder may import which.

    shared/      used by both images (the worker and the app)
    ingestion/   the worker image
    retrieval/   the app image
    evaluation/  offline evaluation: in neither image (GitHub Actions or a laptop)

Because each image is built from its own folder plus shared/, an import that crosses these rules would
break that image at run time. These tests make it a test failure instead."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

ALLOWED = {
    "shared": {"shared"},
    "ingestion": {"ingestion", "shared"},
    "retrieval": {"retrieval", "shared"},
    "evaluation": {"evaluation", "retrieval", "shared"},  # the evaluation may run the app's code; nothing runs the evaluation
}
FOLDERS = set(ALLOWED)


def module_files():
    return sorted((folder, path) for folder in ALLOWED for path in (SRC / folder).glob("*.py") if path.name != "__init__.py")


def imports_of(path: Path):
    """(line, top-level module name) for every import in the file, including those inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module.split(".")[0]


ALL_MODULE_NAMES = {path.stem for _, path in module_files()}


def test_src_holds_only_the_four_folders():
    assert sorted(p.name for p in SRC.iterdir() if p.is_file() and p.suffix == ".py") == []
    folders = {p.name for p in SRC.iterdir() if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".")}
    assert folders == FOLDERS  # (hidden folders such as .deepeval, made by a library, do not count)


@pytest.mark.parametrize("folder", sorted(FOLDERS))
def test_every_folder_is_a_package(folder):
    assert (SRC / folder / "__init__.py").exists()


def test_a_module_name_is_used_only_once_across_the_folders():
    names = [path.stem for _, path in module_files()]
    assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})


@pytest.mark.parametrize("folder,path", module_files(), ids=lambda value: value.name if isinstance(value, Path) else value)
def test_a_module_imports_only_from_the_folders_it_may_use(folder, path):
    broken = [
        f"{path.name}:{line} imports {target}"
        for line, target in imports_of(path)
        if target in FOLDERS and target not in ALLOWED[folder]
    ]
    assert not broken, f"{folder}/ may import only {sorted(ALLOWED[folder])}: " + "; ".join(broken)


@pytest.mark.parametrize("folder,path", module_files(), ids=lambda value: value.name if isinstance(value, Path) else value)
def test_a_module_never_imports_another_by_its_bare_old_name(folder, path):
    """After the move every import names its folder (`from shared import db`); a bare `import db` would still
    work on a machine with a stale path and fail inside the image."""
    bare = [f"{path.name}:{line} imports {name}" for line, name in imports_of(path) if name in ALL_MODULE_NAMES]
    assert not bare, "; ".join(bare)


def test_the_app_does_not_depend_on_the_evaluation_folder():
    """online_eval and online_report run inside the app, so nothing in retrieval/ or shared/ may reach evaluation/."""
    for folder in ("retrieval", "shared", "ingestion"):
        for line, target in (pair for f, path in module_files() if f == folder for pair in imports_of(path)):
            assert target != "evaluation", f"{folder}/ imports evaluation (line {line})"
