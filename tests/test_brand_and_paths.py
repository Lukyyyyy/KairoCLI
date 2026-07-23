from pathlib import Path

from kairocli.brand import API_KEY_HEADER, ENV_PREFIX, PRODUCT_NAME, PROJECT_MEMORY_FILE
from kairocli.paths import KairoPaths


def test_brand_contract() -> None:
    assert PRODUCT_NAME == "Kairo CLI"
    assert ENV_PREFIX == "KAIROCLI_"
    assert PROJECT_MEMORY_FILE == "KAIRO.md"
    assert API_KEY_HEADER == "X-Kairo-CLI-API-Key"


def test_paths_are_isolated(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    assert paths.user_dir == (tmp_path / "home" / ".kairocli").resolve()
    assert paths.project_dir == (tmp_path / "work" / ".kairocli").resolve()
    assert paths.memory_file.parts[-3:] == (".kairocli", "memory", "long_term_memory.json")


def test_no_legacy_brand_tokens() -> None:
    root = Path(__file__).parents[1]
    fragments = ["Pai" + "CLI", "pai" + "cli", "." + "pai" + "cli", "PAI" + "CLI_"]
    for path in root.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(token in content for token in fragments), path
