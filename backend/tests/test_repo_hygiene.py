from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gitignore_blocks_local_secret_files() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "backend/.env" in gitignore
    assert "frontend/.env" in gitignore
    assert "!.env.example" in gitignore
    assert "!backend/.env.example" in gitignore


def test_gitignore_blocks_runtime_data_but_keeps_gitkeep_placeholders() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "backend/data/**" in gitignore
    assert "data/**" in gitignore
    assert "!backend/data/**/.gitkeep" in gitignore
    assert "!data/**/.gitkeep" in gitignore
