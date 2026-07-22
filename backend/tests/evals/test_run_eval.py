from pathlib import Path

import pytest

from evals.models import EvalCase
from evals.run_eval import live_preflight_errors, require_live_profile_ready


def test_live_preflight_reports_missing_key_and_corpus(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("evals.run_eval.settings.OPENAI_API_KEY", None)
    monkeypatch.setattr("evals.run_eval.settings.TAVILY_API_KEY", None)
    monkeypatch.setattr("evals.run_eval.settings.CHROMA_DIR", str(tmp_path / "missing-chroma"))

    errors = live_preflight_errors([EvalCase(id="fresh", question="Latest?", requires_fresh_context=True)], ["full_agentic_rag"])

    assert "OPENAI_API_KEY is required for live LLM and embedding calls." in errors
    assert any(error.startswith("CHROMA_DIR does not exist:") for error in errors)
    assert any("fresh-context cases will need external search" in error for error in errors)


def test_live_preflight_accepts_configured_live_environment(monkeypatch, tmp_path) -> None:
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "chroma.sqlite3").write_text("fixture", encoding="utf-8")
    monkeypatch.setattr("evals.run_eval.settings.OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("evals.run_eval.settings.TAVILY_API_KEY", "test-tavily")
    monkeypatch.setattr("evals.run_eval.settings.CHROMA_DIR", str(chroma_dir))

    assert live_preflight_errors([EvalCase(id="factual", question="What?")], ["vector_only_rag"]) == []


def test_live_preflight_exits_with_actionable_message(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("evals.run_eval.settings.OPENAI_API_KEY", None)
    monkeypatch.setattr("evals.run_eval.settings.CHROMA_DIR", str(Path(tmp_path) / "missing-chroma"))

    with pytest.raises(SystemExit) as exc_info:
        require_live_profile_ready([EvalCase(id="factual", question="What?")], ["vector_only_rag"])

    message = str(exc_info.value)
    assert "Live evaluation preflight failed." in message
    assert "--profile offline_fixture" in message
