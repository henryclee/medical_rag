"""Tests for medical_rag.data (MedQA loader and StatPearls chunker).

The MedQA-USMLE dataset is small and fast, so its loader is tested live
against Hugging Face (relies on the datasets cache after the first run).
The StatPearls corpus requires a ~1.9GB download, so only its pure-Python
extraction logic is tested here, against a small in-memory fixture, with
no network access.
"""

from pathlib import Path

from medical_rag.data.load_medqa import MedQAQuestion, load_medqa
from medical_rag.data.load_statpearls import StatPearlsChunk, _extract_article_snippets

FIXTURE_NXML = """\
<book-part>
  <book-part-meta>
    <title-group>
      <title>Test Article</title>
    </title-group>
  </book-part-meta>
  <body>
    <sec>
      <title>Introduction</title>
      <p>This is a short test paragraph used to validate the StatPearls chunker.</p>
    </sec>
  </body>
</book-part>
"""


def test_load_medqa_schema_and_count():
    questions = load_medqa(split="test")
    assert len(questions) == 1273
    first = questions[0]
    assert isinstance(first, MedQAQuestion)
    assert first.question
    assert set(first.options) <= {"A", "B", "C", "D"}
    assert first.answer_idx in first.options


def test_extract_article_snippets(tmp_path: Path):
    fpath = tmp_path / "test.nxml"
    fpath.write_text(FIXTURE_NXML)

    snippets = _extract_article_snippets(fpath)

    assert len(snippets) == 1
    snippet = snippets[0]
    assert snippet["id"] == "test_0"
    assert snippet["title"] == "Test Article -- Introduction"
    assert "test paragraph" in snippet["content"]

    chunk = StatPearlsChunk(chunk_id=snippet["id"], title=snippet["title"], content=snippet["content"])
    assert chunk.source == "statpearls"
