from app.knowledge.chunking import _token_spans, approximate_token_count, chunk_document
from app.knowledge.loader import KnowledgeLoadError, load_knowledge_document


def _document(filename: str, text: str):
    return load_knowledge_document(
        filename,
        text.encode("utf-8"),
        document_id="doc-1",
        knowledge_base_id="kb-default",
        dataset_id="dataset-1",
    )


def test_loader_accepts_utf8_markdown_and_normalizes_text():
    doc = _document("口径.md", "# 销售\r\n\r\n销售额定义。\r\n\r\n\r\n")
    assert doc.title == "口径"
    assert doc.content == "# 销售\n\n销售额定义。"
    assert doc.dataset_id == "dataset-1"
    assert len(doc.content_hash) == 64


def test_loader_rejects_unsupported_extension_and_invalid_utf8():
    import pytest

    with pytest.raises(KnowledgeLoadError):
        load_knowledge_document("table.csv", b"a,b", document_id="d", knowledge_base_id="kb")
    with pytest.raises(KnowledgeLoadError):
        load_knowledge_document("notes.txt", b"\xff", document_id="d", knowledge_base_id="kb")


def test_markdown_chunking_preserves_heading_path_and_metadata():
    doc = _document("methods.md", "# 分析方法\n## 相关性\n\n相关性不代表因果关系。")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].section == "分析方法 > 相关性"
    assert chunks[0].knowledge_base_id == "kb-default"
    assert chunks[0].dataset_id == "dataset-1"
    assert chunks[0].language == "zh"


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    spans = _token_spans(text)
    return text[spans[max(0, len(spans) - overlap_tokens)][0]:]


def test_txt_long_english_document_splits_with_overlap_under_hard_limit():
    # Single newlines keep the text in one `.txt` section, so `_split_long_text` can
    # actually emit several chunks and apply the documented overlap.
    text = "\n".join([f"Paragraph {index}: sales profit margin analysis uses documented business definitions."
                      for index in range(90)])
    chunks = chunk_document(_document("methods.txt", text), target_tokens=40, overlap_tokens=8, max_tokens=60)
    assert len(chunks) > 1
    assert all(approximate_token_count(chunk.text) <= 60 for chunk in chunks)
    assert all(chunk.language in {"en", "mixed"} for chunk in chunks)
    assert chunks[0].chunk_index == 0
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    overlapped = 0
    for previous, current in zip(chunks, chunks[1:]):
        tail = _overlap_tail(previous.text, 8)
        if current.text.startswith(tail):
            overlapped += 1
            continue
        # The documented branch: overlap is dropped only to keep the hard bound.
        assert approximate_token_count(tail) + approximate_token_count(current.text) > 60
    assert overlapped >= 1


def test_overlap_is_dropped_when_it_would_exceed_the_hard_token_limit():
    text = "\n".join(["Sales profit margin analysis uses documented business definitions "
                      "and regional comparisons." for _ in range(60)])
    chunks = chunk_document(_document("methods.txt", text), target_tokens=50, overlap_tokens=20, max_tokens=60)
    assert len(chunks) > 1
    assert all(approximate_token_count(chunk.text) <= 60 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        tail = _overlap_tail(previous.text, 20)
        assert not current.text.startswith(tail)
        # Every pair skipped the overlap because carrying it would exceed max_tokens.
        assert approximate_token_count(tail) + approximate_token_count(current.text) > 60


def test_long_chinese_sentence_is_split_to_maximum():
    text = "销售利润分析方法" * 500
    chunks = chunk_document(_document("long.txt", text), target_tokens=100, overlap_tokens=10, max_tokens=120)
    assert len(chunks) > 1
    assert all(approximate_token_count(chunk.text) <= 120 for chunk in chunks)


def test_chinese_sentences_split_near_target_size():
    text = "。".join(["销售利润分析方法用于业务指标说明" for _ in range(20)])
    chunks = chunk_document(_document("sentences.txt", text), target_tokens=50, overlap_tokens=5, max_tokens=70)
    assert len(chunks) > 1
    assert all(approximate_token_count(chunk.text) <= 70 for chunk in chunks)


def test_unbroken_english_text_still_obeys_hard_token_limit():
    text = "businessanalysis" * 1_000
    chunks = chunk_document(_document("unbroken.txt", text), target_tokens=100, overlap_tokens=10, max_tokens=120)
    assert len(chunks) > 1
    assert all(approximate_token_count(chunk.text) <= 120 for chunk in chunks)
