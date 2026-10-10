from __future__ import annotations

import hashlib
import re

from app.models.knowledge import KnowledgeChunk, KnowledgeDocument

TARGET_TOKENS = 450
OVERLAP_TOKENS = 60
MAX_TOKENS = 700

# One CJK character, or one Latin/numeric word, is a practical lightweight
# token approximation for chunk boundaries without a model-specific tokenizer.
_TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*", re.UNICODE)
_HEADING_PATTERN = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？.!?])\s*|\n+")
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


def approximate_token_count(text: str) -> int:
    return len(_token_spans(text))


def _token_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _TOKEN_PATTERN.finditer(text):
        if _CJK_PATTERN.fullmatch(match.group(0)):
            spans.append((match.start(), match.end()))
            continue
        # Approximate common English BPE behavior at about four characters per token.
        for offset in range(match.start(), match.end(), 4):
            spans.append((offset, min(offset + 4, match.end())))
    return spans


def _language(text: str) -> str:
    has_cjk = bool(_CJK_PATTERN.search(text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "unknown"


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        body_lines = [line for line in current_lines if not _HEADING_PATTERN.match(line)]
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((current_heading, body))

    for line in text.splitlines():
        match = _HEADING_PATTERN.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_heading = " > ".join(item[1] for item in heading_stack)
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()
    return sections or [("", text.strip())]


def _split_long_text(text: str, *, target_tokens: int, overlap_tokens: int, max_tokens: int) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if not sentences:
        return []
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if approximate_token_count(sentence) > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            matches = _token_spans(sentence)
            start = 0
            step = max(1, max_tokens - overlap_tokens)
            while start < len(matches):
                end = min(len(matches), start + max_tokens)
                left = matches[start][0]
                right = matches[end - 1][1]
                piece = sentence[left:right].strip()
                if piece:
                    chunks.append(piece)
                if end == len(matches):
                    break
                start += step
            continue
        candidate = f"{current}\n{sentence}".strip() if current else sentence
        if current and approximate_token_count(candidate) > target_tokens:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)

    if overlap_tokens <= 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        previous_tokens = _token_spans(chunks[len(overlapped) - 1])
        if previous_tokens:
            tail = chunks[len(overlapped) - 1][previous_tokens[max(0, len(previous_tokens) - overlap_tokens)][0]:]
            combined = f"{tail}\n{chunk}".strip()
            # Preserve the hard bound; do not force overlap for near-max chunks.
            if approximate_token_count(combined) <= max_tokens:
                chunk = combined
        overlapped.append(chunk)
    return overlapped


def chunk_document(
    document: KnowledgeDocument,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[KnowledgeChunk]:
    if not (0 < target_tokens <= max_tokens and 0 <= overlap_tokens < max_tokens):
        raise ValueError("Chunk size configuration is invalid.")
    result: list[KnowledgeChunk] = []
    markdown = document.source.lower().endswith(".md")
    sections = _markdown_sections(document.content) if markdown else [("", part.strip()) for part in re.split(r"\n\s*\n", document.content) if part.strip()]
    for section, body in sections:
        for text in _split_long_text(body, target_tokens=target_tokens, overlap_tokens=overlap_tokens, max_tokens=max_tokens):
            if approximate_token_count(text) > max_tokens:
                raise ValueError("Chunking produced text above the configured hard token limit.")
            index = len(result)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            result.append(KnowledgeChunk(
                chunk_id=f"{document.document_id}:{index}",
                document_id=document.document_id,
                knowledge_base_id=document.knowledge_base_id,
                dataset_id=document.dataset_id,
                title=document.title,
                source=document.source,
                section=section,
                chunk_index=index,
                content_hash=digest,
                language=_language(text),
                text=text,
            ))
    return result
