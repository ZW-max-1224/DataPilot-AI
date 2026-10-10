"""RAG 端到端自检工具（只读诊断 + 索引内置知识）。

用途：在配置好 Embedding Provider 之后，确认"知识库真的能检索到对的东西"。
与 scripts/check_embedding.py 配套：先确认 Provider 可用，再用本脚本确认检索可用。

用法
----
# 1) 索引内置知识文档并做命中 / 排序检查（推荐第一次这样跑）
python scripts/check_rag.py

# 2) 复用已有索引，不重新索引
python scripts/check_rag.py --skip-index

# 3) 换自己的问题与期望来源
python scripts/check_rag.py --query "利润率怎么算？" --expect metrics_glossary.md

退出码：全部通过 0，任一失败 1。

关于"没有相关结果"的诚实说明
------------------------------
本项目**刻意不设置未经校准的相似度阈值**（见 README）。因此只要索引非空，
一个与知识库无关的问题通常仍会返回若干切片，status 是 ok 而不是 no_results。
`no_results` 只会在索引为空或过滤后无候选时出现。所以本脚本不把
"无关问题必须返回 no_results"当成通过条件——那样只会逼着代码去编造阈值。
脚本改为检查**排序区分度**：相关问题的最高分应明显高于无关问题。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import EmbeddingSettings, QdrantSettings  # noqa: E402
from app.knowledge.loader import load_knowledge_document  # noqa: E402
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError  # noqa: E402
from app.services.knowledge_service import KnowledgeService  # noqa: E402
from app.vectorstore.qdrant_store import QdrantVectorStore, VectorStoreError  # noqa: E402

DEFAULT_KNOWLEDGE_BASE_ID = "datapilot-default"
BUILTIN_DIR = PROJECT_ROOT / "data" / "knowledge"

# (问题, 期望命中的来源文件) —— 问题只用文档里真实存在的说法，不使用原文照抄的整句。
DEFAULT_CASES = (
    ("利润率应该怎么计算？", "metrics_glossary.md"),
    ("展示随时间变化的趋势应该用什么图？", "chart_selection.md"),
    ("相关系数能不能证明因果关系？", "analysis_methods.md"),
)
OFF_TOPIC_QUERY = "如何给猫咪洗澡和修剪指甲？"


def build_service() -> tuple[KnowledgeService, QdrantVectorStore]:
    embedding_settings = EmbeddingSettings.from_env()
    if not embedding_settings.is_configured:
        print("Embedding 未配置：", embedding_settings.configuration_problem or "请先配置 EMBEDDING_*")
        raise SystemExit(2)
    qdrant_settings = QdrantSettings.from_env()
    path = qdrant_settings.path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    store = QdrantVectorStore(
        path=path, collection_name=qdrant_settings.collection, dimensions=embedding_settings.dimensions,
    )
    available, code = store.availability()
    if not available:
        print(f"向量库不可用：{code}")
        print("如果提示维度不一致，请在页面点“重建知识库索引”，或清空 data/qdrant 后重试。")
        raise SystemExit(2)
    return KnowledgeService(EmbeddingService(embedding_settings), store), store


def index_builtin(service: KnowledgeService) -> int:
    sources = sorted([*BUILTIN_DIR.glob("*.md"), *BUILTIN_DIR.glob("*.txt")])
    if not sources:
        print(f"没有找到内置知识文档：{BUILTIN_DIR}")
        return 0
    indexed_chunks = 0
    for source_path in sources:
        document = load_knowledge_document(
            source_path.name, source_path.read_bytes(),
            document_id=f"builtin-{source_path.stem}"[:64],
            knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
            source_type="builtin",
        )
        result = service.index_document(document)
        if result.status == "indexed":
            indexed_chunks += result.chunk_count
            print(f"  已索引 {document.source}：{result.chunk_count} 个切片")
        else:
            print(f"  索引失败 {document.source}：{result.error_code}")
    return indexed_chunks


def search(service: KnowledgeService, query: str, top_k: int = 5):
    return service.search(query, knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID, dataset_id=None, top_k=top_k)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG end-to-end self-check")
    parser.add_argument("--query", default=None, help="自定义问题（配合 --expect 使用）")
    parser.add_argument("--expect", default=None, help="期望命中的来源文件名")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-index", action="store_true", help="复用已有索引，不重新索引内置文档")
    args = parser.parse_args()

    service, store = build_service()
    failures = 0

    if args.skip_index:
        print("[1/3] 索引 .......................... 跳过（--skip-index）")
    else:
        print("[1/3] 索引内置知识文档")
        chunks = index_builtin(service)
        if not chunks:
            print("      失败：没有任何切片被索引，后续检索无法验证")
            return 1
        print(f"      通过：共 {chunks} 个切片")

    cases = ((args.query, args.expect),) if args.query else DEFAULT_CASES
    if args.query and not args.expect:
        print("使用 --query 时必须同时给出 --expect")
        return 2

    print("[2/3] 命中检查（相关问题的期望来源是否在结果中）")
    hits = []
    for query, expected in cases:
        try:
            response = search(service, query, args.top_k)
        except Exception as exc:
            print(f"      失败：检索报错 {type(exc).__name__}")
            failures += 1
            continue
        if response.status != "ok" or not response.results:
            print(f"      失败：{query!r} -> status={response.status}（索引可能为空）")
            failures += 1
            continue
        sources = [item.source for item in response.results]
        rank = sources.index(expected) + 1 if expected in sources else None
        top = response.results[0]
        if rank is None:
            print(f"      失败：{query!r} 未命中 {expected}；实际来源={sources}")
            failures += 1
        else:
            hits.append(top.score)
            print(f"      通过：{query!r} -> {expected} 排名第 {rank}，top1={top.source} score={top.score:.4f}")

    print("[3/3] 排序区分度（无关问题的最高分必须低于相关问题的最高分）")
    try:
        off_topic = search(service, OFF_TOPIC_QUERY, args.top_k)
        off_score = off_topic.results[0].score if off_topic.results else 0.0
    except Exception as exc:
        print(f"      失败：无关问题检索报错 {type(exc).__name__}")
        return 1
    if hits:
        best_hit = max(hits)
        ok = best_hit > off_score
        print(f"      {'通过' if ok else '失败'}：相关问题最高分 {best_hit:.4f} vs 无关问题最高分 {off_score:.4f}")
        if not ok:
            failures += 1
            print("      无关问题得分不低于相关问题，说明向量区分度不足，检索结果不可信。")
    else:
        print("      跳过：上一步没有成功命中，无法比较")
        failures += 1

    print("-" * 66)
    if failures:
        print(f"结果：{failures} 项未通过。RAG 还不能算跑通。")
        return 1
    print("结果：全部通过。RAG 检索已真正跑通（索引、命中、排序区分度）。")
    print("注意：无相似度阈值是刻意设计，无关问题仍可能返回低分切片；")
    print("      判断答案是否可信要看 Sources 与切片内容，而不是只看 status=ok。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EmbeddingServiceError, VectorStoreError) as exc:
        print(f"环境错误：{type(exc).__name__}: {exc}")
        raise SystemExit(2) from exc
