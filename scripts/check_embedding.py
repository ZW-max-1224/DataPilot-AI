"""Embedding Provider 自检工具（只读诊断，不修改任何配置）。

用途：在把一个 Embedding 服务商写进 .env 之前，先确认它真的可用、真的有意义。

用法示例
--------
# 1) 先进自检：不需要网络和 Key，验证本脚本的判断逻辑是否正确
python scripts/check_embedding.py --self-test

# 2) 用 .env 里已配置的 EMBEDDING_* 检查
python scripts/check_embedding.py

# 3) 在写入 .env 之前，直接试一个候选服务商
python scripts/check_embedding.py `
  --base-url https://api.siliconflow.cn/v1 `
  --api-key sk-xxxx `
  --model BAAI/bge-m3 `
  --dimensions 1024

退出码：全部通过 0，任一失败 1。脚本从不打印 API Key。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 两个语义相近的句子，和一个无关句子。用于验证向量"有意义"而不只是"有数字"。
NEAR_A = "8月销售额下降，需要分析原因。"
NEAR_B = "八月份销售业绩下滑，要找出原因。"
FAR = "今天天气很好，我想去公园散步。"


class FakeEmbedder:
    """自检用的确定性假向量：相近句子共享更多词，相似度应更高。不访问网络。"""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def create(self, *, model: str, input: Sequence[str]) -> Any:
        class Item:
            def __init__(self, index: int, embedding: list[float]) -> None:
                self.index = index
                self.embedding = embedding

        class Response:
            def __init__(self, data: list[Any]) -> None:
                self.data = data

        items = []
        for index, text in enumerate(input):
            vector = [0.0] * self.dimensions
            for char in text:
                vector[ord(char) % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            items.append(Item(index, [value / norm for value in vector]))
        return Response(items)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left)) or 1.0
    norm_right = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (norm_left * norm_right)


def build_embedder(args: argparse.Namespace) -> tuple[Callable[..., Any], str, str, int]:
    """返回 (embed 函数, base_url, model, dimensions)。"""
    if args.self_test:
        fake = FakeEmbedder()
        return fake.create, "(self-test)", "(fake)", 0

    from dotenv import dotenv_values
    from openai import OpenAI

    values = dict(dotenv_values(PROJECT_ROOT / ".env"))
    values.update(os.environ)
    base_url = (args.base_url or values.get("EMBEDDING_BASE_URL") or "").strip()
    api_key = (args.api_key or values.get("EMBEDDING_API_KEY") or "").strip()
    model = (args.model or values.get("EMBEDDING_MODEL") or "").strip()
    dimensions = args.dimensions or int(values.get("EMBEDDING_DIMENSIONS") or 0)

    missing = [name for name, value in
               (("base_url", base_url), ("api_key", api_key), ("model", model)) if not value]
    if missing:
        print("缺少参数（可用命令行传入，或先配置 .env 的 EMBEDDING_*）：" + ", ".join(missing))
        raise SystemExit(2)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout, max_retries=0)
    return client.embeddings.create, base_url, model, dimensions


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding provider self-check")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API 根地址，通常以 /v1 结尾")
    parser.add_argument("--api-key", default=None, help="API Key（不会打印）")
    parser.add_argument("--model", default=None, help="Embedding 模型名，如 BAAI/bge-m3")
    parser.add_argument("--dimensions", type=int, default=None, help="期望的向量维度，如 1024")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--self-test", action="store_true", help="不联网，用假向量验证本脚本逻辑")
    args = parser.parse_args()

    try:
        embed, base_url, model, expected = build_embedder(args)
    except SystemExit as exc:
        return int(exc.code or 2)

    print(f"provider : {base_url}")
    print(f"model    : {model}")
    print(f"期望维度 : {expected or '(不校验)'}")
    print("-" * 66)

    failures = 0

    # 检查 1：端点可达 + 单条向量
    try:
        single = list(embed(model=model, input=[NEAR_A]).data[0].embedding)
        actual = len(single)
        print(f"[1/5] 端点可达 + 单条向量 .......... 通过（维度 {actual}）")
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        print(f"[1/5] 端点可达 + 单条向量 .......... 失败：{type(exc).__name__} status={status}")
        if status == 404:
            print("      404 表示服务商没有 /embeddings 这个端点（而不是模型名写错）。")
            print("      换一个支持 Embedding 的服务商，或改用本地向量模型。")
        elif status == 401:
            print("      401 表示 Key 无效或没有 Embedding 权限。")
        elif status == 400:
            print("      400 通常表示模型名不对：请到服务商控制台核对模型 ID。")
        return 1

    # 检查 2：维度与配置一致（维度不符会让 Qdrant 无法使用）
    if expected and actual != expected:
        print(f"[2/5] 维度与配置一致 ............... 失败：实际 {actual}，配置 {expected}")
        print(f"      请把 EMBEDDING_DIMENSIONS 改为 {actual}，或换一个返回 {expected} 维的模型。")
        failures += 1
    else:
        print(f"[2/5] 维度与配置一致 ............... 通过")

    # 检查 3：批量请求（项目按批索引知识文档，最多 128 条）
    try:
        batch = embed(model=model, input=[NEAR_A, NEAR_B, FAR]).data
        ok = len(batch) == 3 and abs(batch[0].embedding[0] - single[0]) < 1e-6
        print(f"[3/5] 批量 3 条 + 顺序稳定 ......... {'通过' if ok else '失败'}")
        if not ok:
            failures += 1
    except Exception as exc:
        print(f"[3/5] 批量 3 条 + 顺序稳定 ......... 失败：{type(exc).__name__}")
        failures += 1

    # 检查 4：确定性（同一文本两次必须完全一致）
    try:
        again = list(embed(model=model, input=[NEAR_A]).data[0].embedding)
        ok = all(abs(a - b) < 1e-9 for a, b in zip(single, again))
        print(f"[4/5] 同一文本结果确定 ............. {'通过' if ok else '失败（向量不稳定，检索会抖动）'}")
        if not ok:
            failures += 1
    except Exception as exc:
        print(f"[4/5] 同一文本结果确定 ............. 失败：{type(exc).__name__}")
        failures += 1

    # 检查 5：语义有效性（相近句的相似度必须高于无关句）—— 这条最关键
    try:
        vectors = embed(model=model, input=[NEAR_A, NEAR_B, FAR]).data
        near = cosine(list(vectors[0].embedding), list(vectors[1].embedding))
        far = cosine(list(vectors[0].embedding), list(vectors[2].embedding))
        ok = near > far
        print(f"[5/5] 语义有效（近义 > 无关）....... {'通过' if ok else '失败'} "
              f"(near={near:.4f} far={far:.4f})")
        if not ok:
            failures += 1
            print("      相似度关系不成立，说明这个端点返回的可能是随机或占位向量，")
            print("      用它做 RAG 检索不会有意义。")
    except Exception as exc:
        print(f"[5/5] 语义有效（近义 > 无关）....... 失败：{type(exc).__name__}")
        failures += 1

    print("-" * 66)
    if failures:
        print(f"结果：{failures} 项未通过，不要把这个 provider 写进 .env。")
        return 1

    print("结果：全部通过。可以把下面这段写进 .env：\n")
    print("EMBEDDING_PROVIDER=openai_compatible")
    print(f"EMBEDDING_MODEL={model}")
    print("EMBEDDING_API_KEY=<你的 Key>")
    print(f"EMBEDDING_BASE_URL={base_url}")
    print(f"EMBEDDING_DIMENSIONS={actual}")
    print("\n注意：维度必须与 Qdrant collection 一致；换维度后需要新建 collection 或清空 data/qdrant 重建。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
