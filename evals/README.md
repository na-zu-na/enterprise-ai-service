# 检索效果评测

本目录使用同一份人工标注问题，对比以下三套检索方案：

1. 仅向量检索（Vector Only）
2. 向量检索 + BM25 + RRF
3. 向量检索 + BM25 + RRF + Reranker

## 运行评测

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m evals.retrieval_eval
```

也可以显式指定参数：

```powershell
.\.venv\Scripts\python.exe -m evals.retrieval_eval `
  --dataset evals/datasets/retrieval_eval.json `
  --candidate-k 30 `
  --rrf-top-k 20 `
  --top-k 5 `
  --output evals/results/retrieval_eval_results.json
```

参数含义：

- `candidate-k`：向量检索和 BM25 各自召回的候选数量。
- `rrf-top-k`：RRF 融合后送入 Reranker 的候选数量。
- `top-k`：三套方案最终参与指标计算的结果数量。
- `output`：逐题评测结果和汇总指标的 JSON 输出路径。

运行前需要确保 PostgreSQL 和 Elasticsearch 可用。第一次执行 Reranker
时，可能还需要加载配置的模型。评测模式下，Reranker 加载或推理失败会直接
终止评测，不会静默回退到 RRF 排序，避免产生无效的 Reranker 对比结果。

## 指标口径

当前计算以下三个指标：

- `Hit@K`：Top K 中是否至少出现一个正确 Chunk。
- `Recall@K`：Top K 命中的正确 Chunk 数除以全部正确 Chunk 数。
- `MRR`：第一个正确 Chunk 排名倒数的平均值。

无答案问题会保留在逐题 JSON 明细中，但不参与 Hit@K、Recall@K 和 MRR
汇总，因为不存在相关 Chunk 时这些指标没有定义。

评测启动前还会检查人工标注的 Chunk 是否真实存在，以及它是否属于标注的
Document 和 Knowledge Base。知识库权限由 Spring 层负责，本评测不验证用户
权限逻辑。

## 输出结果

控制台会输出三套方案的总指标、分类指标，以及 Reranker 相对于
Hybrid + RRF 的指标变化。默认的完整明细文件为：

```text
evals/results/retrieval_eval_results.json
```

## 运行自动化测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
