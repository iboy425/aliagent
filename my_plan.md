# 三等奖目标优化计划

## 1. 当前基线

- 已完成 A 榜 baseline 端到端提交。
- 线上分数：
  - 总分：`0.2975`
  - 分类任务：`0.3777`
  - 推荐任务：`0.2173`
- 已提交预测包备份：
  - `framework/output/baseline_submitted/prediction_20260623_182014_score_0.2975.zip`

## 2. 总目标

本项目后续采用“稳健冲奖 + CPU 为主”的优化路线。

目标不是盲目堆复杂模型，而是在可复现、可解释、可记录的前提下提升线上分数，并保留真实 Agent 实验轨迹，满足复审对自动化实验控制能力的要求。

优先级：

1. 优先提升 A2 推荐任务分数。
2. 稳健提升 A1 分类任务分数。
3. 强化 Agent 的诊断、设计和 trajectory 记录质量。
4. 每次实验都记录到 `/home/aliagent/record.md`。

## 3. 实验记录制度

每次实验必须记录：

- 实验编号
- 日期时间
- 修改文件
- 修改内容
- 运行命令
- 本地指标
- 线上指标
- 结论
- 是否保留
- 下一步计划

提交前必须确认：

- `prediction.zip` 根目录只包含 `A1.csv` 和 `A2.csv`。
- `A1.csv` 两列为 `test_idx,label`。
- `A2.csv` 两列为 `uid,prediction`。
- A2 每行正好 10 个合法 item，且无重复。

## 4. A2 推荐任务优化路线

A2 当前线上分数为 `0.2173`，优先优化。

### 4.1 热门度兜底

目的：

- 处理空历史用户。
- 处理短历史用户。
- 补全模型 Top-K 中非法、重复或不足的结果。

实现：

- 从 `train.csv.target_iid` 统计全局热门 item。
- 推理阶段将热门 item 作为 fallback。
- 对每个用户确保输出 10 个合法不重复 item。

预期实验：

- `Exp-001`：热门 Top10 兜底，不重训，只改推理。

### 4.2 模型分数 + 热门度融合

目的：

- 推荐模型负责个性化。
- 热门度负责稀疏场景稳定性。

实现：

- 将热门度归一化为 `0~1`。
- 最终分数为：
  - `final_score = model_score + pop_weight * popularity_score`
- 默认 `pop_weight=0.15`，后续实验搜索 `0.05/0.10/0.15/0.25/0.40`。

预期实验：

- `Exp-002`：模型分数 + 热门度融合。

### 4.3 item 共现重排

目的：

- 利用用户历史 item 的共现规律。
- 对短序列用户补充更相关的候选 item。

实现：

- 从 `train.csv` 的 `item_seq_dedup` 和 `item_seq_raw` 构建 item-item 共现表。
- 对测试用户最近若干历史 item，给共现 item 加分。
- 最终分数为：
  - `final_score = model_score + pop_weight * popularity_score + cooccur_weight * cooccur_score`
- 默认 `cooccur_weight=0.25`。

预期实验：

- `Exp-003`：模型分数 + 热门度 + 共现重排。

## 5. A1 分类任务优化路线

A1 当前线上分数为 `0.3777`，采用稳健训练和轻量 ensemble。

### 5.1 分层验证

目的：

- 避免普通随机划分导致验证集类别分布偏移。

实现：

- 默认支持 `--stratified_split`。
- Agent 配置优先启用该参数。

### 5.2 类别权重

目的：

- 训练集类别分布不均衡，当前模型容易偏向大类。

实现：

- 新增 `--class_weight` 参数。
- 支持 `none` 和 `balanced`。
- `balanced` 使用训练标签频次计算 `CrossEntropyLoss(weight=...)`。

预期实验：

- `Exp-004`：SAGE + stratified split + class weight。

### 5.3 多模型和多 seed

目的：

- 减少单次训练随机性。
- 提升分类预测稳定性。

实现：

- 支持多个 checkpoint 做 logits 平均。
- 先比较 `sage/gcn`，不优先使用当前 GAT，因为它构造 `N x N` 注意力，CPU 风险高。

预期实验：

- `Exp-005`：GCN/SAGE 对比。
- `Exp-006`：多 seed ensemble。

## 6. Agent 质量优化路线

目标：让 Agent 的输出更可执行，而不是泛泛建议。

### 6.1 Diagnosis 约束

- 诊断必须基于已有 metrics。
- 如果没有实验指标，必须输出“冷启动探索”，不能误判为“实验未初始化”。
- A1 主要指标：`val_acc`。
- A2 主要指标：`val_ndcg`，辅助看 `val_hit` 和 `val_mrr`。

### 6.2 Design 结构化输出

- Design 优先生成 `config_update.json`。
- 每轮只改一个主变量，方便归因。
- A1 搜索空间：
  - `model_type`: `sage/gcn`
  - `lr`: `0.01/0.005/0.001`
  - `hidden_dim`: `128/256`
  - `dropout`: `0.3/0.5`
  - `class_weight`: `none/balanced`
- A2 搜索空间：
  - `model_type`: `gru4rec/sasrec`
  - `lr`: `0.001/0.0005`
  - `max_len`: `50/100`
  - `embedding_dim`: `64/128`
  - `pop_weight`: `0.05/0.15/0.25`
  - `cooccur_weight`: `0.0/0.25/0.5`

### 6.3 trajectory 输出

- A 榜保留内部轨迹。
- B 榜需要导出：
  - `trajectory_B1.json`
  - `trajectory_B2.json`
- 每条记录必须包含：
  - 实验轮次
  - 当前配置
  - 指标反馈
  - 下一步策略
  - 耗时

## 7. 测试计划

每次修改后执行：

```bash
cd /home/aliagent/framework
python3 -m py_compile code/*.py agent/*.py main.py
```

A2 快速推理检查：

```bash
python3 code/infer.py \
  --task task2 \
  --data_path data/rec_data \
  --checkpoint <best_model.pt> \
  --output_path <A2.csv> \
  --device cpu \
  --topk 10
```

提交格式检查：

- A1 行数必须为 `2751`。
- A2 行数必须为 `10000`。
- `prediction.zip` 只包含 `A1.csv` 和 `A2.csv`。

## 8. 实施顺序

1. 初始化计划和实验记录。
2. 实现 A2 热门度兜底。
3. 实现 A2 模型分数 + 热门度融合。
4. 实现 A2 item 共现重排。
5. 实现 A1 class weight + stratified split。
6. 实现 A1 多 checkpoint ensemble。
7. 强化 Agent Diagnosis/Design 和 trajectory。
8. 形成正式比赛命令和提交包生成流程。

## 9. 重要约束

- 当前以 CPU 为主，不依赖 GPU。
- 不覆盖原始数据。
- 不把 API key 写入代码、配置或提交包。
- 所有注释和文档保持中文。
- 不保证固定获得三等奖，因为奖项线取决于实时榜单和 B 榜，但该路线以稳健提升和复审可解释为目标。
