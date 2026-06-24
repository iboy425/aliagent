# 实验修改记录

本文件用于记录每一次代码修改、实验运行和线上提交结果。所有记录必须真实、可复现，不得补写与实际运行不一致的实验轨迹。

## 记录模板

```text
实验编号：
时间：
目标：
修改文件：
修改内容：
运行命令：
本地指标：
线上指标：
结论：
是否保留：
下一步：
```

---

## Exp-000 baseline 复现与线上提交

实验编号：Exp-000

时间：2026-06-23 18:20:14

目标：复现 baseline，跑通 A1/A2 训练、推理、Agent 流程和 A 榜提交。

修改文件：

- `framework/agent/tools.py`
- `framework/agent/phases.py`
- `framework/run_agent_with_api.sh`
- `framework/repro_agent_step6.yaml`
- `framework/.env.local.example`
- `framework/.gitignore`

修改内容：

- 修复工具层使用 `python` 的兼容问题，改为使用当前解释器 `sys.executable`。
- 修复 `phases.py` 缺少 `re` 导入导致 Design 阶段报错的问题。
- 增加 qwen-plus API 运行脚本和 `.env.local` 读取机制。
- 增加快速复现配置。
- 清理复现临时输出，仅保留已提交预测包备份。

运行命令：

```bash
cd /home/aliagent/framework
./run_agent_with_api.sh
```

本地指标：

- A1 快速复现验证准确率约 `0.1555`
- A2 快速复现验证 `val_ndcg≈0.1333`，`val_hit≈0.225`，`val_mrr≈0.1053`

线上指标：

- 总分：`0.2975`
- 分类任务分数：`0.3777`
- 推荐任务分数：`0.2173`

提交文件：

- `framework/output/baseline_submitted/prediction_20260623_182014_score_0.2975.zip`

结论：

- baseline 已完整跑通。
- qwen-plus API 已能调用。
- 当前主要短板是 A2 推荐任务和 Agent 结构化实验控制。

是否保留：

- 保留作为基线。

下一步：

- 优先实现 A2 热门度兜底、热门度融合和 item 共现重排。

---

## Repo-001 仓库初始化与安全提交

实验编号：Repo-001

时间：2026-06-23

目标：在开始新实验前，将当前 baseline 项目状态提交到 GitHub 仓库，形成可回滚起点。

修改文件：

- `.gitignore`
- `framework/config.yaml`

修改内容：

- 新增根目录 `.gitignore`，排除 `.env.local`、运行输出、日志、缓存、模型权重和本地工具目录。
- 将 `framework/config.yaml` 中的明文 API 配置清空，改为通过环境变量读取。
- 初始化本地 git 仓库，绑定远端 `git@github.com:iboy425/aliagent.git`。

运行命令：

```bash
cd /home/aliagent
python3 -m py_compile framework/code/*.py framework/agent/*.py framework/main.py
git push -u origin main
```

本地指标：

- Python 语法检查通过。
- 密钥扫描未发现 `sk-...` 形式明文 key。

线上指标：

- 不涉及线上评测。

结论：

- 当前 baseline 项目状态已提交到 GitHub。
- `.env.local`、`framework/output/`、`__pycache__/`、日志和模型/提交产物未入库。

是否保留：

- 保留，作为后续实验前的版本基线。

下一步：

- 在该基线上开始 A2 离线验证与优化实验。

---

## Exp-001 基础设施与优化代码落地

实验编号：Exp-001

时间：2026-06-23

目标：落地阶段 1 和阶段 2 基础设施：提交文件校验器与 A2 本地离线评分器，为后续优化提供安全检查和离线对比标准。

修改文件：

- `framework/code/validate_submission.py`
- `framework/code/a2_offline_eval.py`
- `record.md`

修改内容：

- 新增 `validate_submission.py`，校验 `A1.csv`、`A2.csv` 或 `prediction.zip` 的列名、行数、顺序、类别范围、推荐长度、重复 item、非法 item 和 zip 根目录结构。
- 新增 `a2_offline_eval.py`，从 `train.csv` 切分拟合集/验证集，计算 A2 的 NDCG@10、Hit@10、MRR，并按历史长度分桶分析。
- A2 离线评分器支持 `popular`、`last_item`、`history`、`hybrid` 四种基础策略，以及 `history_filter=none/soft/hard` 对照。

运行命令：

```bash
cd /home/aliagent/framework
python3 -m py_compile code/*.py
python3 code/validate_submission.py --zip_path output/baseline_submitted/prediction_20260623_182014_score_0.2975.zip
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy all --history_filter none --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy all --history_filter hard --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy all --history_filter soft --topk 10
```

本地指标：

- 语法检查通过。
- baseline 已提交 zip 校验通过：
  - `A1.csv` 行数 `2751`，类别分布 `{4: 2701, 8: 50}`
  - `A2.csv` 行数 `10000`，每行 10 个合法候选 item
- A2 离线评估，`history_filter=none`：
  - `popular`: `NDCG@10=0.327051`, `Hit@10=0.562250`, `MRR=0.254091`
  - `last_item`: `NDCG@10=0.508651`, `Hit@10=0.740000`, `MRR=0.436607`
  - `history`: `NDCG@10=0.527845`, `Hit@10=0.775250`, `MRR=0.450592`
  - `hybrid`: `NDCG@10=0.522988`, `Hit@10=0.774000`, `MRR=0.444575`
- A2 离线评估，`history_filter=hard`：
  - 最优 `history`: `NDCG@10=0.177787`, `Hit@10=0.292125`, `MRR=0.142518`
- A2 离线评估，`history_filter=soft`：
  - 最优 `popular`: `NDCG@10=0.254420`, `Hit@10=0.496500`, `MRR=0.179741`

线上指标：

- 未提交线上。

结论：

- 阶段 1/2 工具链已跑通。
- A2 中“硬排除历史 item”会显著降低离线 NDCG；当前 baseline 推理中存在类似逻辑，后续应优先改为不排除或可配置过滤。
- 历史 item 到 target 的共现信号非常强，离线 `history` 策略明显优于纯热门策略，是下一阶段 A2 提分的优先方向。

是否保留：

- 保留。

下一步：

- 阶段 3：把 A2 推理逻辑改成可配置的热门度/共现融合，并首先测试 `history_filter=none`。

---

## Exp-002 A2共现推理融合

实验编号：Exp-002

时间：2026-06-24

目标：把阶段 2 离线验证出的 A2 强信号接入正式推理链路，支持热门度、历史共现、模型分数融合，并显式控制是否过滤历史 item。

修改文件：

- `framework/code/rec_heuristics.py`
- `framework/code/a2_offline_eval.py`
- `framework/code/infer.py`
- `record.md`

修改内容：

- 新增 `rec_heuristics.py`，抽出 A2 推荐公共逻辑：序列解析、热门度统计、历史 item 到 target 的共现统计、模型/热门/共现分数融合、历史 item 过滤和 TopK 补齐。
- `a2_offline_eval.py` 改为调用公共推荐逻辑，保证离线评估与正式推理使用同一套排序规则。
- `infer.py` 的 Task 2 推理新增可控参数：
  - `--rec_strategy model|popular|last_item|history|hybrid`
  - `--history_filter none|soft|hard`
  - `--seq_col`
  - `--recent_n`
  - `--model_weight`
  - `--pop_weight`
  - `--cooccur_weight`
- Task 2 允许不提供 checkpoint，便于直接生成热门度/共现启发式推荐；Task 1 仍要求 checkpoint。
- Task 2 输出顺序改为严格保持 `test.csv` 原始顺序，不再按 `uid` 排序。

运行命令：

```bash
cd /home/aliagent
python3 -m py_compile framework/code/*.py framework/agent/*.py framework/main.py

cd /home/aliagent/framework
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy all --history_filter none --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy all --history_filter hard --topk 10
python3 code/infer.py --task task2 --data_path data/rec_data --output_path output/exp002_history_none/A2.csv --rec_strategy history --history_filter none --topk 10 --recent_n 20 --device cpu
unzip -p output/baseline_submitted/prediction_20260623_182014_score_0.2975.zip A1.csv > output/exp002_history_none/A1.csv
zip -j output/exp002_history_none/prediction.zip output/exp002_history_none/A1.csv output/exp002_history_none/A2.csv
python3 code/validate_submission.py --zip_path output/exp002_history_none/prediction.zip
```

本地指标：

- Python 语法检查通过。
- 离线评估保持阶段 2 结论：
  - `history_filter=none` 下，`history` 策略 `NDCG@10=0.527845`, `Hit@10=0.775250`, `MRR=0.450592`
  - `history_filter=hard` 下，`history` 策略 `NDCG@10=0.177787`, `Hit@10=0.292125`, `MRR=0.142518`
- 新生成 `output/exp002_history_none/A2.csv`：
  - 行数 `10000`
  - 每行推荐 `10` 个合法 item
  - 与 baseline A2 相比，`10000/10000` 行 prediction 发生变化
- 候选提交包 `output/exp002_history_none/prediction.zip` 校验通过：
  - `A1.csv` 行数 `2751`，类别分布 `{4: 2701, 8: 50}`
  - `A2.csv` 行数 `10000`，候选 item 数 `2156`

线上指标：

- 已提交 A 榜测试。
- 推荐任务分数（A2）：`0.4548`
- 相比 Exp-000 推荐任务分数 `0.2173`，提升 `+0.2375`。
- 若 A1 仍沿用 baseline `0.3777`，估算总分约为 `(0.3777 + 0.4548) / 2 = 0.41625`。

结论：

- A2 正式推理链路已支持不依赖 checkpoint 的历史共现推荐。
- 当前候选包只替换 A2，A1 沿用 baseline；线上反馈证明 A2 历史共现策略有效。
- 离线结果强烈支持 `history_filter=none`，暂不建议继续使用硬过滤历史 item。
- Exp-002 成为当前 A2 最优线上基线。

是否保留：

- 保留。

下一步：

- 围绕 Exp-002 做小范围可解释搜索：优先比较 `recent_n`、`seq_col`、`last_item/history/hybrid`，每天只提交离线最有把握的 1-2 个候选。

---

## Exp-003 A2序列列与共现窗口搜索

实验编号：Exp-003

时间：2026-06-24

目标：在 Exp-002 线上有效的基础上，搜索更优的历史序列来源和共现窗口，尽量只做低风险、可解释的 A2 参数优化。

修改文件：

- `framework/code/a2_grid_search.py`
- `record.md`

修改内容：

- 新增 `a2_grid_search.py`，用于批量比较 `seq_col`、`recent_n`、`strategy` 和 `history_filter` 的离线指标。
- 完整大网格运行较慢，实际本轮采用聚焦搜索：固定 `strategy=history`、`history_filter=none`，重点比较 `item_seq_dedup` 与 `item_seq_raw`、不同 `recent_n`。

运行命令：

```bash
cd /home/aliagent/framework
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_dedup --recent_n 1 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_dedup --recent_n 3 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_dedup --recent_n 5 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_dedup --recent_n 10 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_dedup --recent_n 20 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_dedup --recent_n 50 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_dedup --recent_n 0 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 1 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 3 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 5 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 20 --topk 10

python3 code/infer.py --task task2 --data_path data/rec_data --output_path output/exp003_raw_recent10/A2.csv --rec_strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10 --device cpu
unzip -p output/baseline_submitted/prediction_20260623_182014_score_0.2975.zip A1.csv > output/exp003_raw_recent10/A1.csv
zip -j output/exp003_raw_recent10/prediction.zip output/exp003_raw_recent10/A1.csv output/exp003_raw_recent10/A2.csv
python3 code/validate_submission.py --zip_path output/exp003_raw_recent10/prediction.zip
```

本地指标：

- `item_seq_dedup`：
  - `recent_n=1`: `NDCG@10=0.505198`
  - `recent_n=3`: `NDCG@10=0.535210`
  - `recent_n=5`: `NDCG@10=0.534937`
  - `recent_n=10`: `NDCG@10=0.530643`
  - `recent_n=20`: `NDCG@10=0.527845`
  - `recent_n=50`: `NDCG@10=0.523756`
  - `recent_n=0`: `NDCG@10=0.520730`
- `item_seq_raw`：
  - `recent_n=1`: `NDCG@10=0.505198`
  - `recent_n=3`: `NDCG@10=0.533586`
  - `recent_n=5`: `NDCG@10=0.538631`
  - `recent_n=10`: `NDCG@10=0.544448`
  - `recent_n=20`: `NDCG@10=0.543116`
- 当前离线最优：`item_seq_raw + recent_n=10 + history_filter=none + strategy=history`，`NDCG@10=0.544448`。
- 新候选包 `output/exp003_raw_recent10/prediction.zip` 校验通过。
- 与 Exp-002 的 A2 相比，Exp-003 有 `5914/10000` 行推荐发生变化。

线上指标：

- 未提交线上。

结论：

- raw 序列保留重复交互，比 dedup 序列更能表达近期强偏好。
- 最优窗口从 Exp-002 的 `recent_n=20` 改为 `recent_n=10`，说明使用过长历史会引入噪声。
- Exp-003 是当前下一次 A 榜提交的优先候选。

是否保留：

- 保留。

下一步：

- 若今日仍有提交次数，优先提交 `framework/output/exp003_raw_recent10/prediction.zip` 验证线上 A2。
- 若线上继续提升，再围绕 `item_seq_raw + recent_n=10` 做轻量融合或 A1 提升。
