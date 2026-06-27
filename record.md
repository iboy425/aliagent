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

---

## Exp-004 官方提分方向能力补齐与用户画像候选

实验编号：Exp-004

时间：2026-06-24

目标：根据官方提分指南补齐可实验能力，包括 A1 特征/图增强、A2 序列训练增强、用户画像融合和热门惩罚，并生成一个符合官方方向的 A2 候选提交包。

修改文件：

- `framework/code/utils.py`
- `framework/code/train.py`
- `framework/code/datasets.py`
- `framework/code/rec_heuristics.py`
- `framework/code/infer.py`
- `framework/code/a2_offline_eval.py`
- `framework/code/a2_grid_search.py`
- `record.md`

修改内容：

- Task 1 新增官方建议的数据增强开关：
  - `--feature_norm none|row|l2`：节点特征归一化。
  - `--dropedge_rate`：训练前随机丢弃部分边。
  - `--feature_mask_rate`：训练期随机遮蔽部分节点特征。
  - `--class_weight none|balanced`：按类别频次给交叉熵加权。
- Task 1 推理阶段读取 checkpoint 中的 `feature_norm`，保证训练和推理预处理一致；DropEdge 只在训练使用。
- Task 2 训练新增官方建议的可控项：
  - `--seq_col`：选择 `item_seq_raw` 或 `item_seq_dedup`。
  - `--neg_sampling_strategy random|popularity`。
  - `--eval_history_filter none|soft|hard`。
- 修复 Task 2 BPR 多负样本路径，支持 `--neg_samples > 1`。
- 修复 Task 2 `SASRec + CE` 训练路径，避免把 `seq_lens` 传给只接收序列的 SASRec。
- Task 2 启发式推理新增用户画像融合：
  - 从 `user.csv` 的 `u_cat_01` 到 `u_cat_08` 统计画像分组 target 热门度。
  - 推理时用 `--user_weight` 控制画像分数融合强度。
- Task 2 启发式推理新增 `--pop_penalty_weight`，用于测试官方建议的热门惩罚。
- A2 离线评估器和网格搜索器同步支持用户画像权重和热门惩罚权重，保证离线与提交逻辑一致。

运行命令：

```bash
cd /home/aliagent
python3 -m py_compile framework/code/*.py framework/agent/*.py framework/main.py

cd /home/aliagent/framework
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10 --user_weight 0.01 --user_profile_cols auto
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10 --user_weight 0.02 --user_profile_cols auto
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10 --user_weight 0.03 --user_profile_cols auto
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10 --user_weight 0.04 --user_profile_cols auto
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10 --user_weight 0.05 --user_profile_cols auto
python3 code/a2_offline_eval.py --data_path data/rec_data --val_ratio 0.2 --seed 42 --strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10 --pop_penalty_weight 0.02

python3 code/infer.py --task task2 --data_path data/rec_data --output_path output/exp004_official_user002/A2.csv --rec_strategy history --history_filter none --seq_col item_seq_raw --recent_n 10 --topk 10 --user_weight 0.02 --user_profile_cols auto --device cpu
unzip -p output/baseline_submitted/prediction_20260623_182014_score_0.2975.zip A1.csv > output/exp004_official_user002/A1.csv
zip -j output/exp004_official_user002/prediction.zip output/exp004_official_user002/A1.csv output/exp004_official_user002/A2.csv
python3 code/validate_submission.py --zip_path output/exp004_official_user002/prediction.zip

python3 code/train.py --task task1 --data_path data/cls_data/A1.npz --output_dir output/smoke_a1_official --epochs 1 --model_type sage --hidden_dim 16 --num_layers 2 --feature_norm row --dropedge_rate 0.05 --feature_mask_rate 0.1 --class_weight balanced --stratified_split --device cpu --log_interval 1
python3 code/train.py --task task2 --data_path data/rec_data --output_dir output/smoke_a2_sasrec_neg5 --epochs 1 --model_type sasrec --embedding_dim 16 --num_layers 1 --num_heads 2 --max_len 30 --batch_size 1024 --loss_type bpr --neg_samples 5 --neg_sampling_strategy random --seq_col item_seq_raw --eval_history_filter none --device cpu --log_interval 1
python3 code/train.py --task task2 --data_path data/rec_data --output_dir output/smoke_a2_sasrec_ce --epochs 1 --model_type sasrec --embedding_dim 16 --num_layers 1 --num_heads 2 --max_len 30 --batch_size 1024 --loss_type ce --seq_col item_seq_raw --eval_history_filter none --device cpu --log_interval 1
```

本地指标：

- Python 语法检查通过。
- A2 Exp-003 对照：`item_seq_raw + recent_n=10 + history_filter=none`，`NDCG@10=0.544448`，`Hit@10=0.779750`，`MRR=0.470771`。
- 用户画像融合：
  - `user_weight=0.01`: `NDCG@10=0.544531`
  - `user_weight=0.02`: `NDCG@10=0.544541`
  - `user_weight=0.03`: `NDCG@10=0.544524`
  - `user_weight=0.04`: `NDCG@10=0.544492`
  - `user_weight=0.05`: `NDCG@10=0.544478`
- 热门惩罚：
  - `pop_penalty_weight=0.02`: `NDCG@10=0.543714`，本地负向，暂不启用。
- 新候选包 `output/exp004_official_user002/prediction.zip` 校验通过：
  - `A1.csv` 行数 `2751`，类别分布 `{4: 2701, 8: 50}`
  - `A2.csv` 行数 `10000`，每行 10 个合法候选 item
  - 与 Exp-003 相比，A2 有 `3686/10000` 行 prediction 发生变化。
- Smoke test：
  - A1 `feature_norm=row + DropEdge=0.05 + FeatureMask=0.1 + class_weight=balanced` 1 epoch 跑通。
  - A2 `SASRec + BPR + neg_samples=5` 1 epoch 跑通。
  - A2 `SASRec + CE` 1 epoch 跑通。

线上指标：

- 未提交线上。

结论：

- 官方提分指南中的主要可实验开关已经落到代码里。
- A2 用户画像融合只有非常小的本地提升，推荐作为低风险候选提交，但线上收益不保证。
- 热门惩罚当前本地为负向，暂不建议提交启用。
- A1 增强能力已跑通，但尚未做完整训练和线上验证；后续应先小网格比较 `feature_norm`、`class_weight`、`dropedge_rate`，再考虑替换 A1。

是否保留：

- 保留。

下一步：

- 若还有 A 榜提交次数，可以优先比较：
  - 更稳妥：提交 Exp-003 `output/exp003_raw_recent10/prediction.zip`。
  - 官方融合候选：提交 Exp-004 `output/exp004_official_user002/prediction.zip`。
- 后续不要先大跑 SASRec；当前启发式 A2 已明显更强，训练模型应作为融合补充而不是主替换。

---

## Exp-006 A1稳健训练与A2官方SASRec验证

实验编号：Exp-006

时间：2026-06-24

目标：按官方提分方向重新训练 A1/A2，但避免“所有增强全开”。先用 A1 稳健配置追求 Accuracy，再用 A2 SASRec 官方配置验证深度序列模型是否有机会超过当前启发式。

修改文件：

- `record.md`

修改内容：

- 本轮不改代码，只运行训练、推理、打包和校验。
- A1 先尝试过 `class_weight=balanced` 的全增强配置，验证准确率长期偏低，人工中断；结论是类别均衡损失不适合当前线上 Accuracy 目标。
- A1 改用稳健配置：GraphSAGE、hidden_dim=256、2层、分层验证、无类别权重、无 DropEdge/FeatureMask。
- A2 启动官方 SASRec 大配置：`item_seq_raw`、`max_len=100`、`embedding_dim=128`、`num_layers=2`、`num_heads=4`、`neg_samples=5`、BPR。
- A2 SASRec 前 5 轮验证 NDCG 明显低于启发式，停止训练，保留为“官方模型路线已验证但暂不替换”的结论。

运行命令：

```bash
cd /home/aliagent/framework

# A1 反例：全增强 + class_weight，验证指标低，人工中断
python3 code/train.py --task task1 --data_path data/cls_data/A1.npz --output_dir output/exp005_a1_sage_official_full --epochs 300 --model_type sage --hidden_dim 256 --num_layers 3 --lr 0.005 --dropout 0.5 --weight_decay 0.0005 --patience 30 --normalize symmetric --feature_norm row --dropedge_rate 0.05 --feature_mask_rate 0.05 --class_weight balanced --stratified_split --device cpu --log_interval 5

# A1 正式候选：稳健 GraphSAGE
python3 code/train.py --task task1 --data_path data/cls_data/A1.npz --output_dir output/exp006_a1_sage_stable --epochs 300 --model_type sage --hidden_dim 256 --num_layers 2 --lr 0.005 --dropout 0.5 --weight_decay 0.0005 --patience 40 --normalize symmetric --feature_norm none --dropedge_rate 0 --feature_mask_rate 0 --class_weight none --stratified_split --device cpu --log_interval 5
python3 code/infer.py --task task1 --data_path data/cls_data/A1.npz --checkpoint output/exp006_a1_sage_stable/best_model.pt --output_path output/exp006_a1_sage_stable/A1.csv --device cpu

# A2 官方模型验证：SASRec + 多负样本，前5轮后因远低于启发式而停止
python3 code/train.py --task task2 --data_path data/rec_data --output_dir output/exp006_a2_sasrec_official --epochs 80 --model_type sasrec --embedding_dim 128 --num_layers 2 --num_heads 4 --max_len 100 --batch_size 512 --loss_type bpr --neg_samples 5 --neg_sampling_strategy random --seq_col item_seq_raw --eval_history_filter none --lr 0.001 --dropout 0.2 --weight_decay 0.00001 --patience 12 --device cpu --log_interval 2

# 组合提交包
mkdir -p output/exp006_submit_a1_stable_a2_exp003
mkdir -p output/exp006_submit_a1_stable_a2_exp004
cp output/exp006_a1_sage_stable/A1.csv output/exp006_submit_a1_stable_a2_exp003/A1.csv
cp output/exp006_a1_sage_stable/A1.csv output/exp006_submit_a1_stable_a2_exp004/A1.csv
unzip -p output/exp003_raw_recent10/prediction.zip A2.csv > output/exp006_submit_a1_stable_a2_exp003/A2.csv
unzip -p output/exp004_official_user002/prediction.zip A2.csv > output/exp006_submit_a1_stable_a2_exp004/A2.csv
zip -j output/exp006_submit_a1_stable_a2_exp003/prediction.zip output/exp006_submit_a1_stable_a2_exp003/A1.csv output/exp006_submit_a1_stable_a2_exp003/A2.csv
zip -j output/exp006_submit_a1_stable_a2_exp004/prediction.zip output/exp006_submit_a1_stable_a2_exp004/A1.csv output/exp006_submit_a1_stable_a2_exp004/A2.csv
python3 code/validate_submission.py --zip_path output/exp006_submit_a1_stable_a2_exp003/prediction.zip
python3 code/validate_submission.py --zip_path output/exp006_submit_a1_stable_a2_exp004/prediction.zip
```

本地指标：

- A1 全增强 + `class_weight=balanced`：
  - 人工中断前验证准确率约 `0.1498`，明显不适合提交。
  - 原因判断：类别均衡权重更关注少数类，但线上指标是总体 Accuracy，导致主类准确率受损。
- A1 稳健 GraphSAGE：
  - 最优验证准确率：`0.632877`
  - 最优 epoch：`146`
  - 训练集准确率后期约 `0.996`，说明模型容量充足，也开始过拟合；使用 best checkpoint。
  - 新 A1 与 baseline A1 相比，`1483/2751` 行预测变化。
  - 新 A1 类别分布：`{0: 65, 1: 382, 2: 254, 3: 69, 4: 1262, 5: 44, 6: 54, 7: 144, 8: 447, 9: 30}`。
  - baseline A1 类别分布：`{4: 2701, 8: 50}`。
- A2 SASRec 官方配置：
  - Epoch 1：`Val NDCG@10=0.2078`
  - Epoch 2：`Val NDCG@10=0.2241`
  - Epoch 3：`Val NDCG@10≈0.2294`
  - Epoch 4：`Val NDCG@10=0.2405`
  - Epoch 5：`Val NDCG@10≈0.2415`
  - 明显低于当前启发式离线 `NDCG@10=0.544448`，停止训练。
- 提交包校验：
  - `output/exp006_submit_a1_stable_a2_exp003/prediction.zip` 校验通过。
  - `output/exp006_submit_a1_stable_a2_exp004/prediction.zip` 校验通过。

线上指标：

- 已提交 `framework/output/exp006_submit_a1_stable_a2_exp004/prediction.zip` 到 A 榜。
- 提交时间：2026-06-24 22:55:03
- 总分：`0.5508`
- 分类任务分数：`0.6369`
- 推荐任务分数：`0.4647`

结论：

- A1 不应该开启 `class_weight=balanced`；稳健 SAGE 配置明显更好。
- A1 新候选线上验证有效：本地 `val_acc=0.632877`，线上分类分数 `0.6369`，说明当前 A1 验证集划分与线上分布比较一致。
- A2 官方 SASRec 训练可以跑通，但当前不适合作为主提交方案；A2 仍应使用 `item_seq_raw + recent_n=10 + history_filter=none` 的启发式共现路线。
- A2 用户画像融合线上有效：推荐分从 Exp-002 的 `0.4548` 提升到 `0.4647`。
- 当前最高线上总分为 `0.5508`，Exp-006 成为新的线上基线。

是否保留：

- 保留 A1 稳健模型和两个提交候选包。
- 不保留 A2 SASRec 作为当前提交主模型。

下一步：

- 以 `A1=0.6369`、`A2=0.4647` 为新基线。
- 下一轮优先做 A1 多 seed / 多 checkpoint 集成；A2 只做小范围启发式参数搜索，不再优先大跑 SASRec。

---

## Tool-001：补齐 A1 多 checkpoint 集成推理

日期：2026-06-25

目标：

- 为下一阶段 GPU 多 seed 训练做准备。
- 让 `code/infer.py` 已有的 `--ensemble_checkpoints` 参数真正支持 Task 1。
- 支持多个 A1 checkpoint 做 logits 平均，生成一个更稳健的 `A1.csv`。

修改文件：

- `framework/code/infer.py`

修改内容：

- 新增 `_prepare_task1_tensors()`：
  - 按每个 checkpoint 保存的训练参数读取 `feature_norm` 和 `normalize`。
  - 分别复现训练时的特征归一化和邻接矩阵归一化。
  - 这样不同 seed 或不同配置训练出来的模型，可以用各自正确的数据预处理方式推理。
- 修改 `infer_task1()`：
  - 原来 Task 1 只能使用单个 `--checkpoint`。
  - 现在支持 `--ensemble_checkpoints ckpt1 ckpt2 ...`。
  - 每个 checkpoint 单独前向计算测试节点 logits。
  - 对所有 logits 求平均，再取 `argmax` 得到最终类别。
  - 每个模型推理结束后释放显存，降低 GPU 显存占用。

原理说明：

- 单模型输出的 `label` 只告诉我们“它最终选了哪个类别”。
- logits 是模型对 10 个类别的原始信心分，比最终标签包含更多信息。
- 多 seed 模型的错误通常不完全相同，把 logits 平均后可以降低单个模型随机性带来的波动。
- 这一步不改变训练过程，只改变最终 A1 推理融合方式，风险较低。

验证命令：

```bash
cd /home/aliagent
python3 -m py_compile framework/code/infer.py
python3 framework/code/infer.py \
  --task task1 \
  --data_path framework/data/cls_data/A1.npz \
  --ensemble_checkpoints \
    framework/output/exp006_a1_sage_stable/best_model.pt \
    framework/output/exp006_a1_sage_stable/best_model.pt \
  --output_path framework/output/exp007_smoke_a1_ensemble/A1.csv \
  --device cpu
rm -rf framework/output/exp007_smoke_a1_ensemble
```

验证结果：

- `py_compile` 通过。
- 重复同一个 checkpoint 做 2 模型集成，输出类别分布与单模型一致：
  - `{0: 65, 1: 382, 2: 254, 3: 69, 4: 1262, 5: 44, 6: 54, 7: 144, 8: 447, 9: 30}`
- 临时烟测输出已删除，避免污染正式实验目录。

线上指标：

- 尚未提交。该记录只是工具能力补齐。

是否保留：

- 保留。下一轮 GPU 实验将使用该功能做 A1 多 seed 集成。

---

## Tool-002：固定 A1 验证集划分并增加 ensemble 验证工具

日期：2026-06-25

背景：

- GPU 上已完成第一轮 A1 多 seed 训练：
  - seed=42：最优验证准确率 `0.6502`
  - seed=2024：最优验证准确率 `0.6575`
  - seed=3407：最优验证准确率 `0.6639`
  - seed=777：最优验证准确率 `0.6493`
  - seed=2026：最优验证准确率 `0.6749`
- 这些结果说明 A1 仍有提升空间，单模型已经高于 Exp-006 的 `0.632877`。

发现的问题：

- 当前 `--seed` 同时控制模型初始化、Dropout随机性、训练/验证集划分。
- 因此不同 seed 的 `val_acc` 不是在同一份验证集上评估，不能严格按数值直接排序。
- 如果直接用这组日志决定 Top-3/Top-5 ensemble，存在验证集口径不一致的问题。

修改文件：

- `framework/code/train.py`
- `framework/code/a1_ensemble_eval.py`

修改内容：

- `train.py` 新增 `--split_seed`：
  - 默认 `None`，保持旧行为：不传时仍沿用 `--seed`。
  - 传入后，Task 1 的训练/验证集划分固定使用 `--split_seed`。
  - 模型初始化和训练随机性仍由 `--seed` 控制。
- 新增 `a1_ensemble_eval.py`：
  - 输入多个 A1 checkpoint。
  - 在指定 `--split_seed` 的固定验证集上重新评估单模型准确率。
  - 按单模型验证准确率排序后评估 Top-1 / Top-3 / Top-5 等 logits 平均 ensemble。
  - 输出 JSON 和 CSV，方便记录与复盘。

原理说明：

- 科学实验里，比较模型必须尽量保证“考试卷一致”。
- `seed=42` 和 `seed=2026` 如果对应不同验证集，`0.6502` 和 `0.6749` 的差异可能同时包含：
  - 模型初始化差异；
  - 训练过程随机性差异；
  - 验证集难度差异。
- 增加 `--split_seed` 后，可以固定验证集，只改变模型 seed。
- 这样得到的多个 checkpoint 更适合做 ensemble 对比，也更适合判断线上提交候选。

验证命令：

```bash
cd /home/aliagent
python3 -m py_compile framework/code/train.py framework/code/a1_ensemble_eval.py framework/code/infer.py
```

验证结果：

- 语法检查通过。

线上指标：

- 尚未提交。该记录是实验方法修正。

是否保留：

- 保留。后续 A1 GPU 实验统一使用 `--split_seed 42`，只改变 `--seed`。

---

## Exp-008：A1 固定验证集多 seed 训练与 Top-2 ensemble 选择

日期：2026-06-25

目标：

- 使用 GPU 重训 A1 多个 seed。
- 固定 `split_seed=42`，只改变模型训练随机种子，保证多个模型在同一份验证集上比较。
- 用 `a1_ensemble_eval.py` 选择最优 A1 ensemble 组合。

训练配置：

- 模型：GraphSAGE
- `hidden_dim=256`
- `num_layers=2`
- `lr=0.005`
- `dropout=0.5`
- `weight_decay=0.0005`
- `patience=80`
- `normalize=symmetric`
- `feature_norm=none`
- `dropedge_rate=0`
- `feature_mask_rate=0`
- `class_weight=none`
- `stratified_split=True`
- `split_seed=42`
- `device=cuda`

运行命令：

```bash
cd /home/aliagent/framework

for seed in 42 2024 3407 777 2026
do
  CUDA_VISIBLE_DEVICES=0 python3 code/train.py \
    --task task1 \
    --data_path data/cls_data/A1.npz \
    --output_dir output/exp008_a1_sage_fixedsplit_seed${seed} \
    --epochs 500 \
    --model_type sage \
    --hidden_dim 256 \
    --num_layers 2 \
    --lr 0.005 \
    --dropout 0.5 \
    --weight_decay 0.0005 \
    --patience 80 \
    --normalize symmetric \
    --feature_norm none \
    --dropedge_rate 0 \
    --feature_mask_rate 0 \
    --class_weight none \
    --stratified_split \
    --split_seed 42 \
    --device cuda \
    --seed ${seed} \
    --log_interval 5
done

CUDA_VISIBLE_DEVICES=0 python3 code/a1_ensemble_eval.py \
  --data_path data/cls_data/A1.npz \
  --checkpoints \
    output/exp008_a1_sage_fixedsplit_seed42/best_model.pt \
    output/exp008_a1_sage_fixedsplit_seed2024/best_model.pt \
    output/exp008_a1_sage_fixedsplit_seed3407/best_model.pt \
    output/exp008_a1_sage_fixedsplit_seed777/best_model.pt \
    output/exp008_a1_sage_fixedsplit_seed2026/best_model.pt \
  --device cuda \
  --val_ratio 0.1 \
  --split_seed 42 \
  --stratified_split \
  --topks 1,2,3,4,5 \
  --output_json output/exp008_a1_fixedsplit_eval/results.json
```

固定验证集单模型结果：

| 排名 | seed | val_acc |
|---:|---:|---:|
| 1 | 777 | 0.652055 |
| 2 | 42 | 0.650228 |
| 3 | 3407 | 0.641096 |
| 4 | 2026 | 0.631963 |
| 5 | 2024 | 0.622831 |

集成验证结果：

| 方案 | val_acc |
|---|---:|
| Top-1 | 0.652055 |
| Top-2 | 0.659361 |
| Top-3 | 0.657534 |
| Top-4 | 0.652968 |
| Top-5 | 0.644749 |

结论：

- 固定验证集后，单模型最好的是 `seed=777`，不是第一轮非固定验证集里的 `seed=2026`。
- 这证明之前不同 seed 的验证分数确实受验证集划分影响，不能直接比较。
- 最优 ensemble 是 Top-2：`seed=777 + seed=42`，验证准确率 `0.659361`。
- Top-3/Top-4/Top-5 反而下降，说明较弱 checkpoint 会拉低平均 logits。
- 下一步提交候选应使用 A1 Top-2 ensemble，并沿用当前最优 A2 启发式推荐方案。

线上指标：

- 已提交 `framework/output/exp008_submit_a1_top2_a2_best/prediction.zip` 到 A 榜。
- 提交时间：2026-06-25 15:34:38
- 总分：`0.5571`
- 分类任务分数：`0.6496`
- 推荐任务分数：`0.4647`

是否保留：

- 保留 Top-2 A1 ensemble 作为下一次提交候选。

线上结论：

- 相比 Exp-006：
  - 总分：`0.5508 -> 0.5571`，提升 `+0.0063`
  - A1：`0.6369 -> 0.6496`，提升 `+0.0127`
  - A2：`0.4647 -> 0.4647`，持平
- A1 Top-2 ensemble 线上有效，但提升幅度小于本地固定验证集增益。
- 继续简单增加 seed 的收益可能递减；下一步需要扩大 A1 模型/特征/归一化搜索空间，并用固定验证集筛选。

---

## Tool-003：新增 A1 标签传播融合工具

日期：2026-06-25

目标：

- 为 Exp-009 准备 A1 后处理提分工具。
- 在 GNN logits 之外，额外利用图上已知训练标签的传播信号。

修改文件：

- `framework/code/a1_labelprop.py`

核心思路：

- GNN 输出：模型根据节点特征和邻接关系给每个节点打 10 类 logits。
- 标签传播输出：把已知训练节点标签转为 one-hot，然后沿图边迭代传播，得到每个节点的类别置信度。
- 融合输出：`final_score = (1 - lp_weight) * model_prob + lp_weight * label_prop_score`。

为什么可能有效：

- A1 是图节点分类任务，测试节点虽然没有标签，但它们和训练节点在同一张图中。
- 如果图存在同质性，相邻或近邻产品更可能属于相同类别。
- 当前 GraphSAGE 线上 A1 只有 `0.6496`，距离第一名 A1 `0.76845` 差距仍大，说明还需要更强的结构利用方式。
- 标签传播是低成本结构后处理，不需要重新训练模型，适合快速验证。

脚本功能：

- 支持多个 checkpoint logits 平均。
- 支持固定验证集搜索：
  - `alpha`：标签传播保留邻居信息的比例。
  - `num_iter`：传播轮数。
  - `lp_weight`：标签传播分数和模型分数的融合权重。
- 支持用搜索出的参数生成最终 `A1.csv`。

验证结果：

- `python3 -m py_compile framework/code/a1_labelprop.py framework/code/infer.py framework/code/train.py` 通过。
- CPU 烟测因 dense 图矩阵乘法较慢中止；正式验证应在 GPU 服务器执行。

线上指标：

- 尚未提交。

是否保留：

- 保留。下一步在 GPU 上执行 Exp-009 标签传播参数搜索。

---

## Exp-009：A1 Top-2 ensemble + 标签传播融合搜索

日期：2026-06-25

目标：

- 在 Exp-008 A1 Top-2 ensemble 基础上，尝试使用图标签传播后处理继续提升 A1。
- 使用固定验证集 `split_seed=42` 搜索标签传播参数。

输入 checkpoint：

- `output/exp008_a1_sage_fixedsplit_seed777/best_model.pt`
- `output/exp008_a1_sage_fixedsplit_seed42/best_model.pt`

运行命令：

```bash
cd /home/aliagent/framework

CUDA_VISIBLE_DEVICES=0 python3 code/a1_labelprop.py \
  --data_path data/cls_data/A1.npz \
  --checkpoints \
    output/exp008_a1_sage_fixedsplit_seed777/best_model.pt \
    output/exp008_a1_sage_fixedsplit_seed42/best_model.pt \
  --device cuda \
  --val_ratio 0.1 \
  --split_seed 42 \
  --stratified_split \
  --lp_normalize random_walk \
  --alphas 0.1,0.3,0.5,0.7,0.85 \
  --num_iters 5,10,20,40 \
  --lp_weights 0,0.02,0.05,0.1,0.15,0.2,0.3 \
  --output_json output/exp009_a1_labelprop_search/results.json
```

本地固定验证集结果：

- 模型原始验证准确率：`0.659361`
- 最佳标签传播融合：
  - `alpha=0.7`
  - `num_iter=5`
  - `lp_weight=0.3`
  - `val_acc=0.660274`

Top 结果摘要：

| 方案 | val_acc |
|---|---:|
| model only | 0.659361 |
| label propagation best | 0.660274 |

结论：

- 标签传播方向在固定验证集上有正收益，但只提升 `+0.000913`。
- 这个提升约等于验证集上多预测对 1 个节点，统计信号很弱。
- 不建议优先消耗线上提交次数，除非当天提交次数充足。
- 标签传播工具保留，后续如果有更强 A1 checkpoint，可再次尝试融合。

线上指标：

- 尚未提交。

是否保留：

- 保留搜索结果和工具。
- 暂不把 Exp-009 作为优先提交方案。

---

## Tool-004：A1 GCN 稀疏邻接训练加速

日期：2026-06-25

背景：

- Exp-010 A1 结构搜索已经显示 GCN 明显强于当前 SAGE：
  - `gcn hidden=384 seed=777` 固定验证集 `val_acc=0.677626`
  - `gcn hidden=256 seed=777` 固定验证集 `val_acc=0.674886`
  - Exp-008 Top-2 ensemble 固定验证集为 `0.659361`
- GPU 监控显示 GPU 使用率约 `11%`，说明当前训练不是算力吃满，而是存在数据结构/调度瓶颈。

发现的问题：

- 原始 `normalize_adj()` 会把 scipy 稀疏邻接矩阵转成 dense Tensor。
- A1 图有 13,752 个节点，dense 邻接矩阵约为 `13752 x 13752`。
- 图本身是稀疏结构，dense 矩阵乘法会做大量无效零元素计算，并占用不必要显存。
- 当前最优方向是 GCN，而 GCN 层只需要 `adj @ x`，天然适合稀疏邻接矩阵。

修改文件：

- `framework/code/utils.py`
- `framework/code/train.py`
- `framework/code/infer.py`

修改内容：

- `utils.py`：
  - 新增 `normalize_adj_sparse()`：稀疏对称归一化 `D^-1/2 (A+I) D^-1/2`。
  - 新增 `random_walk_normalize_sparse()`：稀疏随机游走归一化。
- `train.py`：
  - 新增参数 `--adj_format dense|sparse`。
  - 当 `--model_type gcn --adj_format sparse` 时，使用稀疏邻接训练。
  - 非 GCN 模型传入 `sparse` 时自动回退 dense，避免 GraphSAGE degree 逻辑产生兼容问题。
- `infer.py`：
  - 推理时读取 checkpoint 中保存的 `adj_format`。
  - 稀疏训练的 GCN checkpoint 会自动使用稀疏邻接推理。

验证命令：

```bash
cd /home/aliagent
python3 -m py_compile framework/code/utils.py framework/code/train.py framework/code/infer.py framework/code/a1_ensemble_eval.py

python3 framework/code/train.py \
  --task task1 \
  --data_path framework/data/cls_data/A1.npz \
  --output_dir framework/output/exp011_smoke_sparse_gcn \
  --epochs 1 \
  --model_type gcn \
  --hidden_dim 32 \
  --num_layers 2 \
  --lr 0.005 \
  --dropout 0.5 \
  --weight_decay 0.0005 \
  --patience 1 \
  --normalize symmetric \
  --adj_format sparse \
  --feature_norm none \
  --class_weight none \
  --stratified_split \
  --split_seed 42 \
  --device cpu \
  --seed 42 \
  --log_interval 1
rm -rf framework/output/exp011_smoke_sparse_gcn
```

验证结果：

- 语法检查通过。
- 1 epoch 稀疏 GCN 烟测通过。
- 日志显示 `邻接矩阵格式: sparse`，训练前向/反向可运行。

线上指标：

- 尚未提交。该记录是速度优化工具改造。

是否保留：

- 保留。后续 GCN 搜索统一优先使用 `--adj_format sparse`。

---

## Exp-010：A1 结构网格搜索与强模型 ensemble 评估

日期：2026-06-25

目标：

- 扩大 A1 模型结构搜索空间，不再只依赖 Exp-008 的 SAGE 多 seed。
- 比较 GCN / SAGE、hidden_dim、归一化方式等配置。
- 在固定验证集 `split_seed=42` 上评估强 checkpoint 的 ensemble。

关键发现：

- GCN 在本轮明显强于 Exp-008 的 SAGE 方案。
- 单模型最好结果：
  - `output/exp010_a1_grid_c6_gcn_h384_l2_symmetric_none_seed777`
  - 固定验证集 `val_acc=0.677626`
- 当前线上提交使用的 Exp-008 A1 Top-2 ensemble 固定验证集为 `0.659361`。
- 因此仅单模型已提升 `+0.018265`。

强 checkpoint 固定验证集单模型结果：

| 排名 | checkpoint | val_acc |
|---:|---|---:|
| 1 | `c6_gcn_h384_seed777` | 0.677626 |
| 2 | `c5_gcn_h256_seed777` | 0.674886 |
| 3 | `c1_sage_h384_seed3407` | 0.673059 |
| 4 | `c5_gcn_h256_seed42` | 0.673059 |
| 5 | `c7_sage_h256_random_walk_seed777` | 0.672146 |
| 6 | `c5_gcn_h256_seed3407` | 0.672146 |
| 7 | `c6_gcn_h384_seed3407` | 0.669406 |
| 8 | `c6_gcn_h384_seed42` | 0.668493 |

ensemble 结果：

| 方案 | val_acc |
|---|---:|
| Top-1 | 0.677626 |
| Top-2 | 0.671233 |
| Top-3 | 0.678539 |
| Top-4 | 0.677626 |
| Top-5 | 0.689498 |
| Top-6 | 0.683105 |
| Top-7 | 0.681279 |
| Top-8 | 0.679452 |

结论：

- 最优方案是 Top-5 ensemble，固定验证集准确率 `0.689498`。
- Top-5 相比 Exp-008 提交用 A1 方案提升 `+0.030137`。
- Top-2 反而下降，说明 ensemble 不是越少越好；Top-6 之后也下降，说明弱模型会拉低融合。
- 下一次提交候选应使用 Exp-010 Top-5 A1 ensemble，并继续沿用当前线上最优 A2 方案。

线上指标：

- 尚未提交。

是否保留：

- 保留 Top-5 A1 ensemble 作为下一次优先提交候选。

---

## Tool-005：修复 A2 序列模型取错位置并增强 GPU 训练吞吐

日期：2026-06-25

背景：

- 当前线上 A2 最优仍为 `0.4647`，距离第一名 A2 `0.50967` 有明显差距。
- 之前官方 SASRec 配置训练 5 轮左右，本地验证 NDCG 只有约 `0.24`，明显低于启发式共现方案。
- 用户反馈 GPU 使用率偏低，希望在仅剩少量提交次数前提高 A2 训练质量和 GPU 利用率。

发现的问题：

- `load_rec_data()` 对序列采用左侧 padding：
  - `[0, 0, 0, item_a, item_b, item_c]`
  - 最新交互在最右侧。
- 旧版 `GRU4Rec.forward()` 使用 `seq_len - 1` 取最后有效位置。
- 旧版 `SASRec.forward()` 也使用 `(item_seq != 0).sum() - 1` 取输出位置。
- 这对左侧 padding 是错误的：短序列会取到左侧 padding 区域，而不是最右侧真实交互。
- 结果是模型经常用“空白位置表示”做推荐，训练目标与真实序列状态错位。

修改文件：

- `framework/code/models.py`
- `framework/code/train.py`
- `framework/code/a2_model_hybrid_eval.py`

修改内容：

- `models.py`：
  - 修复 `GRU4Rec.forward()`：统一取 `output[:, -1, :]`。
  - 修复 `SASRec.forward()`：统一取 `output[:, -1, :]`。
  - 原因：当前数据是左侧 padding，最后一个时间步就是最近一次真实交互。
- `train.py`：
  - 新增 `--num_workers`。
  - A2 DataLoader 支持 `pin_memory=True`、`persistent_workers=True`。
  - GPU 训练时 Tensor 拷贝使用 `non_blocking=True`。
  - 目标是减少 CPU 数据加载和拷贝造成的 GPU 等待。
- `a2_model_hybrid_eval.py`：
  - 新增 A2 模型+启发式融合离线评估脚本。
  - 批量计算验证集模型 Top-N 分数。
  - 搜索 `model_weight`、`recent_n`、`user_weight` 等融合参数。
  - 只有离线超过当前启发式基线，才建议消耗线上提交次数。

验证命令：

```bash
cd /home/aliagent
python3 -m py_compile framework/code/models.py framework/code/train.py framework/code/a2_model_hybrid_eval.py framework/code/infer.py

python3 framework/code/train.py \
  --task task2 \
  --data_path framework/data/rec_data \
  --output_dir framework/output/exp012_smoke_a2_gru_ce \
  --epochs 1 \
  --model_type gru4rec \
  --embedding_dim 32 \
  --hidden_dim 64 \
  --num_layers 1 \
  --max_len 50 \
  --batch_size 1024 \
  --loss_type ce \
  --seq_col item_seq_raw \
  --eval_history_filter none \
  --lr 0.001 \
  --dropout 0.2 \
  --weight_decay 0.00001 \
  --patience 1 \
  --device cpu \
  --num_workers 0 \
  --log_interval 1
rm -rf framework/output/exp012_smoke_a2_gru_ce
```

验证结果：

- 语法检查通过。
- 修复后 GRU4Rec CE 仅 1 epoch 烟测：
  - `Val NDCG@10=0.3115`
  - `Hit@10=0.5112`
  - `MRR=0.2489`
- 相比此前 SASRec 多轮仍约 `0.24` 的结果，说明“取错序列位置”确实是 A2 模型训练失败的重要原因。

线上指标：

- 尚未提交。

是否保留：

- 保留。
- 下一步在 GPU 上正式训练 A2 GRU4Rec/SASRec，并使用 `a2_model_hybrid_eval.py` 判断是否纳入提交包。

---

## Exp-012：A2 修复版 GRU4Rec 训练与 hybrid 融合验证

日期：2026-06-25

目标：

- 在修复 A2 序列模型“取错最后状态”问题后，重新训练 GRU4Rec。
- 检查模型分数是否能在当前最强启发式共现方案上继续带来增益。

训练模型：

- `output/exp012_a2_gru_fixed_ce/best_model.pt`
- 模型：GRU4Rec
- 损失：CE
- 序列列：`item_seq_raw`
- 历史过滤：`none`
- GPU 训练，大 batch。

融合验证命令：

```bash
cd /home/aliagent/framework

CUDA_VISIBLE_DEVICES=0 python3 code/a2_model_hybrid_eval.py \
  --data_path data/rec_data \
  --checkpoint output/exp012_a2_gru_fixed_ce/best_model.pt \
  --device cuda \
  --batch_size 4096 \
  --val_ratio 0.1 \
  --seed 42 \
  --seq_col item_seq_raw \
  --recent_ns 8,10,12,15,20 \
  --model_weights 0,0.01,0.02,0.05,0.1,0.2,0.5 \
  --user_weights 0,0.01,0.02,0.03,0.04 \
  --model_topn 300 \
  --output_json output/exp012_a2_gru_hybrid_eval/results.json
```

本地验证结果：

- 最佳参数：
  - `recent_n=10`
  - `model_weight=0.5`
  - `cooccur_weight=1.0`
  - `user_weight=0.02`
  - `pop_weight=1.0`
  - `pop_penalty_weight=0.0`
  - `NDCG@10=0.547844`
- 同一验证集上纯启发式近似结果：
  - `model_weight=0`
  - `recent_n=10`
  - `user_weight=0.02`
  - `NDCG@10=0.547670`

结论：

- 修复后的 GRU4Rec 可以提供正向信号，但增益很小：
  - `0.547844 - 0.547670 = +0.000174`
- 该提升远小于 Exp-010 A1 Top-5 ensemble 的验证集提升。
- SASRec 结果全为 0，暂不使用 SASRec 融合。
- 如果今天只剩 2 次提交，下一次优先提交：
  - A1：Exp-010 Top-5 ensemble
  - A2：Exp-012 GRU hybrid 参数
- 但需要认识到：本次提交的主要预期收益来自 A1，A2 hybrid 只作为小幅增益尝试。

线上指标：

- 尚未提交。

是否保留：

- 保留 GRU hybrid 作为下一次提交候选。
- 不保留 SASRec。

---

## Tool-006：A2 hybrid 推理批量化

日期：2026-06-25

目标：

- 让正式生成 A2.csv 时的模型分数计算方式与 `a2_model_hybrid_eval.py` 更一致。
- 避免旧推理逻辑逐用户调用一次 GRU/SASRec，降低 Python 调度开销，提高 GPU 利用率。

修改文件：

- `framework/code/infer.py`

修改内容：

- 新增参数 `--model_topn`：
  - A2 模型融合时，每个用户保留模型 Top-N 候选分数。
  - 默认 `300`，与 Exp-012 离线融合验证保持一致。
- `infer_task2_v2()` 中新增批量模型分数预计算：
  - 先把全部测试用户序列按 batch 送入模型。
  - 一次性计算 `seq_repr @ item_embedding.T`。
  - 每个用户只保留模型 Top-N 分数，再交给 `rank_items()` 与共现、热门度、用户画像融合。

原理说明：

- 旧逻辑每个测试用户单独前向一次，GPU 每次只处理 1 条序列，利用率很低。
- 新逻辑按 `--batch_size` 批量前向，例如一次处理 4096 条序列。
- 这不会改变融合策略，只改变模型分数计算方式和保留候选数量。
- 因为离线验证也使用 `model_topn=300`，正式推理默认同样使用 300，避免本地/提交口径不一致。

验证命令：

```bash
cd /home/aliagent
python3 -m py_compile framework/code/infer.py framework/code/models.py framework/code/train.py framework/code/a2_model_hybrid_eval.py
```

验证结果：

- 语法检查通过。

线上指标：

- 尚未提交。

是否保留：

- 保留。下一次生成 A2 hybrid 提交文件时使用。

---

## Exp-013：A1 Top-5 ensemble + A2 GRU hybrid 线上提交

日期：2026-06-25

目标：

- 使用 Exp-010 的 A1 Top-5 ensemble 替换 Exp-008 A1。
- 使用 Exp-012 的 GRU hybrid 尝试提升 A2。

提交配置：

- A1：
  - `output/exp010_a1_grid_c6_gcn_h384_l2_symmetric_none_seed777/best_model.pt`
  - `output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed777/best_model.pt`
  - `output/exp010_a1_grid_c1_sage_h384_l2_symmetric_none_seed3407/best_model.pt`
  - `output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed42/best_model.pt`
  - `output/exp010_a1_grid_c7_sage_h256_l2_random_walk_none_seed777/best_model.pt`
- A2：
  - checkpoint：`output/exp012_a2_gru_fixed_ce/best_model.pt`
  - `rec_strategy=hybrid`
  - `seq_col=item_seq_raw`
  - `recent_n=10`
  - `model_weight=0.5`
  - `model_topn=300`
  - `user_weight=0.02`

线上指标：

- 提交时间：2026-06-25 17:08:59
- 总分：`0.5667`
- 分类任务分数：`0.6790`
- 推荐任务分数：`0.4545`

结论：

- A1 成功：
  - Exp-008 A1：`0.6496`
  - Exp-013 A1：`0.6790`
  - 提升 `+0.0294`
- A2 失败：
  - Exp-008/Exp-010 前的启发式 A2：`0.4647`
  - Exp-013 GRU hybrid A2：`0.4545`
  - 下降 `-0.0102`
- 总分仍提升：
  - Exp-008 总分：`0.5571`
  - Exp-013 总分：`0.5667`
  - 提升 `+0.0096`
- 下一次提交应保留 A1 Top-5 ensemble，同时把 A2 换回线上验证过的启发式 best。

是否保留：

- 保留 A1 Top-5。
- 不保留 GRU hybrid 作为线上提交方案。

---

## Exp-015：A2 历史频次信号与无效支线清理

日期：2026-06-25

目标：

- 不再保留线上已经验证下降的 GRU hybrid 提交路径。
- 不保留未验证出收益的 item feature 试验代码。
- 只新增一个可解释、低风险的 A2 信号：`item_seq_counts` 用户历史频次。

修改文件：

- `framework/code/rec_heuristics.py`
  - 新增 `parse_item_counts()`，解析 `item_seq_counts` 中的 `item:count`。
  - 新增历史频次加分逻辑，把用户长期高频 item 作为额外推荐信号。
  - `rank_items()` 新增 `history_counts` 和 `history_count_weight` 参数。
- `framework/code/a2_offline_eval.py`
  - 离线评估支持 `--history_count_weight`。
  - 每条验证样本从 `item_seq_counts` 解析当前用户历史频次并传入排序函数。
- `framework/code/a2_grid_search.py`
  - 网格搜索支持 `--history_count_weights`。
- `framework/code/infer.py`
  - 正式 A2 推理支持 `--history_count_weight`。
  - 生成提交时和离线评估使用同一套 `rank_items()` 逻辑。
- `framework/code/a2_model_hybrid_eval.py`
  - 删除。该脚本服务 GRU hybrid 离线融合，但 Exp-013 线上 A2 从 `0.4647` 降到 `0.4545`，不再作为保留路径。

原理说明：

- `item_seq_raw` 代表用户最近的行为顺序，适合做“最近兴趣”的共现召回。
- `item_seq_counts` 代表用户历史里各 item 出现次数，适合做“长期偏好/复购倾向”的加分。
- 旧启发式主要看最近 10 个历史 item 的共现，可能忽略用户反复购买但不在最近窗口里的 item。
- 新逻辑不是替换原策略，而是在原共现分和用户画像分之外，加一个受 `history_count_weight` 控制的频次分。

验证命令：

```bash
cd /home/aliagent
python3 -m py_compile framework/code/rec_heuristics.py framework/code/a2_grid_search.py framework/code/a2_offline_eval.py framework/code/infer.py framework/code/train.py framework/code/models.py

rg -n "a2_model_hybrid_eval|item_feature" framework/code

python3 framework/code/a2_offline_eval.py \
  --data_path framework/data/rec_data \
  --val_ratio 0.1 \
  --seed 42 \
  --topk 10 \
  --strategy history \
  --history_filter none \
  --seq_col item_seq_raw \
  --recent_n 10 \
  --user_weight 0.02 \
  --history_count_weight 0.0 \
  --output_json framework/output/exp015_a2_history_count_control/metrics.json

python3 framework/code/a2_offline_eval.py \
  --data_path framework/data/rec_data \
  --val_ratio 0.1 \
  --seed 42 \
  --topk 10 \
  --strategy history \
  --history_filter none \
  --seq_col item_seq_raw \
  --recent_n 10 \
  --user_weight 0.02 \
  --history_count_weight 0.4 \
  --output_json framework/output/exp015_a2_history_count_single/metrics.json
```

验证结果：

- 语法检查：通过。
- 残留引用检查：`framework/code` 下没有 `a2_model_hybrid_eval` 或 `item_feature` 残留。
- A2 对照配置：
  - `history_count_weight=0.0`
  - `NDCG@10=0.548205`
  - `Hit@10=0.786250`
  - `MRR=0.473521`
- A2 新配置：
  - `history_count_weight=0.4`
  - `NDCG@10=0.550125`
  - `Hit@10=0.788000`
  - `MRR=0.475525`
- 离线提升：
  - `NDCG@10 +0.001920`
  - `Hit@10 +0.001750`
  - `MRR +0.002004`

线上指标：

- 尚未提交。

结论：

- 保留 `history_count_weight`，作为下一次 A2 提交候选。
- 下一次提交建议使用：
  - A1：Exp-010 Top-5 ensemble。
  - A2：`rec_strategy=history, seq_col=item_seq_raw, recent_n=10, user_weight=0.02, history_count_weight=0.4`。

是否保留：

- 保留。

---

## Exp-016：A1 Correct and Smooth 后处理工具

日期：2026-06-25

目标：

- 按图半监督节点分类的强基线思路，引入 Correct and Smooth 后处理。
- 不重新训练 A1 模型，直接读取已有 checkpoint 的预测概率做图上传播。
- 先用本地已有旧 checkpoint 做真实数据烟测；最终高分搜索需要在 GPU 服务器上使用 Exp-010 Top-5 checkpoint。

修改文件：

- `framework/code/a1_correct_smooth.py`
  - 新增 A1 Correct and Smooth 独立工具。
  - 支持多 checkpoint logits 平均。
  - 支持固定验证集参数搜索。
  - 支持带 `--output_path` 生成 `A1.csv`。
  - C&S 传播邻接矩阵默认使用 PyTorch sparse tensor，避免 dense `N x N` 邻接矩阵浪费显存和计算。
- `framework/tests/test_a1_correct_smooth.py`
  - 新增核心单元测试。
  - 覆盖残差传播、标签锚点平滑、alpha=0 重启行为、全零行概率修复。

原理说明：

- 普通 GCN/SAGE 只把节点特征和邻居特征聚合进模型参数，训练标签只通过损失函数影响模型。
- A1 图的已标注节点之间同类边比例约 `0.7853`，说明图上标签同质性很强。
- Correct 阶段：
  - 在训练节点上计算 `真实 one-hot 标签 - 模型预测概率`。
  - 把这个残差沿图传播到邻居，相当于告诉模型“哪些区域容易被错分”。
- Smooth 阶段：
  - 把训练节点真实标签作为锚点。
  - 把修正后的预测概率沿图平滑，使相邻节点预测更一致。
- 本次烟测最佳参数中 `correct_weight=0`，说明在旧单模型上主要收益来自 Smooth；后续 Top-5 ensemble 需要继续验证 Correct 是否有额外增益。

测试命令：

```bash
cd /home/aliagent
python3 -m unittest framework.tests.test_a1_correct_smooth -v
python3 -m py_compile framework/code/a1_correct_smooth.py framework/code/utils.py framework/code/infer.py
```

测试结果：

- 单元测试：4 个测试全部通过。
- 语法检查：通过。

本地烟测命令：

```bash
cd /home/aliagent
python3 framework/code/a1_correct_smooth.py \
  --data_path framework/data/cls_data/A1.npz \
  --checkpoints framework/output/exp006_a1_sage_stable/best_model.pt \
  --device cpu \
  --val_ratio 0.1 \
  --split_seed 42 \
  --stratified_split \
  --cs_normalize random_walk \
  --correct_alphas 0.5,0.85 \
  --correct_iters 5,10 \
  --correct_weights 0,0.5,1.0 \
  --smooth_alphas 0.5,0.85 \
  --smooth_iters 5,10 \
  --smooth_weights 0,0.25,0.5 \
  --output_json framework/output/exp016_a1_cs_smoke/results.json
```

本地烟测结果：

- checkpoint：`framework/output/exp006_a1_sage_stable/best_model.pt`
- 原始验证准确率：`0.632877`
- C&S 最佳验证准确率：`0.649315`
- 本地提升：`+0.016438`
- 最佳小网格参数：
  - `correct_alpha=0.5`
  - `correct_iter=5`
  - `correct_weight=0.0`
  - `smooth_alpha=0.85`
  - `smooth_iter=5`
  - `smooth_weight=0.5`

GPU 服务器 Top-5 ensemble 搜索结果：

- 时间：2026-06-25
- checkpoint：
  - `output/exp010_a1_grid_c6_gcn_h384_l2_symmetric_none_seed777/best_model.pt`
  - `output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed777/best_model.pt`
  - `output/exp010_a1_grid_c1_sage_h384_l2_symmetric_none_seed3407/best_model.pt`
  - `output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed42/best_model.pt`
  - `output/exp010_a1_grid_c7_sage_h256_l2_random_walk_none_seed777/best_model.pt`
- 原始 Top-5 ensemble 验证准确率：`0.689498`
- C&S 最佳验证准确率：`0.700457`
- 本地固定验证集提升：`+0.010959`
- 当前最佳参数：
  - `correct_alpha=0.3`
  - `correct_iter=5`
  - `correct_weight=0.0`
  - `smooth_alpha=0.7`
  - `smooth_iter=10`
  - `smooth_weight=1.0`

补充结论：

- C&S 在强 Top-5 ensemble 上仍然有效，说明 A1 还有后处理收益。
- Top-20 中大量配置并列，且 `correct_weight` 影响很小，当前主要收益来自 Smooth 阶段。
- 下一步不应立刻浪费线上提交次数，应先做：
  - 多 split seed 稳健性验证。
  - 围绕 `smooth_alpha=0.7`、`smooth_iter=10` 的细网格搜索。
  - `random_walk` 与 `symmetric` 传播归一化对比。

多 split 验证结果与修正结论：

- 用户在 GPU 服务器上用同一组 Exp-010 Top-5 checkpoint 评估多个 `split_seed`：
  - `seed42.json`: `base=0.689498`, `cs=0.700457`, `gain=+0.010959`
  - `seed2024.json`: `base=0.979909`, `cs=0.979909`, `gain=+0.000000`
  - `seed2026.json`: `base=0.968950`, `cs=0.968950`, `gain=+0.000000`
  - `seed3407.json`: `base=0.969863`, `cs=0.969863`, `gain=+0.000000`
  - `seed777.json`: `base=0.971689`, `cs=0.971689`, `gain=+0.000000`
- 该多 split 结果无效：
  - Exp-010 Top-5 checkpoint 是围绕 `split_seed=42` 训练和筛选的。
  - 当用同一批 checkpoint 评估其他 split 时，其他 split 的验证节点大部分已出现在 checkpoint 训练集中。
  - 因此 `0.97` 左右的准确率是训练标签泄漏，不是泛化性能。
- 修正后的可信结论：
  - `split_seed=42` 仍然是当前这组 checkpoint 的可信 holdout。
  - C&S 在该可信 holdout 上从 `0.689498` 提升到 `0.700457`，这个 `+0.010959` 仍可作为有效信号。
  - 若要做真正多 split 验证，必须为每个 split 重新训练对应 checkpoint，不能复用只在 `split_seed=42` 下训练筛选出的模型。

细网格搜索结果：

- `random_walk` 归一化：
  - 原始验证准确率：`0.689498`
  - 最佳 C&S 验证准确率：`0.700457`
  - 最佳参数：
    - `correct_alpha=0.3`
    - `correct_iter=5`
    - `correct_weight=0.0`
    - `smooth_alpha=0.7`
    - `smooth_iter=7`
    - `smooth_weight=1.0`
  - 多个 `smooth_iter=7/10/12/15/20` 并列，说明该区域是稳定平台。
- `symmetric` 归一化：
  - 原始验证准确率：`0.689498`
  - 最佳 C&S 验证准确率：`0.699543`
  - 最佳参数：
    - `correct_alpha=0.3`
    - `correct_iter=5`
    - `correct_weight=0.0`
    - `smooth_alpha=0.6`
    - `smooth_iter=5`
    - `smooth_weight=1.0`
- 结论：
  - `random_walk` 比 `symmetric` 高 `+0.000914`。
  - 继续细搜没有突破 `0.700457`，当前 A1 C&S 参数可收敛为 `random_walk + smooth_alpha=0.7 + smooth_iter=7/10 + smooth_weight=1.0`。
  - 因为 `correct_weight=0.0`，当前实际有效方法是训练标签锚点平滑，不是残差纠错。

GPU 服务器下一步搜索命令：

```bash
cd /home/aliagent
git pull --ff-only

cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 python3 code/a1_correct_smooth.py \
  --data_path data/cls_data/A1.npz \
  --checkpoints \
    output/exp010_a1_grid_c6_gcn_h384_l2_symmetric_none_seed777/best_model.pt \
    output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed777/best_model.pt \
    output/exp010_a1_grid_c1_sage_h384_l2_symmetric_none_seed3407/best_model.pt \
    output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed42/best_model.pt \
    output/exp010_a1_grid_c7_sage_h256_l2_random_walk_none_seed777/best_model.pt \
  --device cuda \
  --val_ratio 0.1 \
  --split_seed 42 \
  --stratified_split \
  --cs_normalize random_walk \
  --correct_alphas 0.3,0.5,0.7,0.85,0.95 \
  --correct_iters 5,10,20,40 \
  --correct_weights 0,0.1,0.25,0.5,0.75,1.0 \
  --smooth_alphas 0.5,0.7,0.85,0.9,0.95 \
  --smooth_iters 3,5,10,20,40 \
  --smooth_weights 0,0.1,0.25,0.5,0.75,1.0 \
  --output_json output/exp016_a1_cs_top5_search/results.json
```

线上指标：

- 尚未提交。

结论：

- 保留。C&S 在旧 A1 单模型上已有明显本地收益。
- 下一步优先在 GPU 服务器上用 Exp-010 Top-5 ensemble 跑搜索。
- 如果 Top-5 ensemble 的 C&S 本地验证超过当前 `0.689498`，再生成新 A1 提交候选。

是否保留：

- 保留。

---

## Exp-019：A1 C&S + 稳妥 A2 线上提交

日期：2026-06-25

提交文件：

- `framework/output/exp019_submit_a1_cs_a2_best/prediction.zip`

提交配置：

- A1：
  - 使用 Exp-010 Top-5 ensemble。
  - C&S 参数：
    - `cs_normalize=random_walk`
    - `correct_alpha=0.3`
    - `correct_iter=5`
    - `correct_weight=0.0`
    - `smooth_alpha=0.7`
    - `smooth_iter=7`
    - `smooth_weight=1.0`
- A2：
  - 使用线上验证过的稳妥启发式路线。
  - `rec_strategy=history`
  - `seq_col=item_seq_raw`
  - `recent_n=10`
  - `history_filter=none`
  - `user_weight=0.02`
  - `history_count_weight=0`

线上指标：

- 提交时间：2026-06-25 22:06:36
- 总分：`0.5760`
- 分类任务分数：`0.6874`
- 推荐任务分数：`0.4647`

结论：

- A1：
  - Exp-013 A1：`0.6790`
  - Exp-019 A1：`0.6874`
  - 线上提升：`+0.0084`
  - C&S 有正收益，但线上提升低于本地 holdout 的 `+0.010959`。
- A2：
  - 仍为 `0.4647`，未提升。
- 总分：
  - Exp-013 总分：`0.5667`
  - Exp-019 总分：`0.5760`
  - 线上提升：`+0.0093`

下一步：

- A1 C&S 暂时收敛，不继续盲搜。
- 优先处理 A2：
  - 当前随机验证集和线上 test 分布差异大。
  - 先实现按 test 历史长度分布加权的 A2 离线验证。
  - 再针对 `len=0` 冷启动和 `len=1~3` 短历史用户做策略。

---

## Exp-020：A2 test-like 离线验证

日期：2026-06-26

目标：

- 修正 A2 随机验证集与线上 test 分布不一致的问题。
- 线上 test 中空历史和短历史用户占比很高，普通随机验证会高估长历史共现策略。

修改文件：

- `framework/code/a2_offline_eval.py`
  - 新增 `compute_bucket_weights()`。
  - 新增 `truncate_history_fields()`。
  - 新增 `apply_test_like_history_distribution()`。
  - 新增 `add_weighted_metrics()`。
  - 支持 `--test_like_eval` 和 `--sort_metric weighted_ndcg`。
- `framework/code/a2_grid_search.py`
  - 支持 test-like 截断验证和按 `weighted_ndcg` 排序。
- `framework/tests/test_a2_test_like_eval.py`
  - 覆盖桶权重、历史截断、`item_seq_counts` 同步截断、加权指标。

原理说明：

- test 的 `item_seq_raw` 长度分布：
  - `len=0`: `0.3515`
  - `len=1`: `0.1003`
  - `len=2-3`: `0.4474`
  - `len=4-10`: `0.0061`
  - `len>10`: `0.0947`
- 训练集随机验证大多是长历史用户，不符合线上分布。
- test-like 验证会从 test 长度分布抽样，把验证样本历史截断到对应长度。
- 截断历史时同步重写 `item_seq_counts`，避免频次字段偷看完整历史。

旧 A2 稳妥策略 test-like 指标：

- 配置：
  - `rec_strategy=history`
  - `seq_col=item_seq_raw`
  - `recent_n=10`
  - `user_weight=0.02`
  - `history_count_weight=0`
- 指标：
  - `NDCG@10=0.466722`
  - `weighted_NDCG@10=0.469048`
  - `Hit@10=0.712000`
  - `MRR=0.390515`
- 分桶：
  - `len=0`: `ndcg=0.331765`
  - `len=1`: `ndcg=0.489303`
  - `len=2-3`: `ndcg=0.548932`
  - `len=4-10`: `ndcg=0.481965`
  - `len>10`: `ndcg=0.578922`

结论：

- A2 最大短板是 `len=0` 冷启动用户。
- 下一步优先增强用户画像 prior，而不是继续训练 GRU/SASRec。

是否保留：

- 保留。

---

## Exp-021：A2 用户画像组合 prior

日期：2026-06-26

目标：

- 提升 A2 冷启动用户表现。
- 在单列用户画像统计之外，加入前缀组合画像统计。

数据诊断：

- `user.csv` 有 8 个画像列。
- 前缀组合覆盖与稀疏度：
  - 前 1 列：训练组 `65`，中位样本数 `285`
  - 前 2 列：训练组 `127`，中位样本数 `182`
  - 前 3 列：训练组 `663`，中位样本数 `16`，test 覆盖 `0.9994`
  - 前 4 列：训练组 `7750`，中位样本数 `2`，已经偏稀疏
- 因此采用 `3,2,1` 前缀组合，并设置 `min_count=5`。

修改文件：

- `framework/code/rec_heuristics.py`
  - 新增 `build_user_combo_profile_stats()`。
  - 新增 `get_user_combo_profile_counters()`。
  - `rank_items()` 新增 `user_combo_counters` 和 `user_combo_weight`。
- `framework/code/a2_offline_eval.py`
  - 支持 `--user_combo_weight`、`--user_combo_sizes`、`--user_combo_min_count`。
- `framework/code/a2_grid_search.py`
  - 支持组合画像权重搜索。
- `framework/code/infer.py`
  - 正式推理支持组合画像 prior。
- `framework/tests/test_rec_heuristics.py`
  - 覆盖组合画像计数和 `min_count` 过滤。

test-like 网格结果：

- 旧稳妥配置：
  - `weighted_NDCG@10=0.469048`
- 新最佳配置：
  - `rec_strategy=history`
  - `seq_col=item_seq_raw`
  - `recent_n=10`
  - `user_weight=0.02`
  - `user_combo_weight=0.2`
  - `user_combo_sizes=3,2,1`
  - `user_combo_min_count=5`
  - `history_count_weight=0`
  - `weighted_NDCG@10=0.479477`
- test-like 离线提升：
  - `+0.010429`

新配置分桶指标：

- `len=0`: `ndcg=0.360740`
- `len=1`: `ndcg=0.492172`
- `len=2-3`: `ndcg=0.549062`
- `len=4-10`: `ndcg=0.481965`
- `len>10`: `ndcg=0.577841`

与旧配置对比：

- `len=0`: `0.331765 -> 0.360740`，提升 `+0.028975`
- 主要收益来自冷启动用户，符合线上 test 分布。

验证：

- 单元测试：10 个测试全部通过。
- 语法检查：通过。
- 正式 A2 推理 smoke test：生成 10000 行 `A2.csv` 成功。
- 提交格式校验：通过。

线上指标：

- 尚未提交。

结论：

- 保留。该改动针对线上占比最高的 `len=0` 冷启动用户，离线 test-like 指标提升明显。
- 下一次提交建议组合：
  - A1：沿用 Exp-019 的 C&S A1。
  - A2：使用组合画像 prior 新配置。

---

## Exp-022：A1 C&S + A2 用户画像组合 prior 线上提交

日期：2026-06-26

提交配置：

- A1：
  - 沿用 Exp-019 的 C&S A1。
- A2：
  - `rec_strategy=history`
  - `seq_col=item_seq_raw`
  - `recent_n=10`
  - `history_filter=none`
  - `user_weight=0.02`
  - `user_combo_weight=0.2`
  - `user_combo_sizes=3,2,1`
  - `user_combo_min_count=5`
  - `history_count_weight=0`

线上指标：

- 提交时间：2026-06-26 14:47:17
- 总分：`0.5808`
- 分类任务分数：`0.6874`
- 推荐任务分数：`0.4742`

结论：

- A1 与 Exp-019 一致：`0.6874`。
- A2：
  - Exp-019：`0.4647`
  - Exp-022：`0.4742`
  - 线上提升：`+0.0095`
- 总分：
  - Exp-019：`0.5760`
  - Exp-022：`0.5808`
  - 线上提升：`+0.0048`
- test-like 验证和用户画像组合 prior 是有效方向。

下一步：

- 继续优化 A2。
- 冷启动画像 prior 已有效，下一阶段重点转向 `len=1~3` 短历史用户。
- 候选方向：item 类目转移，即用历史 item 的 `item.csv` 类目特征召回 target。

---

## Exp-023：A2 item 特征转移实验

日期：2026-06-26

目标：

- 按 Exp-022 后的方向，验证 `item.csv` 中物品特征是否能帮助 `len=1~3` 短历史用户。
- 思路：把测试用户历史 item 映射到 `i_cat_01/i_cat_02/i_bucket_01` 等物品特征，再用训练集统计“历史物品特征 -> target_iid”的转移热门度。

修改文件：

- `framework/code/rec_heuristics.py`
- `framework/code/a2_offline_eval.py`
- `framework/code/a2_grid_search.py`
- `framework/code/infer.py`
- `framework/tests/test_rec_heuristics.py`

修改内容：

- 新增 `build_item_feature_transition_stats()` 和 `get_item_feature_counters()`。
- `rank_items()` 支持 `item_feature_counters` 与 `item_feature_weight`。
- A2 离线评估、网格搜索、正式推理均支持：
  - `--item_feature_weight`
  - `--item_feature_cols`
  - `--item_feature_min_count`
  - `--item_feature_recent_n`
- 新增单元测试覆盖：
  - 物品特征转移能返回同类目 target 计数器。
  - 样本数不足的特征分组会被过滤。

运行命令：

```bash
cd /home/aliagent
python3 -m unittest framework.tests.test_a1_correct_smooth framework.tests.test_a2_test_like_eval framework.tests.test_rec_heuristics -v
python3 -m py_compile framework/code/rec_heuristics.py framework/code/a2_offline_eval.py framework/code/a2_grid_search.py framework/code/infer.py
python3 framework/code/a2_grid_search.py \
  --data_path framework/data/rec_data \
  --val_ratio 0.1 \
  --seed 42 \
  --topk 10 \
  --seq_cols item_seq_raw \
  --recent_ns 3,5,10 \
  --strategies history \
  --history_filters none \
  --user_weights 0.02 \
  --user_combo_weights 0.2 \
  --user_combo_sizes 3,2,1 \
  --user_combo_min_count 5 \
  --item_feature_weights 0,0.02,0.05,0.1,0.2,0.5 \
  --item_feature_cols i_cat_01,i_cat_02,i_bucket_01 \
  --item_feature_min_count 20 \
  --pop_penalty_weights 0 \
  --history_count_weights 0 \
  --test_like_eval \
  --sort_metric weighted_ndcg \
  --top_results 30 \
  --output_json framework/output/exp023_a2_item_feature_grid/results.json
```

本地指标：

- 单元测试：13 项通过。
- 语法检查：通过。
- 最优结果仍为 `item_feature_weight=0`：
  - `weighted_NDCG@10=0.479477`
  - `NDCG@10=0.477359`
- 加入物品特征后均略降，例如：
  - `item_feature_weight=0.02`: `weighted_NDCG@10=0.479311`
  - `item_feature_weight=0.05`: `weighted_NDCG@10=0.479281`

线上指标：

- 未提交。

结论：

- 不保留为提交配置。
- 物品特征转移在当前规则融合中没有贡献，可能原因是 item 类别太粗，和现有热门/共现信号重复。

是否保留：

- 代码保留为可选开关，默认权重为 0，不影响既有提交配置。

下一步：

- 转向历史共现近因衰减，验证最近行为是否应比更早行为权重更高。

---

## Exp-024：A2 历史共现近因衰减

日期：2026-06-26

目标：

- 改进 A2 `history` 策略。旧逻辑对最近 10 个历史 item 基本等权，而序列推荐中越近的行为通常越能解释下一个 target。

修改文件：

- `framework/code/rec_heuristics.py`
- `framework/code/a2_offline_eval.py`
- `framework/code/a2_grid_search.py`
- `framework/code/infer.py`
- `framework/tests/test_rec_heuristics.py`

修改内容：

- `_add_cooccur_scores()` 新增 `decay` 参数。
- `rank_items()` 新增 `cooccur_decay`，默认 `1.0`，完全兼容旧行为。
- A2 离线评估、网格搜索、正式推理均支持：
  - `--cooccur_decay`
  - `--cooccur_decays`
- 新增单元测试：较旧 item 共现计数更高、较新 item 共现计数略低时，开启衰减后应优先推荐最近 item 对应 target。

运行命令：

```bash
cd /home/aliagent
python3 -m unittest framework.tests.test_a1_correct_smooth framework.tests.test_a2_test_like_eval framework.tests.test_rec_heuristics -v
python3 -m py_compile framework/code/rec_heuristics.py framework/code/a2_offline_eval.py framework/code/a2_grid_search.py framework/code/infer.py
python3 framework/code/a2_grid_search.py \
  --data_path framework/data/rec_data \
  --val_ratio 0.1 \
  --seed 42 \
  --topk 10 \
  --seq_cols item_seq_raw \
  --recent_ns 10 \
  --cooccur_decays 1.0,0.995,0.99,0.985,0.98,0.975,0.97,0.965,0.96,0.955,0.95 \
  --strategies history \
  --history_filters none \
  --user_weights 0.02 \
  --user_combo_weights 0.2 \
  --user_combo_sizes 3,2,1 \
  --user_combo_min_count 5 \
  --item_feature_weights 0 \
  --pop_penalty_weights 0 \
  --history_count_weights 0 \
  --test_like_eval \
  --sort_metric weighted_ndcg \
  --top_results 20 \
  --output_json framework/output/exp024_a2_cooccur_decay_refine/results.json
```

本地指标：

- 单元测试：13 项通过。
- 语法检查：通过。
- Exp-022 对应离线配置：
  - `cooccur_decay=1.0`
  - `weighted_NDCG@10=0.479477`
- 细化搜索最佳：
  - `cooccur_decay=0.96`
  - `weighted_NDCG@10=0.479857`
  - `NDCG@10=0.477695`
  - `Hit@10=0.722000`
  - `MRR=0.401636`
- 分桶对比：
  - `len=0` 不变：`0.360740`
  - `len=2-3`: `0.549062 -> 0.549230`
  - `len>10`: `0.577841 -> 0.581059`

线上指标：

- 未提交。

结论：

- 保留为候选配置。收益很小，但方向稳定且不影响空历史用户。
- 不建议单独为该小收益消耗提交次数，应与下一次 A1 或 A2 更大改动一起提交。

是否保留：

- 保留。

下一步：

- 微调用户画像组合 prior 权重。

---

## Exp-025：A2 用户画像 all 组合实验

日期：2026-06-26

目标：

- 验证是否应把用户画像组合从前缀组合扩展为所有列组合。
- 直觉：`user.csv` 有 8 个画像列，最有用的列不一定只在前几列。

修改文件：

- `framework/code/rec_heuristics.py`
- `framework/code/a2_offline_eval.py`
- `framework/code/a2_grid_search.py`
- `framework/code/infer.py`
- `framework/tests/test_rec_heuristics.py`

修改内容：

- `build_user_combo_profile_stats()` 新增 `combo_mode`：
  - `prefix`: 默认旧行为，只使用前缀组合。
  - `all`: 枚举指定长度的所有列组合。
- A2 离线评估、网格搜索、正式推理均支持：
  - `--user_combo_mode prefix/all`
- 新增单元测试：`all` 模式应能使用非前缀画像列。

运行命令：

```bash
cd /home/aliagent
python3 framework/code/a2_grid_search.py \
  --data_path framework/data/rec_data \
  --val_ratio 0.1 \
  --seed 42 \
  --topk 10 \
  --seq_cols item_seq_raw \
  --recent_ns 10 \
  --cooccur_decays 1.0,0.96 \
  --strategies history \
  --history_filters none \
  --user_weights 0.02 \
  --user_combo_weights 0.02,0.05,0.1,0.2 \
  --user_combo_sizes 2,1 \
  --user_combo_mode all \
  --user_combo_min_count 20 \
  --item_feature_weights 0 \
  --pop_penalty_weights 0 \
  --history_count_weights 0 \
  --test_like_eval \
  --sort_metric weighted_ndcg \
  --top_results 30 \
  --output_json framework/output/exp025_a2_user_combo_all_grid/results_s2_s1_min20.json
```

本地指标：

- 最佳 all 组合：
  - `user_combo_mode=all`
  - `user_combo_sizes=2,1`
  - `user_combo_min_count=20`
  - `user_combo_weight=0.1`
  - `cooccur_decay=0.96`
  - `weighted_NDCG@10=0.471957`
- 明显低于当前 prefix 最优 `0.479857`。

线上指标：

- 未提交。

结论：

- 不采用 `all` 组合。它枚举了太多画像分组，平均融合后引入噪声并稀释有效先验。
- 代码保留为可选模式，默认仍为 `prefix`。

是否保留：

- 仅保留可选参数，不作为提交配置。

下一步：

- 继续微调 prefix 组合。

---

## Exp-029：A2 共现衰减 + 用户组合权重微调候选

日期：2026-06-26

目标：

- 在 Exp-022 有效配置上做小范围参数微调，形成下一次提交候选 A2。

运行命令：

```bash
cd /home/aliagent
python3 framework/code/a2_grid_search.py \
  --data_path framework/data/rec_data \
  --val_ratio 0.1 \
  --seed 42 \
  --topk 10 \
  --seq_cols item_seq_raw \
  --recent_ns 10 \
  --cooccur_decays 0.96 \
  --strategies history \
  --history_filters none \
  --user_weights 0,0.01,0.02,0.03,0.05 \
  --user_combo_weights 0.18,0.2,0.22 \
  --user_combo_sizes 3,2,1 \
  --user_combo_mode prefix \
  --user_combo_min_count 5 \
  --item_feature_weights 0 \
  --pop_penalty_weights 0 \
  --history_count_weights 0 \
  --test_like_eval \
  --sort_metric weighted_ndcg \
  --top_results 25 \
  --output_json framework/output/exp029_a2_user_weight_grid/results.json
```

本地指标：

- 最佳配置：
  - `seq_col=item_seq_raw`
  - `recent_n=10`
  - `cooccur_decay=0.96`
  - `rec_strategy=history`
  - `history_filter=none`
  - `user_weight=0.02`
  - `user_combo_weight=0.18`
  - `user_combo_sizes=3,2,1`
  - `user_combo_mode=prefix`
  - `user_combo_min_count=5`
  - `history_count_weight=0`
- 指标：
  - `weighted_NDCG@10=0.479879`
  - `NDCG@10=0.477715`
  - `Hit@10=0.721500`
  - `MRR=0.401798`
- 对比 Exp-022 离线配置：
  - `weighted_NDCG@10=0.479477`
  - 本地提升：`+0.000402`

候选 A2 生成命令：

```bash
python3 framework/code/infer.py \
  --task task2 \
  --data_path framework/data/rec_data \
  --output_path framework/output/exp029_a2_decay_combo018_smoke/A2.csv \
  --rec_strategy history \
  --seq_col item_seq_raw \
  --recent_n 10 \
  --cooccur_decay 0.96 \
  --history_filter none \
  --user_weight 0.02 \
  --user_combo_weight 0.18 \
  --user_combo_sizes 3,2,1 \
  --user_combo_mode prefix \
  --user_combo_min_count 5 \
  --history_count_weight 0 \
  --user_profile_cols auto \
  --topk 10 \
  --device cpu
```

格式检查：

- `A2.csv` 通过格式校验。
- 与 Exp-022 A2 相比，`1551 / 10000` 个用户推荐串发生变化，变化比例 `15.51%`。

线上指标：

- 尚未提交。

结论：

- 可作为下一次提交候选，但收益较小。
- 更大的分数空间仍在 A1；后续应优先提升 A1 训练/集成。

是否保留：

- 保留为候选。

---

## Exp-030：A1 C&S + A2 共现衰减微调线上提交

日期：2026-06-26

提交文件：

- `/home/aliagent/framework/output/exp030_submit_a1cs_a2_decay_combo018/prediction.zip`

提交配置：

- A1：
  - 沿用 Exp-019 的 A1 C&S。
  - `cs_normalize=random_walk`
  - `correct_alpha=0.3`
  - `correct_iter=5`
  - `correct_weight=0`
  - `smooth_alpha=0.7`
  - `smooth_iter=7`
  - `smooth_weight=1.0`
- A2：
  - `rec_strategy=history`
  - `seq_col=item_seq_raw`
  - `recent_n=10`
  - `cooccur_decay=0.96`
  - `history_filter=none`
  - `user_weight=0.02`
  - `user_combo_weight=0.18`
  - `user_combo_sizes=3,2,1`
  - `user_combo_mode=prefix`
  - `user_combo_min_count=5`
  - `history_count_weight=0`

线上指标：

- 提交时间：2026-06-26 15:34:21
- 总分：`0.5810`
- 分类任务分数：`0.6874`
- 推荐任务分数：`0.4746`

结论：

- A2：
  - Exp-022：`0.4742`
  - Exp-030：`0.4746`
  - 线上提升：`+0.0004`
- 总分：
  - Exp-022：`0.5808`
  - Exp-030：`0.5810`
  - 线上提升：`+0.0002`
- A2 启发式微调已经接近平台，继续只调 A2 小权重性价比很低。
- 与第一名相比，最大差距在 A1：
  - 当前 A1：`0.6874`
  - 第一名 A1：约 `0.76845`
  - 差距约 `0.081`

下一步：

- 暂停 A2 小幅参数搜索。
- A1 优先方向：
  - 稀疏 GAT，替代原 dense GAT，形成新模型族。
  - 高置信伪标签 C&S，作为可选后处理。

---

## Tool-006：A1 稀疏 GAT 与伪标签 C&S

日期：2026-06-26

目标：

- 按官方提分方向继续提升 A1。
- 原代码已有 `gat`，但实现会构造 `N x N` 注意力矩阵；A1 有 13,752 个节点，这种 dense 注意力计算不合理，容易慢或 OOM。
- 新增 `gat_sparse`：只在真实边上计算注意力，复杂度从 `O(N^2)` 降到 `O(E)`。
- 新增高置信伪标签 C&S：在已有训练标签锚点之外，把模型高置信测试节点作为软锚点参与平滑。

修改文件：

- `framework/code/models.py`
- `framework/code/train.py`
- `framework/code/infer.py`
- `framework/code/a1_correct_smooth.py`
- `framework/tests/test_models.py`
- `framework/tests/test_a1_correct_smooth.py`
- `record.md`

修改内容：

- 新增 `SparseGATLayer`：
  - 从稀疏邻接矩阵提取真实边。
  - 自动补自环。
  - 对每个目标节点的入边做 attention softmax。
  - 用 `scatter_add_` 聚合边消息。
- `GNNClassifier` 支持：
  - `model_type="gat_sparse"`
  - `gat_heads`
- `train.py` 支持：
  - `--model_type gat_sparse`
  - `--adj_format sparse`
  - `--num_heads`
- `infer.py` 支持加载 `gat_sparse` checkpoint。
- `a1_correct_smooth.py` 支持伪标签搜索和推理：
  - `--pseudo_thresholds`
  - `--pseudo_weights`
  - `--pseudo_threshold`
  - `--pseudo_weight`
- 搜索时自动包含 no-pseudo 对照，避免误判伪标签收益。

测试命令：

```bash
cd /home/aliagent
python3 -m unittest framework.tests.test_models framework.tests.test_a1_correct_smooth framework.tests.test_a2_test_like_eval framework.tests.test_rec_heuristics -v
python3 -m py_compile framework/code/models.py framework/code/train.py framework/code/infer.py framework/code/a1_correct_smooth.py framework/code/a2_offline_eval.py framework/code/a2_grid_search.py framework/code/rec_heuristics.py
python3 framework/code/train.py \
  --task task1 \
  --data_path framework/data/cls_data/A1.npz \
  --output_dir framework/output/smoke_a1_gat_sparse \
  --epochs 1 \
  --model_type gat_sparse \
  --hidden_dim 64 \
  --num_layers 2 \
  --num_heads 4 \
  --lr 0.005 \
  --dropout 0.3 \
  --weight_decay 0.0005 \
  --patience 5 \
  --normalize none \
  --adj_format sparse \
  --feature_norm none \
  --stratified_split \
  --device cpu \
  --log_interval 1
```

测试结果：

- 单元测试：16 项通过。
- 语法检查：通过。
- `gat_sparse` 1 epoch 烟测通过：
  - 输出目录：`framework/output/smoke_a1_gat_sparse`
  - `best_model.pt` 正常生成。

伪标签 C&S 烟测：

```bash
python3 framework/code/a1_correct_smooth.py \
  --data_path framework/data/cls_data/A1.npz \
  --checkpoints framework/output/exp006_a1_sage_stable/best_model.pt \
  --device cpu \
  --val_ratio 0.1 \
  --split_seed 42 \
  --stratified_split \
  --cs_normalize random_walk \
  --correct_alphas 0.3 \
  --correct_iters 5 \
  --correct_weights 0 \
  --smooth_alphas 0.7,0.85 \
  --smooth_iters 5 \
  --smooth_weights 1.0 \
  --pseudo_thresholds 0.9,0.95 \
  --pseudo_weights 0.5,1.0 \
  --output_json framework/output/smoke_a1_pseudo_cs_with_nopseudo/results.json
```

本地结果：

- 旧 checkpoint 原始验证准确率：`0.632877`
- no-pseudo C&S 最佳：`0.666667`
- pseudo C&S 最佳：
  - `pseudo_threshold=0.95`
  - `pseudo_weight=1.0`
  - `pseudo_count=1844`
  - `val_acc=0.667580`
- 伪标签相比 no-pseudo 小幅提升：`+0.000913`

结论：

- `gat_sparse` 是下一轮 GPU 训练重点，因为它提供新的模型族和集成多样性。
- 伪标签 C&S 是低风险小增益后处理，建议在强 Top-5/Top-N ensemble 上搜索验证。
- 当前不能承诺超过第一名；要接近第一名，必须先把 A1 从 `0.6874` 推到 `0.74+`，仅靠 A2 已不现实。

下一步 GPU 实验：

- 训练多个 `gat_sparse` checkpoint。
- 用 `a1_ensemble_eval.py` 比较 GCN/SAGE/GAT sparse 的混合集成。
- 在最佳混合集成上重新跑普通 C&S 和伪标签 C&S。

---

## Exp-031：A1 稀疏 GAT 训练与混合集成评估

日期：2026-06-26

目标：

- 验证 Tool-006 新增的 `gat_sparse` 是否能成为比 Exp-010 GCN/SAGE 更强的新模型族。
- 将 GAT sparse 与已有 GCN/SAGE checkpoint 做混合集成评估。

运行环境：

- GPU 服务器。

训练配置：

- `model_type=gat_sparse`
- `adj_format=sparse`
- `normalize=none`
- `num_layers=2`
- `num_heads=4`
- 搜索两组 hidden：
  - `hidden_dim=256`
  - `hidden_dim=384`

用户返回结果：

单模型固定验证集准确率：

- `0.700457`：`exp031_a1_gat_sparse_h256_heads4_seed2026`
- `0.696804`：`exp031_a1_gat_sparse_h256_heads4_seed3407`
- `0.694977`：`exp031_a1_gat_sparse_h256_heads4_seed42`
- `0.686758`：`exp031_a1_gat_sparse_h384_heads4_seed3407`
- `0.684932`：`exp031_a1_gat_sparse_h384_heads4_seed42`
- `0.682192`：`exp031_a1_gat_sparse_h384_heads4_seed777`
- `0.677626`：Exp-010 最强 GCN

等权 Top-K 集成结果：

- Top-1：`0.700457`
- Top-2：`0.695890`
- Top-3：`0.694977`
- Top-4：`0.689498`
- Top-5：`0.689498`
- Top-10：`0.693151`

结论：

- `gat_sparse_h256` 成功，单模型已经达到 `0.700457`，等于此前 Exp-010 Top-5 + C&S 的本地验证准确率。
- `hidden_dim=384` 明显不如 `hidden_dim=256`，说明该模型族过大后可能过拟合或训练不稳定。
- 等权集成无效，Top-K 平均会把最强 GAT 拉低。
- 后续必须改用：
  - 单模型 GAT + C&S / 伪标签 C&S。
  - 贪心加权 ensemble，而不是 Top-K 等权平均。

是否保留：

- 保留 `gat_sparse_h256_heads4_seed2026` 作为当前 A1 最强单模型候选。

下一步：

- 跑 `a1_correct_smooth.py`，对最强 GAT 单模型搜索普通 C&S 与伪标签 C&S。
- 跑 `a1_ensemble_eval.py --greedy_max_size`，做贪心加权集成搜索。

---

## Tool-007：A1 贪心加权集成评估

日期：2026-06-26

目标：

- 解决 Exp-031 中“等权 Top-K 集成拉低最强模型”的问题。
- 让 ensemble 只在验证集准确率真实提升时加入新模型，并搜索新模型权重。

修改文件：

- `framework/code/a1_ensemble_eval.py`
- `framework/tests/test_a1_ensemble_eval.py`

修改内容：

- 新增 `greedy_weighted_ensemble()`：
  - 从验证集最强单模型开始。
  - 每一步尝试加入一个尚未选择的 checkpoint。
  - 搜索新模型权重 `w`，融合公式：
    - `new_logits = (1 - w) * current_logits + w * candidate_logits`
  - 只有验证准确率提升超过 `--greedy_min_gain` 才接受。
- `a1_ensemble_eval.py` 新增参数：
  - `--greedy_max_size`
  - `--greedy_weights`
  - `--greedy_min_gain`
- 新增测试：
  - 互补模型应能通过加权融合提升准确率。
  - 有害模型不能为了凑数量被加入。

测试命令：

```bash
cd /home/aliagent
python3 -m unittest framework.tests.test_a1_ensemble_eval framework.tests.test_models framework.tests.test_a1_correct_smooth -v
python3 -m py_compile framework/code/a1_ensemble_eval.py framework/code/models.py framework/code/train.py framework/code/infer.py framework/code/a1_correct_smooth.py
```

测试结果：

- 单元测试：8 项通过。
- 语法检查：通过。

结论：

- 工具保留。
- 下一步在 GPU 服务器上对 Exp010 + Exp031 checkpoint 跑贪心加权 ensemble。

---

## Exp-032：A1 贪心加权集成与 GAT 单模型 C&S

日期：2026-06-26

目标：

- 验证 Exp-031 的结论：A1 不能简单等权堆模型，需要用验证集控制模型权重。
- 在当前最强 `gat_sparse_h256_heads4_seed2026` 上搜索 Correct and Smooth 后处理。

用户返回结果：

贪心加权集成：

- Step-1：
  - `val_acc=0.700457`
  - 加入 `exp031_a1_gat_sparse_h256_heads4_seed2026`
  - 权重 `1.0`
- Step-2：
  - `val_acc=0.703196`
  - 加入 `exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed777`
  - 新加入权重 `0.05`
  - 最终权重：
    - `exp031_a1_gat_sparse_h256_heads4_seed2026`: `0.95`
    - `exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed777`: `0.05`

GAT 单模型 C&S：

- 原始模型验证准确率：`0.700457`
- C&S 最优验证准确率：`0.710502`
- 最优参数：
  - Correct：`alpha=0.3, iter=5, weight=0.0`
  - Smooth：`alpha=0.7, iter=5, weight=0.75`
- 伪标签参数没有带来额外提升，因为无伪标签配置和多组伪标签配置并列 `0.710502`。

结论：

- 当前 A1 最有效提升来自 `gat_sparse_h256` + Smooth 型 C&S。
- Correct 阶段最佳权重为 `0.0`，说明训练残差传播没有额外帮助；核心是把训练标签作为锚点做标签平滑。
- 伪标签目前不作为默认提交策略，因为验证集显示它没有贡献，且线上风险更高。
- 贪心加权集成提升了原始 logits，但尚未和 C&S 联合验证。

下一步：

- 给 `a1_correct_smooth.py` 增加 `--checkpoint_weights`，让 `0.95/0.05` 加权 logits 能进入 C&S 搜索。
- 对 `GAT 0.95 + GCN 0.05` 跑 C&S，如果验证准确率超过 `0.710502`，再生成下一版 A1；否则保持单 GAT + C&S。

---

## Tool-008：A1 C&S 支持 checkpoint 加权

日期：2026-06-26

目标：

- 让 Exp-032 搜出的贪心加权 ensemble 可以继续接入 Correct and Smooth。
- 避免把强 GAT 和弱模型等权平均，保留 `0.95/0.05` 这种小比例互补信号。

修改文件：

- `framework/code/a1_correct_smooth.py`
- `framework/tests/test_a1_correct_smooth.py`

修改内容：

- 新增命令行参数 `--checkpoint_weights`：
  - 为空时保持原有等权行为。
  - 传入逗号分隔权重时，数量必须和 checkpoint 数量一致。
  - 权重会自动归一化，例如 `95,5` 等价于 `0.95,0.05`。
- `_average_model_probs()` 从固定等权 logits 平均改为加权 logits 平均。
- JSON 结果中记录归一化后的 `checkpoint_weights`，便于复盘。
- 新增单元测试：
  - 显式权重归一化。
  - 空权重退化为等权。
  - 权重数量不匹配时报错。

测试命令：

```bash
cd /home/aliagent
python3 -m unittest framework.tests.test_a1_correct_smooth framework.tests.test_a1_ensemble_eval framework.tests.test_models -v
python3 -m py_compile framework/code/a1_correct_smooth.py framework/code/a1_ensemble_eval.py framework/code/models.py framework/code/train.py framework/code/infer.py
```

测试结果：

- 单元测试：11 项通过。
- 语法检查：通过。

结论：

- 工具保留。
- 下一步在 GPU 上运行 Exp-033：`GAT 0.95 + GCN 0.05 + C&S`。

---

## Exp-033：A1 加权 GAT+GCN + C&S 搜索

日期：2026-06-26

目标：

- 将 Exp-032 搜出的贪心加权 logits 组合接入 Correct and Smooth。
- 验证 `0.95 * GAT + 0.05 * GCN` 是否不仅原始验证准确率更高，而且经过 C&S 后仍优于单 GAT。

输入 checkpoint：

- `output/exp031_a1_gat_sparse_h256_heads4_seed2026/best_model.pt`
- `output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed777/best_model.pt`

配置：

- `checkpoint_weights=95,5`
- `cs_normalize=random_walk`
- `correct_alpha=0.3`
- `correct_iter=5`
- `correct_weight=0.0`
- `smooth_alpha` 搜索：`0.6,0.65,0.7,0.75,0.8`
- `smooth_iter` 搜索：`3,5,7,10,15`
- `smooth_weight` 搜索：`0.5,0.6,0.7,0.75,0.8,0.9,1.0`
- `pseudo_thresholds=0.95,0.97`
- `pseudo_weights=0.5,1.0`

用户返回结果：

- 原始加权 logits 验证准确率：`0.703196`
- C&S 最优验证准确率：`0.711416`
- 最优参数：
  - Correct：`alpha=0.3, iter=5, weight=0.0`
  - Smooth：`alpha=0.7, iter=5, weight=0.75`
  - 伪标签：无伪标签与多组伪标签并列最优，因此默认不使用伪标签。

对比：

- 单 GAT 原始验证准确率：`0.700457`
- 单 GAT + C&S：`0.710502`
- 加权 GAT+GCN 原始验证准确率：`0.703196`
- 加权 GAT+GCN + C&S：`0.711416`

结论：

- 加权 GAT+GCN + C&S 是当前 A1 本地验证最优方案。
- 增益来自两个部分：
  - GCN 以 `5%` 权重提供少量互补 logits，使原始验证准确率提升 `+0.002739`。
  - Smooth C&S 再提升 `+0.008220`。
- 伪标签没有贡献，不作为默认提交策略。
- 该方案可作为下一次 A1 提交候选，但本地相对单 GAT C&S 只高 `+0.000914`，线上未必稳定放大。

下一步：

- 生成 `exp034_submit_a1_weighted_cs_a2_exp030/prediction.zip`，A1 使用 Exp-033，A2 沿用 Exp-030。
- 如果当天提交次数紧张，优先继续跑 A2 或更强 A1 模型，再决定是否提交。

---

## Exp-035/037：A2 冷启动用户画像探查

日期：2026-06-26

目标：

- 只剩一次提交机会，不能再为 A2 小权重微调消耗提交。
- 先诊断 A2 当前瓶颈，重点看测试集中占比很高的空历史/短历史用户。

数据诊断：

- 训练用户数：`40000`
- 测试用户数：`10000`
- 训练/测试用户重叠：`0`
- 测试 `item_seq_raw` 历史长度分布：
  - 空历史：`3515 / 10000`
  - 长度 1：`1003 / 10000`
  - 长度 2-3：`4474 / 10000`
  - 长度 4-10：`61 / 10000`
  - 长度 >10：`947 / 10000`

结论：

- A2 不是记住老用户，而是对新用户做短历史/冷启动推荐。
- 空历史用户完全没有 item 序列，共现特征不可用，只能依赖用户画像和全局热门。

运行实验：

- `exp035_a2_user_combo_long_prefix`：
  - 将用户画像组合从 `3,2,1` 扩展到 `5,4,3,2,1`
  - 将 `user_combo_min_count` 降到 `3`
  - 搜索多个 `user_weight` 和 `user_combo_weight`

本地结果：

- 最佳 `weighted_NDCG@10=0.477944`
- 低于 Exp-029/030 当前 A2 候选 `0.479879`

结论：

- 更长、更细的用户画像前缀组合引入噪声，没有提升。
- A2 冷启动靠画像继续细分的空间有限，当前最后一次提交不应押注该方向。

是否保留：

- 不作为提交配置。
- 不删除代码，因为 `prefix/all` 组合仍是可复现实验能力。

---

## Tool-009：A1 Exp-038 稀疏 GAT GPU 网格脚本

日期：2026-06-26

目标：

- 在只剩一次提交机会的情况下，寻找比 Exp-033 更明显的 A1 提升。
- 不再继续微调已经平台化的 A2 小权重，而是扩大 `gat_sparse` 的有效配置搜索。

修改文件：

- `framework/scripts/run_exp038_a1_gat_grid.sh`

脚本逻辑：

- 基于 Exp-031 的有效经验固定：
  - `model_type=gat_sparse`
  - `num_layers=2`
  - `normalize=none`
  - `adj_format=sparse`
  - `feature_norm=none`
  - `class_weight=none`
  - `val_ratio=0.1`
  - `split_seed=42`
  - `stratified_split`
- 搜索低风险邻域：
  - `hidden_dim=224/256/288`
  - `heads=4`
  - `dropout=0.40/0.45/0.50`
  - `lr=0.003/0.005`
  - seeds：`42/777/2026/3407`
- 少量探索多头模型：
  - `hidden_dim=256, heads=8`
  - `hidden_dim=320, heads=8`
  - seeds：`2026/3407`
- 每个配置保存到独立目录。
- 若目录下已有 `best_model.pt`，自动跳过，方便中断后续跑。
- 训练完成后自动运行 `a1_ensemble_eval.py`：
  - 单模型排序
  - Top-K 等权集成
  - 贪心加权集成

验证：

```bash
cd /home/aliagent
bash -n framework/scripts/run_exp038_a1_gat_grid.sh
```

验证结果：

- Shell 语法检查通过。

下一步：

- 在 GPU 服务器运行：

```bash
cd /home/aliagent
git pull --ff-only
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda ./scripts/run_exp038_a1_gat_grid.sh
```

- 目标：
  - 找到原始验证准确率超过 `0.71` 的单模型。
  - 或找到贪心加权 ensemble 明显超过 Exp-033 原始 `0.703196` 的组合。

---

## Exp-038：A1 稀疏 GAT GPU 网格结果

日期：2026-06-27

目标：

- 通过更系统的 `gat_sparse` 小网格寻找比 Exp-033 更强的 A1 候选。

用户返回结果：

单模型固定验证集 Top 结果：

- `0.700457`：`h256_heads4_d040_lr005_wd5e4_seed42`
- `0.700457`：`h256_heads4_d050_lr003_wd5e4_seed2026`
- `0.698630`：`h288_heads4_d045_lr005_wd5e4_seed777`
- `0.697717`：`h224_heads4_d045_lr005_wd5e4_seed42`
- `0.697717`：`h288_heads4_d045_lr005_wd5e4_seed42`

等权集成：

- Top-1：`0.700457`
- Top-2：`0.705936`
- Top-3：`0.709589`
- Top-5：`0.705936`
- Top-8：`0.703196`
- Top-12：`0.704110`
- Top-20：`0.696804`

贪心加权集成：

- Step-1：`0.700457`
  - `h256_heads4_d040_lr005_wd5e4_seed42`
- Step-2：`0.704110`
  - 加入 `h224_heads4_d045_lr005_wd5e4_seed42`
  - 权重：`0.8 / 0.2`
- Step-3：`0.706849`
  - 加入 `h256_heads8_d045_lr005_wd5e4_seed3407`
  - 权重：
    - `h256_heads4_d040_lr005_wd5e4_seed42`: `0.64`
    - `h224_heads4_d045_lr005_wd5e4_seed42`: `0.16`
    - `h256_heads8_d045_lr005_wd5e4_seed3407`: `0.20`

结论：

- Exp-038 没有找到明显更强的单模型；最强单模型仍停在 `0.700457`。
- 但 Top-3 等权集成达到 `0.709589`，明显高于 Exp-033 的原始加权 logits `0.703196`。
- 这说明 Exp-038 的几个 GAT 模型虽然单模型相近，但错误互补性强。
- 下一步应该优先对 Exp-038 Top-3 等权集成做 C&S，而不是继续训练。

是否保留：

- 保留。
- Top-3 等权集成是当前最值得继续后处理的 A1 候选。

---

## Tool-010：A1 Exp-039 C&S 候选搜索脚本

日期：2026-06-27

目标：

- 将 Exp-038 的有效集成候选自动送入 Correct and Smooth 搜索。
- 减少手工复制 checkpoint 路径的错误风险。

修改文件：

- `framework/scripts/run_exp039_a1_exp038_cs.sh`

脚本包含四组候选：

- `top2_equal`
  - Exp-038 Top-2 等权。
- `top3_equal`
  - Exp-038 Top-3 等权。
  - 原始验证准确率 `0.709589`，优先级最高。
- `greedy_weighted`
  - Exp-038 贪心加权：`64,16,20`。
- `top3_plus_old_gcn_95_5`
  - Exp-038 Top-3 GAT 占 `95%`
  - Exp-010 旧 GCN 占 `5%`
  - 用于验证跨模型族互补是否仍然有效。

验证：

```bash
cd /home/aliagent
bash -n framework/scripts/run_exp039_a1_exp038_cs.sh
```

验证结果：

- Shell 语法检查通过。

下一步：

在 GPU 服务器运行：

```bash
cd /home/aliagent
git pull --ff-only
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda ./scripts/run_exp039_a1_exp038_cs.sh
```

判断标准：

- 如果最佳 C&S `val_acc > 0.711416`，替代 Exp-033 成为 A1 最终候选。
- 如果没有超过 `0.711416`，继续沿用 Exp-033/Exp034 保底方案。

---

## Exp-039：A1 Exp-038 集成候选 C&S 结果

日期：2026-06-27

目标：

- 验证 Exp-038 中有效的 GAT 集成候选是否能通过 C&S 超过 Exp-033。

用户返回结果：

- `greedy_weighted.json`
  - 验证准确率：`0.713242`
  - Correct：`alpha=0.3, iter=5, weight=0.0`
  - Smooth：`alpha=0.75, iter=5, weight=0.75`
  - 伪标签：关闭
- `top3_plus_old_gcn_95_5.json`
  - 验证准确率：`0.709589`
- `top3_equal.json`
  - 验证准确率：`0.709589`
- `top2_equal.json`
  - 验证准确率：`0.709589`

对比：

- Exp-033 最优：`0.711416`
- Exp-039 greedy weighted + C&S：`0.713242`
- 本地提升：`+0.001826`

结论：

- Exp-039 greedy weighted + C&S 是当前 A1 本地最优。
- 旧 GCN 加入 Exp-038 Top-3 后没有提升，说明这轮最优互补来自 GAT 内部不同配置，而不是跨 GCN/GAT 模型族。
- 伪标签仍然没有贡献，不使用。

是否保留：

- 保留为最终 A1 候选。

---

## Tool-011：Exp-040 最后一次提交候选打包脚本

日期：2026-06-27

目标：

- 用当前最强 A1 和线上最强 A2 生成最后一次提交候选包。
- 降低手工打包、路径复制、CSV 格式错误的风险。

修改文件：

- `framework/scripts/build_exp040_final_candidate.sh`

提交包配置：

- A1：
  - 使用 Exp-039 greedy weighted + C&S。
  - checkpoint：
    - `output/exp038_a1_gat_grid/h256_heads4_d040_lr005_wd5e4_seed42/best_model.pt`
    - `output/exp038_a1_gat_grid/h224_heads4_d045_lr005_wd5e4_seed42/best_model.pt`
    - `output/exp038_a1_gat_grid/h256_heads8_d045_lr005_wd5e4_seed3407/best_model.pt`
  - checkpoint 权重：`64,16,20`
  - C&S：
    - `cs_normalize=random_walk`
    - `correct=(0.3,5,0.0)`
    - `smooth=(0.75,5,0.75)`
- A2：
  - 沿用 Exp-030 线上最佳。
  - `rec_strategy=history`
  - `seq_col=item_seq_raw`
  - `recent_n=10`
  - `cooccur_decay=0.96`
  - `history_filter=none`
  - `user_weight=0.02`
  - `user_combo_weight=0.18`
  - `user_combo_sizes=3,2,1`
  - `user_combo_mode=prefix`
  - `user_combo_min_count=5`

脚本产物：

- `output/exp040_submit_a1_exp039_greedy_a2_exp030/A1.csv`
- `output/exp040_submit_a1_exp039_greedy_a2_exp030/A2.csv`
- `output/exp040_submit_a1_exp039_greedy_a2_exp030/prediction.zip`

验证：

```bash
cd /home/aliagent
bash -n framework/scripts/build_exp040_final_candidate.sh
```

验证结果：

- Shell 语法检查通过。

下一步：

在 GPU 服务器运行：

```bash
cd /home/aliagent
git pull --ff-only
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda ./scripts/build_exp040_final_candidate.sh
```

运行完成后必须确认 `validate_submission.py` 输出“提交文件校验通过”，再考虑提交。

---

## Tool-012：A2 共现打分公式扩展

日期：2026-06-27

目标：

- 线上 Exp-040 证明 A1 固定验证集过拟合，继续追 A1 本地小幅提升风险很高。
- A2 当前线上稳定在 `0.4746`，与第一名 A2 `0.50967` 差距较大。
- 将 A2 历史 item 到 target 的共现打分从单一 `log_count` 扩展为多种 item-item 关联强度。

修改文件：

- `framework/code/rec_heuristics.py`
- `framework/code/a2_offline_eval.py`
- `framework/code/a2_grid_search.py`
- `framework/code/infer.py`
- `framework/tests/test_rec_heuristics.py`

修改内容：

- 新增 `build_cooccur_score_stats()`：
  - `log_count`：旧逻辑，`log1p(count)`
  - `count`
  - `confidence`
  - `jaccard`
  - `lift`
  - `sqrt_lift`
  - `pmi`
  - `log_pmi`
- `rank_items()` 新增 `cooccur_score_mode`：
  - `log_count`：保持旧逻辑。
  - `precomputed`：使用预计算的共现关联分数。
- `a2_offline_eval.py` 新增 `--cooccur_formula`
- `a2_grid_search.py` 新增 `--cooccur_formulas`
- `infer.py` 新增 `--cooccur_formula`
- 默认值均为 `log_count`，兼容 Exp-030 线上最佳旧配置。

测试：

```bash
cd /home/aliagent
python3 -m unittest framework.tests.test_rec_heuristics -v
python3 -m py_compile framework/code/rec_heuristics.py framework/code/a2_offline_eval.py framework/code/a2_grid_search.py framework/code/infer.py
```

测试结果：

- 单元测试：9 项通过。
- 语法检查：通过。

---

## Exp-041：A2 jaccard 共现公式实验

日期：2026-06-27

目标：

- 验证更像 item-item 关联强度的公式是否优于旧 `log_count`。
- 评估方式继续使用 `test_like_eval`，因为测试集短历史/空历史比例远高于训练集原始分布。

快速筛选命令：

```bash
python3 framework/code/a2_grid_search.py \
  --data_path framework/data/rec_data \
  --val_ratio 0.1 \
  --seed 42 \
  --topk 10 \
  --seq_cols item_seq_raw \
  --recent_ns 10,15 \
  --cooccur_decays 0.96,1.0 \
  --cooccur_formulas log_count,confidence,jaccard,lift,sqrt_lift,pmi,log_pmi \
  --strategies history \
  --history_filters none \
  --user_weights 0.02 \
  --user_combo_weights 0.18 \
  --user_combo_sizes 3,2,1 \
  --user_combo_mode prefix \
  --user_combo_min_count 5 \
  --item_feature_weights 0 \
  --pop_penalty_weights 0 \
  --history_count_weights 0 \
  --test_like_eval \
  --sort_metric weighted_ndcg \
  --top_results 30 \
  --output_json framework/output/exp041_a2_cooccur_formula_grid/quick_results.json
```

快速筛选结果：

- 最佳：`jaccard`
  - `recent_n=15`
  - `cooccur_decay=1.0`
  - `weighted_NDCG@10=0.485350`
- 当前 Exp-030 A2 离线候选：
  - `log_count`
  - `recent_n=10`
  - `cooccur_decay=0.96`
  - `weighted_NDCG@10=0.479879`

细化搜索结果：

- 最佳：`jaccard`
  - `seq_col=item_seq_raw`
  - `recent_n=18`
  - `cooccur_decay=1.0`
  - `user_weight=0.01`
  - `user_combo_weight=0.1`
  - `weighted_NDCG@10=0.488987`
  - `NDCG@10=0.486410`
  - `Hit@10=0.725500`
  - `MRR=0.411783`

结论：

- jaccard 明显优于旧 `log_count`。
- 该方向是当前最强 A2 候选，比 A1 继续本地小幅调参更值得提交验证。

---

## Exp-042：A1 Exp-030 + A2 jaccard 候选包

日期：2026-06-27

目标：

- A1 回退到线上已验证更稳的 Exp-030 A1。
- A2 使用 Exp-041 jaccard 最优参数。

候选包路径：

- `framework/output/exp042_submit_a1_exp030_a2_jaccard/prediction.zip`

A1：

- 直接复用 `framework/output/exp030_submit_a1cs_a2_decay_combo018/A1.csv`
- 该 A1 在线上得分为 `0.6874`，高于 Exp040 的 `0.6848`。

A2：

- `rec_strategy=history`
- `seq_col=item_seq_raw`
- `recent_n=18`
- `cooccur_formula=jaccard`
- `cooccur_decay=1.0`
- `history_filter=none`
- `user_weight=0.01`
- `user_combo_weight=0.1`
- `user_combo_sizes=3,2,1`
- `user_combo_mode=prefix`
- `user_combo_min_count=5`

生成与校验：

```bash
python3 framework/code/infer.py \
  --task task2 \
  --data_path framework/data/rec_data \
  --output_path framework/output/exp042_submit_a1_exp030_a2_jaccard/A2.csv \
  --rec_strategy history \
  --seq_col item_seq_raw \
  --recent_n 18 \
  --cooccur_formula jaccard \
  --cooccur_decay 1.0 \
  --history_filter none \
  --user_weight 0.01 \
  --user_combo_weight 0.1 \
  --user_combo_sizes 3,2,1 \
  --user_combo_mode prefix \
  --user_combo_min_count 5 \
  --history_count_weight 0 \
  --user_profile_cols auto \
  --topk 10 \
  --device cpu

python3 framework/code/validate_submission.py \
  --zip_path framework/output/exp042_submit_a1_exp030_a2_jaccard/prediction.zip \
  --cls_data_path framework/data/cls_data/A1.npz \
  --rec_data_dir framework/data/rec_data \
  --topk 10
```

校验结果：

- A1 通过：
  - 行数 `2751`
  - 表头正确
  - `test_idx` 顺序正确
- A2 通过：
  - 行数 `10000`
  - 表头正确
  - 每行 10 个合法 item
  - 无重复 item
- `prediction.zip` 校验通过。

与 Exp-030 A2 差异：

- `6847 / 10000` 个用户推荐串发生变化。
- 变化比例：`68.47%`

结论：

- Exp-042 是当前最强 A2 候选包。
- 若提交机会恢复，优先提交该包验证 A2 jaccard 是否能在线上转化为提升。

---

## Exp-043：A2 jaccard 多 seed 稳定性验证

日期：2026-06-27

目标：

- 验证 Exp-041 的 jaccard 提升是否只是 `seed=42` 验证切分偶然有效。
- 修正此前 A1 在固定 split 上过拟合的问题：只有多 seed 稳定提升才作为可信提交方向。

修改文件：

- `framework/scripts/run_exp043_a2_multiseed_jaccard.sh`

对比方案：

- 旧方案：
  - `cooccur_formula=log_count`
  - `recent_n=10`
  - `cooccur_decay=0.96`
  - `user_weight=0.02`
  - `user_combo_weight=0.18`
- 新方案：
  - `cooccur_formula=jaccard`
  - `recent_n=18`
  - `cooccur_decay=1.0`
  - `user_weight=0.01`
  - `user_combo_weight=0.1`

评估设置：

- `val_ratio=0.1`
- `test_like_eval=True`
- seeds：`42, 777, 2024, 2026, 3407`

结果：

| seed | old weighted_NDCG | new weighted_NDCG | gain |
| --- | ---: | ---: | ---: |
| 42 | 0.479879 | 0.488987 | +0.009109 |
| 777 | 0.478945 | 0.488137 | +0.009192 |
| 2024 | 0.477384 | 0.490289 | +0.012906 |
| 2026 | 0.469374 | 0.481861 | +0.012486 |
| 3407 | 0.470471 | 0.480102 | +0.009631 |

汇总：

- 平均提升：`+0.010665`
- 胜率：`5 / 5`
- 最小提升：`+0.009109`
- 最大提升：`+0.012906`

结论：

- jaccard 不是单 split 偶然提升，是当前最稳的 A2 改进。
- A2 下一步继续围绕 jaccard 做细化，而不是回到 `log_count`。
- 如果有线上提交机会，Exp-042 优先级高于 Exp040。

---

## Exp-044：A2 用户画像序列神经排序模型

日期：2026-06-27

目标：

- 回应“只靠规则不训练”的问题，新增一条真正需要 GPU 训练的 A2 路线。
- 让模型学习 `历史 item 序列 + 用户画像类别` 到 `target_iid` 的映射。
- 后续如果模型离线有效，再把模型分数与 jaccard 共现规则融合。

为什么要这样改：

- 测试用户和训练用户不重合，不能依赖用户 ID 记忆。
- 原 GRU4Rec/SASRec 主要只吃 item 序列，没有使用 `user.csv` 的 8 个用户画像列。
- 测试集有大量空历史和短历史用户，因此模型训练/验证必须模拟线上稀疏历史分布。
- A2 训练集中真实出现过的 target item 只有 235 个左右，直接预测全部 2156 个 item 会浪费类别容量；新模型先只预测训练 target 集合。

修改文件：

- 新增 `framework/code/a2_feature_ranker.py`

新增能力：

- `train` 模式：
  - 构建 item 映射、target 类别映射、用户画像类别映射。
  - 模型输入包括历史 item embedding 均值、最近 item embedding、用户画像 embedding、历史长度桶 embedding。
  - 支持 `--test_like_val`：验证集按测试历史长度分布裁剪。
  - 支持 `--random_test_like_train`：训练时随机裁剪历史，模拟线上短历史。
  - 输出 `best_model.pt` 和 `metrics.json`。
- `predict` 模式：
  - 从 checkpoint 恢复映射和模型。
  - 生成 10000 行 A2 推荐文件。

本地 smoke test：

```bash
python3 framework/code/a2_feature_ranker.py train \
  --data_path framework/data/rec_data \
  --output_dir framework/output/exp044_a2_feature_ranker_smoke \
  --device cpu \
  --epochs 1 \
  --batch_size 4096 \
  --max_len 40 \
  --embedding_dim 32 \
  --user_embedding_dim 4 \
  --hidden_dim 64 \
  --num_workers 0 \
  --test_like_val \
  --random_test_like_train
```

smoke test 结果：

- 训练样本：`36000`
- 验证样本：`4000`
- target 类别数：`235`
- 用户画像列：`u_cat_01` 到 `u_cat_08`
- 1 epoch CPU 检查：`NDCG@10=0.139753`
- 推理检查：成功生成 `10000` 行 A2.csv。

结论：

- 代码链路已通，但 1 epoch 小模型不代表真实效果。
- 下一步需要在 GPU 上跑完整配置：
  - 更大的 embedding / hidden_dim；
  - 50-100 epoch；
  - 多 seed；
  - 与 Exp-043 jaccard 规则做融合验证。
