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

GPU训练结果：

- 用户在 GPU 上运行完整配置后反馈：
  - best_epoch：`38`
  - train_loss：`2.805317717234294`
  - lr：`0.0005`
  - model-only NDCG@10：`0.4834490755702921`
  - Hit@10：`0.7265`
  - MRR：`0.40762589285714285`

分析：

- 模型单独分数已经接近 Exp-043 jaccard 规则，但没有超过规则基线。
- 因为模型和规则捕捉的信号不同，下一步必须评估 `模型分数 + jaccard规则` 的融合，而不是直接提交模型。
- 已修正 `a2_feature_ranker.py eval_fusion`：
  - `model_weight=0` 时可复现 Exp-043：`weighted_NDCG=0.488987`。
  - 融合评估使用拟合集构造规则统计，验证集只用于打分，避免验证泄漏。
  - 输出普通 NDCG 和按测试历史长度分布加权的 weighted_NDCG。

融合评估结果：

- 用户在 GPU 上使用完整模型 checkpoint 运行融合搜索后反馈：
  - 最佳 `model_weight=1.0`
  - NDCG@10：`0.49457351023846674`
  - weighted_NDCG：`0.49705253342802513`
  - Hit@10：`0.73625`
  - MRR：`0.4190993055555556`
- 对比 Exp-043 jaccard 规则：
  - `model_weight=0` weighted_NDCG：`0.488987`
  - 融合增益：`+0.008065`

分桶结果：

| bucket | samples | NDCG |
| --- | ---: | ---: |
| len=0 | 1432 | 0.371689 |
| len=1 | 409 | 0.504250 |
| len=2-3 | 1786 | 0.567481 |
| len=4-10 | 48 | 0.506793 |
| len>10 | 325 | 0.621385 |

结论：

- 模型融合是当前 A2 最强离线候选。
- 下一步生成 `model_weight=1.0` 的 A2 融合提交包，A1 先沿用 Exp-030 的线上稳定版本。
- 后续继续做多 seed 模型训练，验证 `model_weight=1.0` 是否稳定。

线上提交结果：

- 提交日期：2026-06-28
- 总分：`0.59484`
- A1 分类分数：`0.68739`
- A2 推荐分数：`0.50230`

线上结论：

- Exp-044 是当前最高线上总分。
- A2 从 Exp-030 的 `0.4746` 提升到 `0.50230`，线上提升约 `+0.02770`。
- A2 距离当前第一名 `0.50967` 只差约 `0.00737`，继续优化 A2 仍有价值。
- A1 仍停在 `0.68739`，距离第一名 `0.76845` 差约 `0.08106`，后续总分冲第一必须重新攻 A1。

---

## Tool-011：A2 feature ranker 多 seed 融合脚本

日期：2026-06-28

目标：

- 在 Exp-044 单 seed 模型融合线上有效的基础上，训练多个 seed 的 feature ranker。
- 对多个 checkpoint 的 logits 做平均，再与 jaccard 规则融合。
- 如果多 seed 离线优于单 seed，生成新的 `prediction.zip` 候选。

新增文件：

- `framework/scripts/run_exp045_a2_feature_multiseed.sh`

默认配置：

- seeds：`42,777,2024`
- max_len：`120`
- embedding_dim：`256`
- user_embedding_dim：`16`
- hidden_dim：`512`
- dropout：`0.25`
- epochs：`120`
- patience：`20`

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp045_a2_feature_multiseed.sh
```

预期输出：

- `output/exp045_a2_feature_multiseed/seed*/best_model.pt`
- `output/exp045_a2_feature_multiseed/fusion_eval.json`
- 若最佳 `model_weight > 0`：
  - `output/exp045_a2_feature_multiseed/A2.csv`
  - `output/exp045_a2_feature_multiseed/prediction.zip`

实验结果：

- 单 seed：
  - seed777：`epoch=54`, `ndcg=0.495881`, `hit=0.726500`, `mrr=0.423429`
  - seed2024：`epoch=28`, `ndcg=0.492018`, `hit=0.734250`, `mrr=0.416377`
  - seed42：`epoch=38`, `ndcg=0.483449`, `hit=0.726500`, `mrr=0.407626`
- 初始融合搜索：
  - baseline `model_weight=0`：`weighted_NDCG=0.4889872445345435`
  - best `model_weight=3.0`：`weighted_NDCG=0.5142448285756511`
  - 已生成候选包：`output/exp045_a2_feature_multiseed/prediction.zip`
- 扩大 `model_weight` 搜索后：
  - best `model_weight=35.0`
  - NDCG@10：`0.5138676200451141`
  - weighted_NDCG：`0.5161714951258527`
  - Hit@10：`0.76075`
  - MRR：`0.4367338293650794`
  - 已生成更强候选包：`output/exp045_a2_feature_multiseed_w35/prediction.zip`

与已有提交文件差异：

- `w35` vs `w3`：`7755 / 10000` 个用户推荐串变化。
- `w35` vs Exp-044：`9900 / 10000` 个用户推荐串变化。

结论：

- Exp-045 多 seed 模型融合明显强于 Exp-044 单 seed。
- `model_weight=35.0` 比初始 `model_weight=3.0` 继续提升 `+0.001927` weighted_NDCG。
- 如果还有提交机会，优先提交：
  - `framework/output/exp045_a2_feature_multiseed_w35/prediction.zip`

线上提交结果：

- 提交时间：2026-06-28 15:30:41
- 总分：`0.5916`
- A1 分类分数：`0.6874`
- A2 推荐分数：`0.4957`

线上结论：

- Exp-045 多 seed / 高权重融合没有转化为线上提升。
- 相比 Exp-044：
  - 总分从 `0.59484` 降到 `0.5916`
  - A2 从 `0.50230` 降到 `0.4957`
- 说明当前 test-like holdout 仍然高估了多 seed 模型 logits 的收益。
- Exp-045 系列暂时不作为默认提交方案；当前线上最佳仍为 Exp-044。
- A2 后续若继续做，必须用线上反馈校准验证策略，不能只看 `weighted_NDCG`。

---

## Tool-013：A2 提交差异分析工具

日期：2026-06-28

背景：

- Exp-045 离线显著强于 Exp-044，但线上 A2 从 `0.50230` 降到 `0.4957`。
- 说明仅凭当前 holdout `weighted_NDCG` 会高估某些模型候选。
- 后续 A2 候选需要先和线上最佳 Exp-044 做分布差异分析，避免过激改动。

新增文件：

- `framework/code/a2_compare_submissions.py`

功能：

- 比较两个 A2.csv：
  - 总体推荐串变化比例。
  - Top1 变化比例。
  - TopK 平均重合率。
  - 按测试历史长度桶分别统计。
  - Top1 分布漂移。

命令示例：

```bash
python3 framework/code/a2_compare_submissions.py \
  --base_a2 framework/output/exp044_a2_feature_fusion/A2.csv \
  --new_a2 framework/output/exp045_a2_feature_multiseed_w35/A2.csv \
  --test_csv framework/data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --topk 10
```

Exp045 w35 相对 Exp044 的差异：

- 总体推荐串变化：`99.0000%`
- Top1 变化：`18.3400%`
- Top10 overlap：`89.7820%`
- 长历史 `len>10`：
  - changed：`99.8944%`
  - top1_changed：`23.1257%`
  - overlap：`85.4171%`
- Top1 分布明显漂移：
  - `i000481` 从 `37.79%` 降到 `30.10%`
  - `i001069` 从 `23.74%` 升到 `27.47%`
  - `i000909` 从 `8.31%` 升到 `10.10%`

结论：

- Exp045 对线上最佳 Exp044 改动过大，尤其长历史用户和 Top1 分布漂移明显。
- 这与线上 A2 回退一致。
- 后续 A2 候选应控制相对 Exp044 的分布漂移，而不是只追离线 weighted_NDCG。

---

## Tool-012：A1 多 split 稳定性审计脚本

日期：2026-06-28

目标：

- Exp-039 在固定 split 本地最强，但线上 A1 不升反降。
- 后续 A1 不能继续只看 `split_seed=42`，必须评估多个 split 的均值、最小值和方差。

新增文件：

- `framework/scripts/run_exp046_a1_multisplit_audit.sh`

默认审计候选：

- Exp-019/030 稳定 A1：Exp-010 Top-5 + 固定 C&S。
- Exp-033：GAT 0.95 + GCN 0.05 + 固定 C&S。
- Exp-039：GAT 贪心加权 + 固定 C&S。

默认 split：

- `42,777,2024,2026,3407`

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp046_a1_multisplit_audit.sh
```

运行结果：

| candidate | n | mean_cs | min_cs | range | std | mean_model | mean_gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp019_top5_fixed_cs | 5 | 0.918174 | 0.700457 | 0.279452 | 0.108927 | 0.915982 | +0.002192 |
| exp039_greedy_fixed_cs | 5 | 0.914703 | 0.713242 | 0.258447 | 0.100791 | 0.913425 | +0.001279 |
| exp033_gat_gcn_95_5_fixed_cs | 5 | 0.901735 | 0.711416 | 0.247489 | 0.095317 | 0.900091 | +0.001644 |

复盘：

- 该审计结果不能直接作为 A1 泛化证据。
- 原因：这些 checkpoint 大多是按 `split_seed=42` 训练得到的。拿同一批 checkpoint 去评估其他 split 时，其他 split 的“验证节点”可能已经参与过原模型训练，导致 `0.94~0.98` 的虚高分数。
- 因此 Exp-046 只能说明“固定 checkpoint 在 split=42 上的相对排序”，不能说明线上泛化。

---

## Tool-014：A1 真多 split 重训练审计脚本

日期：2026-06-28

目标：

- 修正 Exp-046 的训练泄漏问题。
- 对每个 split_seed 重新训练模型，再在同一个 split 上评估 C&S。
- 先跑轻量代表模型，判断 A1 高分是否只来自固定 split 偶然性。

新增文件：

- `framework/scripts/run_exp047_a1_true_multisplit_train.sh`

默认配置：

- split seeds：`42,777,2024`
- GCN 代表：
  - `model_type=gcn`
  - `hidden_dim=384`
  - `seed=777`
  - `normalize=symmetric`
- GAT 代表：
  - `model_type=gat_sparse`
  - `hidden_dim=256`
  - `heads=4`
  - `seed=2026`
  - `normalize=none`

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp047_a1_true_multisplit_train.sh
```

判断标准：

- 如果真多 split 的均值仍接近或超过 `0.70`，说明 A1 还有继续训练/集成空间。
- 如果只有 split=42 高、其他 split 明显低，说明 A1 当前验证方式对线上不可靠，应谨慎提交 A1 替换。

运行结果：

| candidate | n | mean_cs | min_cs | max_cs | std | mean_train | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gat_h256_heads4_seed2026` | 3 | 0.698935 | 0.694977 | 0.705936 | 0.004965 | 0.695282 | 最稳定，适合作为 A1 下一阶段主模型 |
| `gcn_h384_seed777` | 3 | 0.693151 | 0.684018 | 0.706849 | 0.009864 | 0.679756 | 有补充价值，但稳定性弱于 GAT |

复盘：

- Exp-047 证明 Exp-046 的 `0.91+` 多 split 均值是训练泄漏造成的，不能作为泛化证据。
- 真实多 split 重训后，当前 A1 路线稳定水平约 `0.70`，比线上 `0.6874` 有继续提升空间，但距离第一名 A1 `0.76845` 仍有明显差距。
- 下一步不应继续只在 split=42 上做小幅 C&S 搜索，而应尝试“正式提交式全标签训练”：用多 split 决定结构和 epoch，再用全部 `train_idx` 标签重训最终模型。

---

## Tool-015：Exp-048 A1 全标签最终重训候选

日期：2026-06-28

目标：

- 解决 A1 提交模型仍只用约 90% 标签训练的问题。
- 在不改变 A2 的前提下，只尝试提升 A1，避免提交结果难以归因。

修改文件：

- `framework/code/train.py`
  - 新增 `--train_all_labels`：Task1 使用全部 `train_idx` 标签训练，不再保留验证集。
  - 新增 `--disable_early_stop`：完整训练到指定 `--epochs`，用于最终固定 epoch 重训。
  - 普通训练默认行为不变，仍然按验证集早停。
- `framework/scripts/run_exp048_a1_full_label_final.sh`
  - 训练 GAT 全标签模型：`gat_sparse h256 heads4 seed2026 epoch=270`。
  - 训练 GCN 全标签模型：`gcn h256 seed777 epoch=120`。
  - 使用 `GAT:GCN = 0.95:0.05` 做 logits 加权，再执行固定 C&S。
  - A2 沿用当前线上最佳 Exp-044。

原理：

- 线下调参必须保留验证集，否则无法判断模型是否泛化。
- 但正式提交时，验证集标签也是已知训练标签，继续丢弃会减少监督信号。
- 多 split 审计已经告诉我们 GAT 最优 epoch 大致在 `244-283`，因此正式重训取中位附近 `270`，用全部标签训练固定轮数。
- GCN 只给 `0.05` 权重，是为了补一点模型多样性，不让它主导结果。

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp048_a1_full_label_final.sh
```

当前状态：

- 已通过语法检查：
  - `python3 -m py_compile framework/code/train.py framework/code/a1_correct_smooth.py framework/code/validate_submission.py`
  - `bash -n framework/scripts/run_exp048_a1_full_label_final.sh`
- 等待 GPU 运行结果，不建议在结果未知时提交。

GPU返回结果：

- `a1_correct_smooth.py` 显示模型原始验证准确率 `0.979909`。
- C&S 后监控准确率为 `0.953425`。
- 生成候选包：`output/exp048_a1_full_label_final/prediction.zip`。
- A1 类别分布：`{0: 71, 1: 388, 2: 263, 3: 80, 4: 1198, 5: 42, 6: 73, 7: 136, 8: 464, 9: 36}`。

复盘：

- 这个 `0.979909` 不是有效验证分数。
- 原因：Exp-048 使用全部 `train_idx` 训练，随后 C&S 脚本再从 `train_idx` 切验证集，验证节点已经参与过训练，发生训练泄漏。
- 因此 Exp-048 当前只能证明“全标签模型能生成格式正确的提交包”，不能证明线上会显著提升。
- 在没有无泄漏审计之前，不建议提交 Exp-048。

---

## Tool-016：Exp-049 固定 epoch 无泄漏审计

日期：2026-06-28

目标：

- 用不泄漏的方式模拟 Exp-048 的训练方式。
- 判断“固定 epoch + 最终全标签训练”是否真的可能提升 A1，而不是只因为验证集泄漏看起来很高。

修改文件：

- `framework/code/train.py`
  - 新增 `--scheduler {plateau,none}`。
  - `--scheduler none` 用于固定 epoch 审计，避免学习率调度器偷看验证集。
- `framework/scripts/run_exp048_a1_full_label_final.sh`
  - 正式全标签训练也改为 `--scheduler none`，避免用训练内监控驱动学习率产生不稳定行为。
- `framework/scripts/run_exp049_a1_fixed_epoch_no_leak_audit.sh`
  - 每个 split 保留 10% 验证节点不参与训练。
  - 训练策略模拟 Exp-048：固定 epoch、关闭早停、关闭学习率调度。
  - 审计两个候选：
    - `gat_only`
    - `gat_gcn_95_5`

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp049_a1_fixed_epoch_no_leak_audit.sh
```

判断标准：

- 如果 Exp-049 的无泄漏均值明显高于 Exp-047 的 `gat_h256_heads4_seed2026 mean=0.698935`，说明 Exp-048 路线值得继续。
- 如果均值接近或低于 Exp-047，则 Exp-048 不应提交，应继续寻找新的 A1 结构或后处理。

运行结果：

| variant | n | mean | min | max | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gat_gcn_95_5` | 5 | 0.698082 | 0.693151 | 0.703196 | 0.003680 |
| `gat_only` | 5 | 0.695708 | 0.690411 | 0.702283 | 0.004060 |

结论：

- Exp-049 没有超过 Exp-047 的 GAT 真多 split 均值 `0.698935`。
- Exp-048 的全标签候选不具备无泄漏提升证据，不建议提交。
- A1 继续沿着“固定 epoch / 全标签重训”难以产生显著收益，应切换到新的模型族或图特征工程。

---

## Tool-017：Exp-050 A1 SIGN/MLP 多跳传播特征审计

日期：2026-06-28

目标：

- 寻找比当前 GAT/GCN 更有互补性的 A1 模型族。
- 当前训练边同质性约 `0.79`，说明图传播有价值；但两层 GCN/GAT 稳定水平卡在 `0.699` 左右。
- SIGN 把 `X, AX, A^2X, ... A^KX` 直接拼接，再用 MLP 分类，能显式利用多跳邻域，训练也比 GAT 更快。

新增文件：

- `framework/code/a1_sign_mlp.py`
  - 预计算多跳传播特征。
  - 支持有向/无向图、对称/随机游走归一化、特征归一化、每跳特征行归一化。
  - 训练 MLP 并执行轻量 C&S 搜索。
  - 可选生成 `A1.csv`。
- `framework/scripts/run_exp050_a1_sign_audit.sh`
  - 默认跑 5 个 split：`42,777,2024,2026,3407`。
  - 默认审计 5 组配置：
    - `sign_sym_undir_k3_h512_l2_do04_none`
    - `sign_sym_undir_k5_h512_l2_do04_none`
    - `sign_sym_undir_k5_h512_l2_do04_block`
    - `sign_rw_undir_k5_h512_l2_do04_block`
    - `sign_sym_undir_k5_h512_l2_do04_rowblock`

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp050_a1_sign_audit.sh
```

已完成检查：

- `python3 -m py_compile framework/code/a1_sign_mlp.py framework/code/train.py`
- `bash -n framework/scripts/run_exp050_a1_sign_audit.sh`
- CPU smoke test：
  - `hops=1`
  - `hidden_dim=32`
  - `epochs=1`
  - 主流程已走通。

判断标准：

- 如果某组 SIGN 的 `mean` 明显超过 `0.70`，下一步扩展为多 seed / 多配置 ensemble。
- 如果所有 SIGN 配置仍在 `0.69` 左右，说明单纯多跳传播特征也不够，应继续尝试 APPNP/GCNII 或更激进的结构特征。

运行结果：

| candidate | n | mean | min | max | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sign_rw_undir_k5_h512_l2_do04_block` | 5 | 0.725297 | 0.716895 | 0.738813 | 0.008053 |
| `sign_sym_undir_k5_h512_l2_do04_rowblock` | 5 | 0.719817 | 0.703196 | 0.746119 | 0.014433 |
| `sign_sym_undir_k5_h512_l2_do04_block` | 5 | 0.717260 | 0.705023 | 0.731507 | 0.008598 |
| `sign_sym_undir_k5_h512_l2_do04_none` | 5 | 0.692237 | 0.678539 | 0.705023 | 0.009751 |
| `sign_sym_undir_k3_h512_l2_do04_none` | 5 | 0.676164 | 0.666667 | 0.692237 | 0.008771 |

结论：

- Exp-050 是 A1 到目前为止最显著的无泄漏提升。
- 最强 SIGN 配置均值 `0.725297`，相比此前 GAT 稳定均值 `0.698935` 提升约 `+0.02636`。
- 这已经足够进入候选提交阶段。
- 最强配置使用：
  - 无向图；
  - 随机游走归一化；
  - 5跳传播；
  - 每跳特征行归一化；
  - MLP hidden=512, layers=2, dropout=0.4。

---

## Tool-018：Exp-051 A1 SIGN 五折集成提交候选

日期：2026-06-28

目标：

- 基于 Exp-050 最强 SIGN 配置生成正式候选包。
- A2 沿用当前线上最佳 Exp-044，只替换 A1，便于归因。

新增文件：

- `framework/code/a1_sign_infer.py`
  - 读取多个 SIGN checkpoint。
  - 按 checkpoint 内保存的参数重建传播特征和 MLP。
  - 平均多个模型的全节点预测概率。
  - 使用全部训练标签执行 C&S。
  - 输出 `A1.csv`。
- `framework/scripts/build_exp051_a1_sign_ensemble_candidate.sh`
  - 使用 `sign_rw_undir_k5_h512_l2_do04_block` 的 5 个 split checkpoint。
  - 等权平均五个模型概率。
  - C&S 参数：`correct=(0.3,5,0.0)`, `smooth=(0.7,5,0.75)`。
  - A2 复制 `output/exp044_a2_feature_fusion/A2.csv`。
  - 打包并校验 `prediction.zip`。

本地运行命令：

```bash
cd /home/aliagent/framework
DEVICE=cpu ./scripts/build_exp051_a1_sign_ensemble_candidate.sh
```

运行结果：

- 候选包：`framework/output/exp051_submit_a1_sign_ensemble_a2_exp044/prediction.zip`
- A1 类别分布：
  - `{0: 55, 1: 336, 2: 258, 3: 61, 4: 1288, 5: 52, 6: 46, 7: 137, 8: 481, 9: 37}`
- A2 沿用 Exp-044。
- 提交格式校验通过。

提交判断：

- 这是目前最值得提交的候选。
- 预期主要提升 A1；A2 应接近 Exp-044 的线上 `0.50230`。
- 风险：SIGN 五折集成的本地评估不能直接验证，因为多个 split checkpoint 对任意固定验证集都有训练重叠；但其单模型多 split 无泄漏均值已经显著强于旧 A1 路线。

线上结果：

- 提交时间：`2026-06-28 16:56:12`
- 总分：`0.6070`
- A1 分类分数：`0.7117`
- A2 推荐分数：`0.5023`

复盘：

- Exp-051 是有效提交，总分从 Exp-044 的 `0.59484` 提升到 `0.6070`。
- A1 从 `0.68739` 提升到 `0.7117`，确认 SIGN 路线在线上有效。
- A2 与 Exp-044 一致，说明推荐任务没有退化。
- 但 A1 线上 `0.7117` 仍低于 Exp-050 最强配置无泄漏均值 `0.725297`，可能存在 split 集成分布偏移或 C&S 固定参数不完全匹配。
- 下一步优先做异构融合：SIGN-rw + SIGN-row/SIGN-sym + GAT/GCN，而不是只微调单个 SIGN 参数。

---

## Tool-019：Exp-052 A1 异构模型融合无泄漏审计

日期：2026-06-28

目标：

- 在不重新训练的情况下，复用已有 checkpoint，检查异构模型融合能否超过单 SIGN。
- 每个 split 只使用同 split 下训练的模型，在该 split 验证集上评估，避免训练泄漏。

新增文件：

- `framework/code/a1_mixed_ensemble_audit.py`
  - 支持加载 SIGN checkpoint 和 GNN checkpoint。
  - 对每个 split 计算模型概率。
  - 搜索少量有解释性的融合权重组合。
  - 对融合概率执行 C&S 并汇总 mean/min/std。
- `framework/scripts/run_exp052_a1_mixed_ensemble_audit.sh`
  - 默认 split：`42,777,2024,2026,3407`
  - 默认融合模型：
    - `sign_rw`
    - `sign_sym`
    - `sign_row`
    - `gat`
    - `gcn`

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp052_a1_mixed_ensemble_audit.sh
```

已完成检查：

- `python3 -m py_compile framework/code/a1_mixed_ensemble_audit.py framework/code/a1_sign_infer.py framework/code/a1_sign_mlp.py`
- `bash -n framework/scripts/run_exp052_a1_mixed_ensemble_audit.sh`
- CPU smoke test：`SPLIT_SEEDS=42` 已成功输出汇总。

Smoke 观察：

- 单 split 上 `sign_rw:0.85 + gat:0.15` 达到 `0.725114`，高于该 split 单 SIGN 的 `0.718721`。
- 这只是单 split 现象，必须等完整 5 split 均值后才能决定是否提交。

运行结果：

| candidate | n | mean | min | std | smooth |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sign_rw:0.85+gat:0.15` | 5 | 0.724749 | 0.718721 | 0.003770 | 0.75 |
| `sign_rw:0.85+gat:0.15` | 5 | 0.724749 | 0.718721 | 0.005498 | 0.5 |
| `sign_rw:0.85+gcn:0.15` | 5 | 0.724018 | 0.714155 | 0.008899 | 0.75 |
| `sign_rw:1` | 5 | 0.723653 | 0.712329 | 0.009484 | 0.75 |

结论：

- 异构融合没有超过 Exp-050 单 SIGN 最强均值 `0.725297`。
- 不生成 Exp052 提交包。
- 下一步需要改输入/模型结构，而不是继续调 SIGN/GAT/GNN 融合权重。

---

## Tool-020：Exp-053 A1 SIGN + 标签传播特征审计

日期：2026-06-28

目标：

- 做比 ensemble 权重更大的模型输入升级。
- 将训练标签沿图传播后的分布作为 MLP 输入特征，与 SIGN 属性传播特征拼接。
- 验证标签信息是否应该进入模型训练阶段，而不是只在最后 C&S 阶段使用。

修改文件：

- `framework/code/a1_sign_mlp.py`
  - 新增 `--label_feature_hops`。
  - 新增 `--label_feature_norm`。
  - 新增 `--label_feature_include_seed`。
  - 新增 `--label_feature_row_norm`。
  - 新增 `--label_feature_weight`。
  - 训练/验证时只用当前 split 的训练节点标签构造标签传播特征，避免验证泄漏。
  - 正式推理时使用全部 `train_idx` 标签构造标签传播特征。
- `framework/code/a1_sign_infer.py`
  - 更新为通过 `build_model_features()` 构造输入。
  - 兼容旧 SIGN checkpoint；旧模型默认 `label_feature_hops=0`，输出不变。
- `framework/scripts/run_exp053_a1_sign_label_feature_audit.sh`
  - 默认 6 组标签特征配置 x 5 个 split。

默认审计配置：

- 基础属性特征：
  - `hops=5`
  - `prop_norm=random_walk`
  - `graph_mode=undirected`
  - `block_norm=True`
- 标签特征候选：
  - `labelrw_h2_w1`
  - `labelrw_h3_w1`
  - `labelrw_h5_w1`
  - `labelrw_h3_rownorm_w1`
  - `labelsym_h3_w1`
  - `labelrw_h3_w05`

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp053_a1_sign_label_feature_audit.sh
```

已完成检查：

- `python3 -m py_compile framework/code/a1_sign_mlp.py framework/code/a1_sign_infer.py`
- `bash -n framework/scripts/run_exp053_a1_sign_label_feature_audit.sh`
- CPU smoke test：
  - `label_feature_hops=2`
  - `epochs=1`
  - 主流程已走通。
- Exp051 旧 checkpoint 兼容性验证：
  - 重新运行 `build_exp051_a1_sign_ensemble_candidate.sh`。
  - A1 类别分布保持 `{0: 55, 1: 336, 2: 258, 3: 61, 4: 1288, 5: 52, 6: 46, 7: 137, 8: 481, 9: 37}`。

判断标准：

- 如果 Exp-053 均值明显超过 `0.725297`，进入下一轮提交候选。
- 如果低于 Exp-050，说明标签传播更适合作为 C&S 后处理，不适合直接拼进 MLP 输入。

运行结果：

| candidate | n | mean | min | max | std | 最佳后处理 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sign_rw_k5_block_labelrw_h3_rownorm_w1` | 5 | 0.757078 | 0.742466 | 0.779909 | 0.012992 | model_only |
| `sign_rw_k5_block_labelrw_h3_w1` | 5 | 0.753242 | 0.731507 | 0.776256 | 0.015535 | mostly model_only |
| `sign_rw_k5_block_labelrw_h2_w1` | 5 | 0.751233 | 0.736073 | 0.773516 | 0.013893 | mostly model_only |
| `sign_rw_k5_block_labelrw_h5_w1` | 5 | 0.749954 | 0.737900 | 0.767123 | 0.010387 | model_only |
| `sign_rw_k5_block_labelrw_h3_w05` | 5 | 0.749772 | 0.737900 | 0.768950 | 0.010790 | mostly model_only |
| `sign_rw_k5_block_labelsym_h3_w1` | 5 | 0.746301 | 0.729680 | 0.772603 | 0.015297 | mostly model_only |

结论：

- Exp-053 是继 Exp-050 后的第二次大幅提升。
- 最强配置均值 `0.757078`，相比 Exp-050 `0.725297` 再提升约 `+0.03178`。
- 最佳后处理是 `model_only`，说明标签传播信息已经被模型输入吸收，不应再额外强平滑。
- 该结果已经接近当前第一名 A1 分数 `0.76845` 的区间，值得立即生成提交候选。

---

## Tool-021：Exp-054 A1 SIGN标签传播特征五折集成提交候选

日期：2026-06-28

目标：

- 基于 Exp-053 最强配置生成正式候选包。
- A2 沿用当前线上最佳 Exp-044。
- 最终推理不额外做 C&S 平滑，使用 `smooth_weight=0`。

新增文件：

- `framework/scripts/build_exp054_a1_sign_label_ensemble_candidate.sh`
  - 使用 `sign_rw_k5_block_labelrw_h3_rownorm_w1` 的 5 个 split checkpoint。
  - 等权平均五个模型概率。
  - 使用全部 `train_idx` 构造标签传播特征。
  - C&S 参数中 `smooth_weight=0.0`，等价于 model-only 输出。
  - A2 复制 `output/exp044_a2_feature_fusion/A2.csv`。
  - 打包并校验 `prediction.zip`。

本地运行命令：

```bash
cd /home/aliagent/framework
DEVICE=cpu ./scripts/build_exp054_a1_sign_label_ensemble_candidate.sh
```

运行结果：

- 候选包：`framework/output/exp054_submit_a1_sign_label_ensemble_a2_exp044/prediction.zip`
- A1 类别分布：
  - `{0: 64, 1: 360, 2: 257, 3: 72, 4: 1159, 5: 46, 6: 71, 7: 153, 8: 532, 9: 37}`
- A2 沿用 Exp-044。
- 提交格式校验通过。

提交判断：

- 这是当前最值得提交的候选。
- 预期主要提升 A1；A2 应保持在 Exp-044 附近。

线上结果：

- 提交时间：`2026-06-28 19:24:04`
- 总分：`0.6241`
- A1 分类分数：`0.7459`
- A2 推荐分数：`0.5023`

复盘：

- Exp-054 继续显著提升，总分从 Exp-051 的 `0.6070` 提升到 `0.6241`。
- A1 从 `0.7117` 提升到 `0.7459`，证明标签传播特征在线上有效。
- 与当前第一名 A1 `0.76845` 仍有约 `0.02255` 差距。
- Exp-054 使用的是 split checkpoint，每个模型训练时只见过约 90% 标签；正式推理虽然使用了全部标签构造标签传播特征，但模型权重本身仍不是全标签训练得到的。
- 下一步尝试“全标签最终训练”：用全部 `train_idx` 训练 SIGN 标签特征模型，并用多个 seed/epoch 集成。

---

## Tool-022：Exp-055 A1 SIGN标签传播特征全标签最终训练候选

日期：2026-06-29

目标：

- 用全部训练标签重训 Exp-053 最强模型结构。
- 进一步缩小 A1 与第一名的差距。

修改文件：

- `framework/code/a1_sign_mlp.py`
  - 新增 `--train_all_labels`。
  - 新增 `--disable_early_stop`。
  - 全标签模式下，全部 `train_idx` 参与损失和标签传播特征构造，验证指标仅作训练监控。
- `framework/scripts/run_exp055_a1_sign_label_fulltrain_candidate.sh`
  - 使用 Exp-053 最强结构：
    - `hops=5`
    - `prop_norm=random_walk`
    - `graph_mode=undirected`
    - `block_norm=True`
    - `label_feature_hops=3`
    - `label_feature_norm=random_walk`
    - `label_feature_row_norm=True`
  - 训练 5 个全标签模型：
    - `seed42_e34`
    - `seed777_e34`
    - `seed3407_e36`
    - `seed2026_e40`
    - `seed2024_e45`
  - epoch 选择来自 Exp-053 最强配置各 split 的最优 epoch：`34,34,36,40,45`。
  - 五模型等权集成。
  - A2 沿用 Exp-044。

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp055_a1_sign_label_fulltrain_candidate.sh
```

已完成检查：

- `python3 -m py_compile framework/code/a1_sign_mlp.py framework/code/a1_sign_infer.py framework/code/validate_submission.py`
- `bash -n framework/scripts/run_exp055_a1_sign_label_fulltrain_candidate.sh`
- CPU smoke test：
  - `--train_all_labels`
  - `--disable_early_stop`
  - `epochs=1`
  - 主流程已走通。

提交判断：

- Exp-055 生成后需要先查看 A1 类别分布。
- 如果分布没有明显异常，可以作为下一次候选提交。

GPU运行结果：

- 候选包：`output/exp055_a1_sign_label_fulltrain_candidate/prediction.zip`
- A1 类别分布：
  - `{0: 67, 1: 364, 2: 271, 3: 73, 4: 1197, 5: 47, 6: 72, 7: 159, 8: 466, 9: 35}`
- A2 沿用 Exp-044。
- 提交格式校验通过。

提交判断：

- 类别分布正常，没有出现单类塌缩。
- 相比 Exp-054：
  - 类别 4 从 `1159` 回升到 `1197`；
  - 类别 8 从 `532` 回落到 `466`；
  - 其余类别变化较小。
- 这是合理漂移，符合“全标签训练后模型边界发生变化”的预期。
- 可以作为下一次候选提交。

线上结果：

- 提交时间：`2026-06-29 15:10:04`
- 总分：`0.6245`
- A1 分类分数：`0.7466`
- A2 推荐分数：`0.5023`

复盘：

- Exp-055 相比 Exp-054 只小幅提升：
  - 总分 `0.6241 -> 0.6245`
  - A1 `0.7459 -> 0.7466`
- 说明全标签固定 epoch 重训收益有限。
- 继续调 epoch/seed 可能只有小幅收益，不符合“先做大提升”的要求。
- 下一步切换到自训练伪标签：利用强模型对未标注节点的高置信预测，构造第二阶段标签传播特征。

---

## Tool-023：Exp-056 A1 SIGN标签特征伪标签自训练审计

日期：2026-06-29

目标：

- 在 Exp-053/055 基础上做更大方向升级。
- 第一阶段模型只用当前 split 训练标签。
- 用第一阶段模型为未标注节点生成高置信伪标签。
- 第二阶段用“真实标签 + 伪标签”构造标签传播特征，再重新训练 SIGN/MLP。
- 每个 split 的验证标签不参与伪标签选择和特征构造，避免标签泄漏。

判断标准：

- 如果 Exp-056 的 5 split 均值明显超过 Exp-053 的 `0.757078`，再构造正式提交包。
- 如果没有超过，说明当前强模型已经接近伪标签可带来的收益上限，应转向 A2 或更强 A1 结构。

新增文件：

- `framework/code/a1_sign_pseudo_audit.py`
  - 加载 Exp-053 第一阶段 checkpoint。
  - 生成高置信伪标签。
  - 用真实标签 + 伪标签构造第二阶段标签传播特征。
  - 第二阶段只在真实训练标签上计算 loss。
  - 在验证集上汇总无泄漏表现。
- `framework/scripts/run_exp056_a1_sign_pseudo_audit.sh`
  - 默认阈值：`0.6,0.7,0.75`。
  - 默认伪标签权重：`0.5,1.0`。
  - 默认 split：`42,777,2024,2026,3407`。

阈值调整原因：

- 对 split42 的第一阶段模型检查发现，未标注节点最高置信度约 `0.828`。
- `0.85/0.9/0.95` 基本不会选出伪标签。
- 因此默认改为 `0.6,0.7,0.75`。

已完成检查：

- `python3 -m py_compile framework/code/a1_sign_pseudo_audit.py framework/code/a1_sign_mlp.py framework/code/a1_sign_infer.py`
- `bash -n framework/scripts/run_exp056_a1_sign_pseudo_audit.sh`
- CPU smoke test：
  - `threshold=0.7`
  - `pseudo_weight=0.5`
  - `epochs=2`
  - split42 选出 `272` 个伪标签，主流程已走通。

运行结果：

| threshold | pseudo_weight | avg_pseudo | mean | min | std | kind |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.6 | 1.0 | 1496.8 | 0.757260 | 0.740639 | 0.013673 | model_only |
| 0.75 | 1.0 | 671.0 | 0.755434 | 0.744292 | 0.011644 | model_only |
| 0.7 | 0.5 | 911.2 | 0.753425 | 0.741553 | 0.010728 | model_only |
| 0.6 | 0.5 | 1496.8 | 0.752694 | 0.737900 | 0.015308 | model_only |
| 0.7 | 1.0 | 911.2 | 0.752146 | 0.736986 | 0.011661 | model_only |
| 0.75 | 0.5 | 671.0 | 0.750868 | 0.737900 | 0.011786 | model_only |

结论：

- 伪标签自训练最高 `0.757260`，只比 Exp-053 的 `0.757078` 高 `+0.000182`。
- 这不是显著提升，不生成提交包。
- 继续自训练大概率只是小幅波动，应切换到官方提到的结构特征工程。

---

## Tool-024：Exp-057 A1 SIGN标签特征 + 图结构特征审计

日期：2026-06-29

目标：

- 根据官方提分材料中的“节点度特征拼接、低度节点额外标记”做结构特征工程。
- 在 Exp-053 最强 SIGN-label 配置基础上追加结构特征。

修改文件：

- `framework/code/a1_sign_mlp.py`
  - 新增 `--structure_feature_mode {none,basic,label}`。
  - 新增 `--structure_feature_weight`。
  - `basic` 特征：
    - 入度、出度、无向度；
    - `log1p` 度数；
    - 低度/孤立标记。
  - `label` 特征：
    - 在 `basic` 基础上加入当前训练标签邻居类别分布；
    - 每个 split 只用训练折标签统计，不使用验证标签。
- `framework/scripts/run_exp057_a1_sign_structure_feature_audit.sh`
  - 默认审计 4 组配置：
    - `basic_w1`
    - `basic_w05`
    - `label_w1`
    - `label_w05`

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp057_a1_sign_structure_feature_audit.sh
```

已完成检查：

- `python3 -m py_compile framework/code/a1_sign_mlp.py framework/code/a1_sign_infer.py`
- `bash -n framework/scripts/run_exp057_a1_sign_structure_feature_audit.sh`
- CPU smoke test：
  - `structure_feature_mode=label`
  - `epochs=1`
  - 主流程已走通。
- 旧 Exp054 checkpoint 兼容性验证：
  - 重新运行 `build_exp054_a1_sign_label_ensemble_candidate.sh`。
  - A1 类别分布保持不变。

判断标准：

- 如果 Exp-057 均值明显超过 `0.757078`，再生成结构特征提交包。
- 如果没有超过，说明当前结构特征没有提供额外泛化收益，应转向 A2 或更强 A1 架构。

运行结果：

| candidate | mean | min | max | std | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| sign_label_struct_basic_w1 | 0.759087 | 0.746119 | 0.779909 | 0.011829 | 最优 |
| sign_label_struct_label_w1 | 0.757991 | 0.741553 | 0.775342 | 0.011508 | 次优 |
| sign_label_struct_label_w05 | 0.756712 | 0.739726 | 0.778082 | 0.013663 | 不保留 |
| sign_label_struct_basic_w05 | 0.754155 | 0.742466 | 0.775342 | 0.012699 | 不保留 |

结论：

- `basic_w1` 相比 Exp-053 最优均值 `0.757078` 提升到 `0.759087`，提升 `+0.002009`。
- 提升方向有效，说明官方建议的“节点度特征、低度节点标记”确实能补充图结构信号。
- 但提升仍不够大，预计线上 A1 只能从 `0.7466` 提到约 `0.748` 左右，暂不单独提交。
- 下一步继续沿官方“PCA降维、标准化”方向做更大特征工程尝试。

---

## Tool-025：Exp-058 A1 SIGN标签特征 + 结构特征 + SVD降噪特征审计

日期：2026-06-29

目标：

- 在 Exp-057 最强 `sign_label_struct_basic_w1` 基础上加入低秩降噪特征。
- 对应官方资料中的“PCA降维：767维 -> 128维减少噪声”和“特征归一化”。
- 当前实现使用 `TruncatedSVD` 替代传统 PCA，因为原始特征是稀疏矩阵，SVD 可以直接处理稀疏输入。

修改文件：

- `framework/code/a1_sign_mlp.py`
  - 新增 `--feature_transform`：
    - `none`：保持原始逻辑。
    - `standard`：对原始 767 维特征按列标准化。
    - `svd`：只使用 SVD 降维特征。
    - `raw_plus_svd`：保留原始特征，并追加 SVD 降维特征。
    - `raw_plus_standard_svd`：标准化原始特征，并追加 SVD 降维特征。
  - 新增 `--svd_dim`、`--svd_weight`、`--svd_seed`。
  - 默认值保持 `feature_transform=none`，兼容旧 checkpoint。
- `framework/scripts/run_exp058_a1_sign_svd_feature_audit.sh`
  - 默认在 5 个 split 上审计 6 组特征变换：
    - `standard`
    - `svd128`
    - `raw_plus_svd64_w05`
    - `raw_plus_svd128_w05`
    - `raw_plus_svd128_w1`
    - `raw_plus_svd256_w05`

原理说明：

- 原始节点特征有 767 维，里面可能既有类别相关信号，也有噪声。
- SIGN 会计算 `X, AX, A^2X, ..., A^5X`，如果原始特征里有噪声，传播会把噪声也扩散到邻居。
- SVD/PCA 的作用是提取解释方差较大的低秩方向，相当于先把特征压缩成更稳定的“主信号”，再做图传播。
- `raw_plus_svd` 不是直接丢掉原始特征，而是让模型同时看到原始特征和降噪特征，避免 SVD 丢失少数类别的细粒度信息。

已完成检查：

- `python3 -m py_compile framework/code/a1_sign_mlp.py framework/code/a1_sign_infer.py`
- `bash -n framework/scripts/run_exp058_a1_sign_svd_feature_audit.sh`
- CPU smoke test：
  - `feature_transform=svd`
  - `svd_dim=32`
  - `epochs=1`
  - 主流程已走通。

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp058_a1_sign_svd_feature_audit.sh
```

判断标准：

- 若 5 split 均值达到 `0.765+`，认为是比较显著的 A1 提升，优先生成正式提交包。
- 若只在 `0.759~0.761` 附近，说明降维特征只是小修小补，应继续切换到更强 A1 模型或 A2 冷启动方案。

运行结果：

| candidate | mean | min | max | std | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| sign_label_struct_basic_standard | 0.759635 | 0.749772 | 0.778082 | 0.009706 | 最优 |
| sign_label_struct_basic_raw_svd128_w1 | 0.757260 | 0.747945 | 0.775342 | 0.009585 | 不保留 |
| sign_label_struct_basic_raw_svd128_w05 | 0.756895 | 0.744292 | 0.777169 | 0.011500 | 不保留 |
| sign_label_struct_basic_raw_svd64_w05 | 0.755068 | 0.742466 | 0.778995 | 0.013151 | 不保留 |
| sign_label_struct_basic_raw_svd256_w05 | 0.754521 | 0.740639 | 0.773516 | 0.010641 | 不保留 |
| sign_label_struct_basic_svd128_w1 | 0.749954 | 0.741553 | 0.765297 | 0.008633 | 不保留 |

结论：

- `standard` 比 Exp-057 最优 `0.759087` 小幅提升到 `0.759635`，提升 `+0.000548`。
- SVD 降维没有带来更大收益，甚至纯 SVD 明显下降，说明当前类别信号可能分布在原始稀疏特征的细粒度维度中，低秩压缩会丢信息。
- 暂不提交，继续沿真正带来大幅提升的“标签传播特征”方向扩展。

---

## Tool-026：Exp-059 A1 SIGN多方向标签传播特征审计

日期：2026-06-29

目标：

- 在 Exp-058 最优 `standard + structure basic + labelrw_h3` 基础上，扩展标签传播特征的图方向。
- 检查有向边是否包含额外分类信息。

修改文件：

- `framework/code/a1_sign_mlp.py`
  - 新增 `--label_feature_graph_modes`，支持逗号分隔：
    - `undirected`：无向邻居标签分布；
    - `directed`：原始边方向传播；
    - `reverse`：反向边传播。
  - 新增 `--label_feature_norms`，支持同时拼接 `random_walk` 和 `symmetric` 两种归一化标签传播特征。
  - `_make_adj()` 支持按调用方传入图方向，兼容旧 checkpoint。
- `framework/scripts/run_exp059_a1_sign_directed_label_feature_audit.sh`
  - 默认固定 Exp-058 最优特征侧配置：
    - `feature_transform=standard`
    - `structure_feature_mode=basic`
    - `hops=5`
    - `label_feature_hops=2/3/4`
  - 审计方向组合：
    - 单方向：`undirected`、`directed`、`reverse`
    - 双方向：`undirected,directed`、`undirected,reverse`、`directed,reverse`
    - 三方向：`undirected,directed,reverse`
    - 三方向 + 双归一化：`random_walk,symmetric`

原理说明：

- 之前的标签传播默认把图无向化，等价于只问“邻居整体上是什么类别”。
- 如果边方向有业务意义，那么“我指向谁”和“谁指向我”可能代表不同关系。
- 多方向标签传播让模型同时看到三类信号：
  - 无向邻居类别共性；
  - 出边邻居类别；
  - 入边邻居类别。
- 如果 A1 图确实是有向金融产品关系，这一轮可能比 SVD、度数特征带来更大的提升。

已完成检查：

- `python3 -m py_compile framework/code/a1_sign_mlp.py framework/code/a1_sign_infer.py`
- `bash -n framework/scripts/run_exp059_a1_sign_directed_label_feature_audit.sh`
- CPU smoke test：
  - `label_feature_graph_modes=undirected,directed,reverse`
  - `label_feature_norms=random_walk,symmetric`
  - `epochs=1`
  - 主流程已走通。

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp059_a1_sign_directed_label_feature_audit.sh
```

判断标准：

- 若均值达到 `0.765+`，生成 A1 正式候选包。
- 若均值仍停在 `0.760` 左右，说明边方向贡献有限，下一步转向更强的 A2 冷启动/负采样方案或 A1 多模型蒸馏。

运行结果：

| candidate | mean | min | max | std | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| standard_struct_label_undir_reverse_h3_rw | 0.763653 | 0.747032 | 0.789954 | 0.015187 | 最优但波动较大 |
| standard_struct_label_undir_h3_rw | 0.759817 | 0.750685 | 0.778995 | 0.009920 | 更稳定 |
| standard_struct_label_all_h2_rw | 0.759087 | 0.748858 | 0.780822 | 0.011339 | 可作为融合候选 |
| standard_struct_label_directed_reverse_h3_rw | 0.755616 | 0.744292 | 0.777169 | 0.011985 | 不保留 |
| standard_struct_label_undir_directed_h3_rw | 0.754703 | 0.746119 | 0.775342 | 0.010907 | 不保留 |
| standard_struct_label_all_h4_rw | 0.753973 | 0.743379 | 0.773516 | 0.011089 | 不保留 |
| standard_struct_label_all_h3_rw | 0.753973 | 0.736073 | 0.778082 | 0.013833 | 不保留 |
| standard_struct_label_all_h3_rw_sym | 0.739726 | 0.732420 | 0.753425 | 0.007684 | 不保留 |
| standard_struct_label_reverse_h3_rw | 0.726393 | 0.717808 | 0.743379 | 0.008978 | 不保留 |
| standard_struct_label_directed_h3_rw | 0.724201 | 0.710502 | 0.738813 | 0.009277 | 不保留 |

关键观察：

- `undirected+reverse` 比 Exp-058 最优 `0.759635` 提升到 `0.763653`，提升 `+0.004018`。
- 单独 `reverse` 很差，但 `undirected+reverse` 最好，说明反向边标签不能单独用，但能补充无向邻居标签。
- `undirected+reverse` 在 `split42/777/3407` 明显赢，在 `split2024/2026` 输给纯 `undirected`。
- 因此下一步不急着提交单一 `undirected+reverse`，先做配置级概率融合，争取获得更稳的候选。

---

## Tool-027：Exp-060 A1 SIGN配置概率融合审计

日期：2026-06-29

目标：

- 不重新训练，直接融合 Exp-059 已有 checkpoint 的输出概率。
- 解决 `undirected+reverse` 上限高但分折波动大的问题。
- 检查 `undirected`、`undirected+reverse`、`all_h2` 三类配置是否互补。

新增文件：

- `framework/code/a1_sign_config_ensemble_audit.py`
  - 支持 `name=path` 形式传入多个 SIGN 配置目录。
  - 每个 split 只加载对应 split 的 checkpoint。
  - 用 0.1 粒度 simplex 权重网格融合概率。
  - 输出跨 split 的 mean/min/std 汇总。
- `framework/scripts/run_exp060_a1_sign_config_ensemble_audit.sh`
  - 默认融合：
    - `undir=standard_struct_label_undir_h3_rw`
    - `undir_reverse=standard_struct_label_undir_reverse_h3_rw`
    - `all_h2=standard_struct_label_all_h2_rw`
- `framework/scripts/build_exp059_a1_undir_reverse_candidate.sh`
  - 备用：直接用 Exp-059 当前最强 `undirected+reverse` 生成候选提交包。
  - 是否提交取决于 Exp-060 是否能找到更稳组合。

已完成检查：

- `python3 -m py_compile framework/code/a1_sign_config_ensemble_audit.py framework/code/a1_sign_mlp.py framework/code/a1_sign_infer.py`
- `bash -n framework/scripts/run_exp060_a1_sign_config_ensemble_audit.sh`
- `bash -n framework/scripts/build_exp059_a1_undir_reverse_candidate.sh`

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp060_a1_sign_config_ensemble_audit.sh
```

判断标准：

- 如果融合均值超过 `0.765` 且 min 不低于 `0.750`，优先生成融合提交包。
- 如果融合没有超过 `0.763653`，则用 Exp-059 当前最佳候选作为 A1 提交候选，或者转向 A2 做下一轮大提升。

第一次运行结果异常：

```text
mean=0.955982  min=0.938813  undir_reverse:0.2+all_h2:0.8
mean=0.955799  min=0.938813  undir_reverse:0.3+all_h2:0.7
...
```

判定：

- 该结果作废，不能作为提交依据。
- 原因是离线融合审计调用了正式推理加载函数。
- 正式推理函数默认使用全部 `train_idx` 构造标签传播特征；但离线验证时，当前 split 的验证节点也属于全局 `train_idx`。
- 因此验证节点标签被放入标签传播特征，形成标签泄漏，导致分数虚高到 `0.95+`。

修复：

- `framework/code/a1_sign_infer.py`
  - `load_sign_probs()` 新增可选 `label_idx` 参数。
  - 正式推理不传参数，仍使用全部训练标签。
  - 离线审计传当前 split 的训练折，避免验证标签泄漏。
- `framework/code/a1_sign_config_ensemble_audit.py`
  - 加载每个 split checkpoint 时传入 `fit_idx`。
- `framework/code/a1_mixed_ensemble_audit.py`
  - 同步修复 SIGN 离线加载逻辑，避免后续异构融合审计踩同类问题。

需要重新运行：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp060_a1_sign_config_ensemble_audit.sh
```

修复后可信结果：

| candidate | mean | min | std | 结论 |
| --- | ---: | ---: | ---: | --- |
| undir:0.3 + undir_reverse:0.7 | 0.764384 | 0.748858 | 0.013259 | 最优均值 |
| undir:0.7 + undir_reverse:0.3 | 0.763653 | 0.754338 | 0.010044 | 更稳但均值略低 |
| undir_reverse:1 | 0.763653 | 0.747032 | 0.015187 | 单配置基线 |
| undir:0.3 + undir_reverse:0.4 + all_h2:0.3 | 0.763653 | 0.749772 | 0.011941 | 备选 |

结论：

- 修复后分数回到可信范围，第一次 `0.955+` 结果确认是标签泄漏。
- 最优融合 `undir:0.3 + undir_reverse:0.7` 比 Exp-059 单配置 `0.763653` 提升到 `0.764384`，提升 `+0.000731`。
- 这是小幅提升，可以生成候选提交包，但不是足以追第一名的决定性突破。

新增提交脚本：

- `framework/scripts/build_exp060_a1_config_ensemble_candidate.sh`
  - 5个 `undir` checkpoint 总权重 `0.3`，每个权重 `0.06`。
  - 5个 `undir_reverse` checkpoint 总权重 `0.7`，每个权重 `0.14`。
  - A2 沿用 Exp-044 线上最佳结果。

线上结果：

- 提交时间：2026-06-29 16:28:03
- 总分：`0.6286`
- A1：`0.7550`
- A2：`0.5023`

结论：

- A1 从 Exp-055 的 `0.7466` 提升到 `0.7550`，Exp-060 方向线上有效。
- A2 沿用 Exp-044，保持 `0.5023`。
- 距离第一名仍有两块差距：
  - A1：`0.7550 -> 0.76845`
  - A2：`0.5023 -> 0.50967`

---

## Tool-028：Exp-061 A1 Exp-060有效配置全标签重训候选

日期：2026-06-29

目标：

- 基于已被线上验证有效的 Exp-060 配置，做一次更大幅度的 A1 尝试。
- 当前 Exp-060 提交用的是 5 折 split checkpoint，每个模型训练时只看过约 90% 训练标签。
- Exp-061 使用全部 `train_idx` 标签重训模型，让模型参数本身也利用全部监督信号。

新增文件：

- `framework/scripts/run_exp061_a1_fulltrain_config_ensemble_candidate.sh`

设计：

- 训练两套全标签模型：
  - `undir`：标签传播方向 `undirected`，总权重 `0.3`。
  - `undir_reverse`：标签传播方向 `undirected,reverse`，总权重 `0.7`。
- 每套训练 5 个 seed：`42,777,2024,2026,3407`。
- 每个模型的固定训练 epoch 从 Exp-059 对应 split checkpoint 的 `best_epoch` 读取。
- 训练完成后按 Exp-060 最优融合权重生成 A1。
- A2 继续沿用 Exp-044，便于归因。

运行命令：

```bash
cd /home/aliagent/framework
CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp061_a1_fulltrain_config_ensemble_candidate.sh
```

候选提交包：

```text
output/exp061_a1_fulltrain_config_ensemble_candidate/prediction.zip
```

风险：

- 全标签重训没有真实验证集，离线指标不能直接证明线上提升。
- 但 Exp-060 已证明该特征配置线上有效，且过去 Exp-055 全标签重训带来过小幅线上增益，因此值得作为 A1 下一次候选。

生成结果：

- A1 输出：`output/exp061_a1_fulltrain_config_ensemble_candidate/A1.csv`
- 提交包：`output/exp061_a1_fulltrain_config_ensemble_candidate/prediction.zip`
- 提交格式校验：通过
- A1 类别分布：
  - 类别 0：78
  - 类别 1：383
  - 类别 2：283
  - 类别 3：70
  - 类别 4：1215
  - 类别 5：47
  - 类别 6：82
  - 类别 7：146
  - 类别 8：409
  - 类别 9：38

阶段结论：

- Exp-061 与 Exp-060 的类别分布接近，没有出现明显类别坍缩。
- A2 仍沿用 Exp-044，因此如果提交该包，线上变化主要归因于 A1。
- 建议作为下一次 A1 单变量验证提交。

---

## Tool-029：Exp-062 A2冷启动/短历史桶受控替换候选

日期：2026-06-29

目标：

- 当前 A2 最强线上版本是 Exp-044：`0.5023`。
- Exp-045 多 seed 模型离线更强，但整体线上降到 `0.4957`。
- 因此不能整体替换 A2，需要只对冷启动/短历史用户做受控替换，降低分布漂移风险。

新增文件：

- `framework/code/a2_bucket_blend.py`
  - 输入 base A2、alt A2、test.csv。
  - 按历史长度桶选择哪些用户使用 alt 推荐。
  - 输出混合后的 A2.csv，并打印变化率。
- `framework/scripts/run_exp062_a2_bucket_blend_candidates.sh`
  - base：`output/exp044_a2_feature_fusion/A2.csv`
  - alt：`output/exp045_a2_feature_multiseed/A2.csv`
  - 生成 5 个候选：
    - `len0`
    - `len1`
    - `len2_3`
    - `len0_len1`
    - `len0_len1_len2_3`

本地生成结果：

| 候选 | 替换桶 | 总体changed | top1_changed | Top10 overlap | 风险 |
| --- | --- | ---: | ---: | ---: | --- |
| len0 | len=0 | 34.47% | 4.85% | 96.98% | 低 |
| len1 | len=1 | 9.81% | 0.72% | 99.09% | 低但影响小 |
| len2_3 | len=2-3 | 43.65% | 4.87% | 96.29% | 中 |
| len0_len1 | len=0,len=1 | 44.28% | 5.57% | 96.07% | 中 |
| len0_len1_len2_3 | len=0,len=1,len=2-3 | 87.93% | 10.44% | 92.36% | 高 |

建议：

- 若要用一次线上提交探测 A2，优先提交 `len0`。
- 理由：空历史用户有 `3515/10000`，是 A2 最大冷启动桶；`len0` 保持总体 overlap 接近 `97%`，风险比整体替换低得多。
- 不建议优先提交 `len0_len1_len2_3`，因为它影响近 90% 用户，和 Exp-045 整体替换过于接近。

---

## Tool-030：Exp-063 A1 Exp-061 + A2 len0 联合候选

日期：2026-06-29

目标：

- 用户指出一次提交会同时返回 A1/A2 分数，因此可以同时验证两边，不必只做单变量提交。
- 生成一个联合候选：
  - A1：Exp-061 全标签重训候选。
  - A2：Exp-062 低风险 `len=0` 冷启动替换。

新增文件：

- `framework/scripts/build_exp063_a1_exp061_a2_len0_candidate.sh`

设计：

- 默认读取：
  - A1：`output/exp061_a1_fulltrain_config_ensemble_candidate/A1.csv`
  - A2 base：`output/exp044_a2_feature_fusion/A2.csv`
  - A2 alt：`output/exp045_a2_feature_multiseed/A2.csv`
- 自动生成 A2 `len=0` 桶混合结果。
- 打包为：

```text
output/exp063_submit_a1_exp061_a2_len0/prediction.zip
```

解释：

- A1 负责验证全标签重训是否能在 Exp-060 基础上继续提升。
- A2 负责低风险探测冷启动用户是否适合 Exp-045 的模型融合推荐。
- 因为平台会分别返回 A1/A2 分数，所以即使联合提交，也能判断哪一侧有效。
