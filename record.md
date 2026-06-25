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
