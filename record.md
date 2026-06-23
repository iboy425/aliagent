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

## Exp-001 基础设施与优化代码落地

实验编号：Exp-001

时间：待运行

目标：按三等奖目标计划落地第一批代码基础设施，包括 A2 推理融合、A1 稳健训练、提交校验工具和 Agent 轨迹增强。

修改文件：

- 待记录

修改内容：

- 待记录

运行命令：

```bash
cd /home/aliagent/framework
python3 -m py_compile code/*.py agent/*.py main.py
```

本地指标：

- 待运行

线上指标：

- 待提交

结论：

- 待记录

是否保留：

- 待定

下一步：

- 待记录
