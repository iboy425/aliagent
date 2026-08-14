# AliAgent

> 面向“稀疏反馈下的自动化实验挑战”的端到端科研 Agent：在有限预算、串行实验和稀疏反馈约束下，围绕图节点分类与金融产品推荐任务持续完成 **分析 → 设计 → 实验 → 反馈 → 迭代 → 提交**。

本仓库是一个面向天池图学习竞赛的实验型项目，同时包含两部分：

1. **通用自动化科研 Agent 框架**：负责读取任务信息、维护实验记忆、分析历史反馈、生成或修改实验方案、运行训练/推理并决定下一步；
2. **竞赛专项预测器与实验工具箱**：包含节点分类、序列推荐、图传播、后处理、集成、分桶融合、离线评估和提交校验等大量可复现实验代码。

仓库中的 `record.md` 是实验历史的主要事实来源，记录了从基础 Agent baseline 到 100+ 轮专项实验的演化过程、离线结果、线上反馈与失败经验。

---

## 目录

* [1. 项目背景](#1-项目背景)
* [2. 比赛任务](#2-比赛任务)
* [3. 当前历史结果](#3-当前历史结果)
* [4. 系统架构](#4-系统架构)
* [5. Agent 工作流](#5-agent-工作流)
* [6. A1：产品分类方法演化](#6-a1产品分类方法演化)
* [7. A2：产品推荐方法演化](#7-a2产品推荐方法演化)
* [8. 关键实验阶段](#8-关键实验阶段)
* [9. 失败实验与经验总结](#9-失败实验与经验总结)
* [10. 仓库结构](#10-仓库结构)
* [11. 环境安装](#11-环境安装)
* [12. 数据准备](#12-数据准备)
* [13. 运行 Agent](#13-运行-agent)
* [14. 独立训练与推理](#14-独立训练与推理)
* [15. 专项实验脚本](#15-专项实验脚本)
* [16. 提交文件与校验](#16-提交文件与校验)
* [17. 实验记录与可复现性](#17-实验记录与可复现性)
* [18. 合规说明](#18-合规说明)
* [19. 后续方向](#19-后续方向)

---

# 1. 项目背景

比赛关注的并不是单次训练一个固定模型，而是一个受约束的自动化实验闭环：

```text
读取任务 / 数据 / 历史实验
        ↓
分析当前瓶颈
        ↓
提出下一轮假设
        ↓
设计模型 / 特征 / 参数 / 后处理
        ↓
运行实验
        ↓
读取 val / OOF / 线上反馈 / 耗时 / 失败日志
        ↓
更新实验记忆
        ↓
CONTINUE / PIVOT / STOP
        ↓
预算内持续迭代
```

核心约束包括：

| 约束         | 含义                        |
| ---------- | ------------------------- |
| 有限预算       | 无法进行无穷超参数搜索，需要提高单次实验的信息增益 |
| 稀疏反馈       | 主要依赖内部验证、OOF、训练日志和有限线上反馈  |
| 禁止并行       | 实验必须串行组织，Agent 需要做探索/利用权衡 |
| 最终目标不可直接观察 | 本地验证指标与真实测试榜可能存在明显分布偏移    |
| 可复现要求      | 实验、配置、代码、轨迹和最终预测需要能够互相对应  |

因此，本项目将问题视为一种 **受预算约束的长程实验控制问题**，而不只是普通的 GNN / 推荐模型调参。

---

# 2. 比赛任务

## 2.1 A1：产品分类

A1 是图节点分类任务。

输入主要包括：

* CSR 邻接矩阵；
* CSR 节点特征矩阵；
* 已公开训练节点标签；
* `train_idx`；
* `test_idx`。

目标：

```text
test_idx,label
18,3
19,7
...
```

评测指标：

```text
Accuracy
```

本项目针对 A1 先后尝试了：

* GCN；
* GraphSAGE；
* GAT；
* Label Propagation；
* Correct & Smooth；
* SIGN 风格多跳传播特征；
* 图结构统计特征；
* 多方向传播；
* 多 seed / 多 split / 多配置集成；
* full-label retrain；
* OOF meta stacking；
* class-wise expert selection；
* 邻居标签统计特征；
* 伪标签、自训练、SVD、Label Smoothing 等辅助方向。

---

## 2.2 A2：产品推荐

A2 是面向新用户/短历史用户的 Top-10 推荐任务。

输入包括：

* `train.csv`
* `test.csv`
* `user.csv`
* `item.csv`

训练样本中包含：

```text
uid
target_iid
item_seq_raw
item_seq_dedup
item_seq_counts
```

预测输出：

```text
uid,prediction
u000001,"i000001,i000002,...,i000010"
```

评测指标：

```text
NDCG@10
```

本项目针对 A2 先后尝试了：

* Popularity baseline；
* last-item transition；
* full-history co-occurrence；
* history frequency；
* recency decay；
* Jaccard-normalized co-occurrence；
* 用户画像条件先验；
* prefix profile combination；
* item 类别转移；
* GRU4Rec；
* SASRec；
* 神经 feature ranker；
* multi-seed ranker；
* test-like validation；
* 历史长度分桶；
* hard bucket replacement；
* RRF / soft rank fusion；
* suffix transition；
* empirical-Bayes cold-start；
* profile group gate；
* protected Top-N blend；
* item feature embedding；
* bucket-specific heads / experts；
* sparse short-history ranker。

---

## 2.3 最终分数

比赛最终分数：

```text
Score_final = 0.5 * Score_cls + 0.5 * Score_ndcg
```

这意味着 A1 与 A2 同等重要，单独优化一个任务很难获得稳定的总榜提升。

---

# 3. 当前历史结果

以下是 `record.md` 中具有代表性的线上结果演化。数值仅用于记录历史实验，不代表未来数据或 B 榜表现。

| 实验     |      A1 |      A2 |      Final | 关键变化                              |
| ------ | ------: | ------: | ---------: | --------------------------------- |
| Exp000 |  0.3777 |  0.2173 |     0.2975 | 初始完整 Agent baseline               |
| Exp006 |  0.6369 |  0.4647 |     0.5508 | 稳定 SAGE + 用户画像推荐                  |
| Exp013 |  0.6790 |  0.4545 |     0.5667 | A1 提升；GRU hybrid 线上退化             |
| Exp019 |  0.6874 |  0.4647 |     0.5760 | Correct & Smooth 生效               |
| Exp022 |  0.6874 | ≈0.4742 |    ≈0.5808 | 用户 profile prefix prior           |
| Exp030 |  0.6874 | ≈0.4746 |    ≈0.5810 | A2 规则微调                           |
| Exp044 | 0.68739 | 0.50230 |    0.59484 | 神经 feature ranker 突破 A2           |
| Exp051 | ≈0.7117 |  0.5023 |    ≈0.6070 | 五折 SIGN ensemble                  |
| Exp054 |  0.7459 |  0.5023 |     0.6241 | SIGN + label propagation features |
| Exp060 |  0.7550 |  0.5023 |     0.6286 | 多传播配置概率融合                         |
| Exp063 |  0.7590 |  0.5033 |     0.6311 | full-label A1 + A2 len=0 替换       |
| Exp066 |  0.7594 |  0.5033 |     0.6313 | A1 blend + len0/len1 方案           |
| Exp090 |  0.7601 |  0.5035 | **0.6318** | 稳定 A1 + A2 protected gate         |
| Exp111 |  0.7601 |  0.5035 | **0.6318** | 再次确认稳定基线                          |

截至当前实验记录，**0.6318 是仓库中已记录并验证的稳定线上最佳分数之一**。

项目后续 Exp112 / Exp113 继续探索 sparse short-history ranker，并生成了多种风险级别的候选，但在当前 `record.md` 末尾尚未记录新的线上突破结果。

---

# 4. 系统架构

AliAgent 的核心是 `ResearchOrchestrator` 驱动的四阶段科研闭环：

```text
┌──────────────────────────────────────────────────────────────┐
│                     ResearchOrchestrator                     │
│                                                              │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐            │
│  │ Literature │ → │ Diagnosis  │ → │   Design   │            │
│  └────────────┘   └────────────┘   └────────────┘            │
│         ↑                                  ↓                 │
│         │                           ┌────────────┐            │
│         └────────────────────────── │ Experiment │            │
│                                     └────────────┘            │
│                                            ↓                 │
│                                  CONTINUE / PIVOT / STOP      │
└──────────────────────────────────────────────────────────────┘
                         │
                         ↓
                Research Memory / Logs
                         │
                         ↓
                Best Model / Submission
```

主要模块：

```text
framework/agent/
├── config.py
├── orchestrator.py
├── phases.py
├── memory.py
├── llm_client.py
├── tools.py
└── code_generator.py
```

职责大致如下：

| 模块                  | 作用                                                 |
| ------------------- | -------------------------------------------------- |
| `orchestrator.py`   | 控制完整 Agent 生命周期与预算                                 |
| `phases.py`         | Literature / Diagnosis / Design / Experiment 四阶段实现 |
| `memory.py`         | 保存实验配置、指标、结论和历史状态                                  |
| `llm_client.py`     | 调用兼容 API 的大模型服务                                    |
| `tools.py`          | Shell、文件、代码验证等工具能力                                 |
| `code_generator.py` | 生成或修改预测器代码                                         |
| `config.py`         | 配置读取与运行参数管理                                        |

项目目标不是让 LLM 直接“输出答案”，而是让 Agent 在实验环境中形成：

```text
Hypothesis
→ Experiment
→ Feedback
→ Memory
→ Next Hypothesis
```

---

# 5. Agent 工作流

## 5.1 Literature

负责读取：

* 任务 README；
* 数据规模；
* 已有模型；
* 当前代码；
* 历史实验记录；
* 运行限制。

输出当前问题的基础理解和可行动候选。

## 5.2 Diagnosis

根据历史实验与当前指标判断：

* 是欠拟合还是过拟合；
* 当前瓶颈属于模型、特征、验证划分还是后处理；
* 哪些方向已经被实验否定；
* 哪些方向仍存在高信息增益。

## 5.3 Design

生成下一轮可执行配置，例如：

* 调整模型结构；
* 修改传播方式；
* 新增统计特征；
* 改训练策略；
* 改融合方式；
* 新建专项脚本；
* 选择是否回滚到稳定基线。

## 5.4 Experiment

执行：

```text
代码检查
→ Smoke Test
→ 训练
→ 验证
→ 推理
→ 提交格式校验
→ 保存指标
```

最后根据收益决定：

```text
CONTINUE
PIVOT
STOP
```

---

# 6. A1：产品分类方法演化

A1 最终形成的路线不是“单个更大的 GNN”，而是逐步从 message-passing GNN 转向 **预计算图传播特征 + 轻量 MLP + 标签传播特征 + 多专家融合**。

---

## 6.1 GCN / GraphSAGE baseline

仓库基础模型支持 GCN、GraphSAGE 等 GNN。

GraphSAGE 采用：

```text
self representation
+
aggregated neighbor representation
→ concat
→ linear
→ activation
```

早期实验发现：

* 相比简单 GCN，SAGE 更稳定；
* 稳定 split 和 seed 非常重要；
* `class_weight=balanced` 并没有改善本任务，反而造成明显退化；
* 单纯扩大 hidden dim / 层数的收益有限。

---

## 6.2 Multi-seed / fixed validation ensemble

A1 的早期有效提升来自：

* 固定验证划分；
* 多随机种子；
* 多 checkpoint / 多模型概率平均；
* 避免只根据单次随机划分判断模型优劣。

这使 A1 从单模型阶段逐步提升到约 `0.65+`。

---

## 6.3 Label Propagation

实现：

```text
framework/code/a1_labelprop.py
```

基本思想：

```text
已知训练标签
→ 在图上迭代传播
→ 得到每个节点的 label probability
→ 与模型概率融合
```

早期结果表明纯 LP 不是最强模型，但与 GNN 的误差具有一定互补性。

---

## 6.4 Correct & Smooth

实现：

```text
framework/code/a1_correct_smooth.py
```

流程：

```text
模型预测
→ Correct：利用已知标签修正 residual
→ Smooth：在图上平滑修正后的分布
```

C&S 是早期 A1 的明确有效方向之一，帮助 A1 从约 `0.679` 提升到约 `0.687`。

但后期当 label propagation 已经直接进入 SIGN 特征后，再强行做较强 C&S 的边际收益明显下降，甚至可能发生过平滑。

---

## 6.5 Sparse GAT 与 GAT/GCN ensemble

中期尝试过：

* sparse GAT；
* GAT 超参数网格；
* GAT + GCN；
* greedy weighted ensemble；
* checkpoint weighted ensemble；
* ensemble + Correct & Smooth。

这些实验提升了对图注意力模型的理解，但最终没有超过后续 SIGN 路线，因此不是当前主干。

---

## 6.6 SIGN：多跳传播特征

A1 真正的大幅突破来自：

```text
framework/code/a1_sign_mlp.py
framework/code/a1_sign_infer.py
```

核心思路：

不再让深层 GNN 每个 epoch 都进行复杂 message passing，而是先计算多跳图传播特征：

```text
X
A X
A² X
...
```

再将多尺度特征送入 MLP。

简化表示：

```text
H = concat(
    X,
    A X,
    A² X,
    graph_features,
    label_features
)

prediction = MLP(H)
```

这一思路具有几个优点：

* 图传播可以预计算；
* 训练稳定；
* 对小规模稀疏图十分高效；
* 更容易融合不同方向和不同类型的传播结果；
* 在固定运行预算下可以训练更多 seed / config。

---

## 6.7 Label Propagation 作为输入特征

这是 A1 最重要的改进之一。

与“训练后再做 LP”不同，后期方案将标签传播得到的概率直接加入节点输入：

```text
raw attribute propagation
+
label propagation features
+
structure features
→ MLP
```

即：

```text
X_sign = [
    X,
    A X,
    A² X,
    LP_1,
    LP_2,
    ...
]
```

该方案在严格避免 validation-label leakage 的 OOF / split 评估中表现明显强于单纯 SIGN。

Exp054 使用五折 SIGN + label features 后，线上 A1 达到约：

```text
0.7459
```

---

## 6.8 图结构特征

后续加入：

* degree；
* in-degree / out-degree；
* 邻域统计；
* 图传播相关结构量。

结构特征带来小幅但相对稳定的增益。

---

## 6.9 多方向传播

针对原始邻接关系尝试：

```text
undirected
reverse / incoming
mixed direction
all_h2
```

实验表明：

* 只使用单一方向并不是最优；
* reverse / incoming 信息存在补充价值；
* `undir + reverse` 的配置族可以形成有效专家；
* 多配置概率融合优于简单只保留单个 best config。

---

## 6.10 Full-label retrain

完成 OOF / split 选择后，最终候选使用全部公开训练标签重新训练：

```text
OOF / split
    ↓
选择 epoch / config / seed
    ↓
full train_idx retrain
    ↓
test inference
```

这一步避免最终模型白白损失可用标注数据。

Exp061 / Exp063 一带证明 full-label retrain 是有效的线上提升来源。

---

## 6.11 Config ensemble

主要通过：

```text
framework/code/a1_sign_config_ensemble_audit.py
framework/code/a1_ensemble_eval.py
```

对多个传播配置、seed 和模型概率进行融合。

相比只在标签层做多数投票，概率级融合可以保留置信度信息。

---

## 6.12 OOF Meta Stacking

实现：

```text
framework/code/a1_sign_meta_stack.py
framework/code/a1_sign_classwise_stack.py
framework/code/a1_meta_bias_search.py
```

输入不只包含各 SIGN expert 的类别概率，还可以包含：

* confidence；
* top1-top2 margin；
* graph structural feature；
* expert disagreement；
* 简单 meta feature。

目标是在 OOF 上学习：

```text
P_final = MetaModel(
    P_expert_1,
    P_expert_2,
    ...,
    confidence,
    margin,
    graph_meta
)
```

该方向最终将稳定 A1 推到约：

```text
0.7601
```

---

## 6.13 Neighbor gate / label-neighbor statistics

实现：

```text
framework/code/a1_neighbor_gate.py
```

尝试利用：

* 已知标签邻居数量；
* 邻居多数类；
* 邻居纯度；
* 局部标签统计。

实验说明这些统计确实含有信号，但简单 hard override / majority gate 容易破坏强 SIGN 模型，因此没有成为稳定主干。

后续又尝试把邻居统计作为 meta features，而不是直接硬改预测。

---

## 6.14 其他 A1 实验

还尝试过：

| 方法                               | 结论                             |
| -------------------------------- | ------------------------------ |
| `class_weight=balanced`          | 明显不适合当前数据                      |
| SVD / 特征降维                       | 没有稳定超过 raw + propagation       |
| pseudo-label self-training       | OOF 增益很小，线上风险高                 |
| 强 C&S                            | 后期与 label-feature SIGN 重复，易过平滑 |
| naive neighbor majority override | 不稳定                            |
| Label Smoothing                  | 没有形成决定性提升                      |
| class-wise expert selection      | 有小幅 OOF 信号，但线上收益不足             |
| label-seed feature               | 进行过审计，未成为稳定主路线                 |
| bucket / output head 变化          | 局部有效但泛化证据不足                    |

---

# 7. A2：产品推荐方法演化

A2 的难点与普通推荐系统不同：测试集包含大量 **无历史或极短历史用户**。因此一个随机划分上的强序列模型并不一定能够在线上表现好。

项目后期形成的核心原则是：

```text
先构建强且稳定的 heuristic / ranker base
+
使用 test-like validation
+
按历史长度/用户画像做受控融合
+
严格控制预测漂移
```

---

## 7.1 Popularity baseline

最简单基线：

```text
P(item) ∝ item 在训练目标中的频率
```

对无历史用户具有一定能力，但整体上限较低。

---

## 7.2 Last-item transition

统计：

```text
P(target | last_item)
```

相比纯 popularity 有明显提升。

---

## 7.3 Full-history co-occurrence

早期实验发现：

```text
使用整个历史序列提供的 co-occurrence 信号
```

通常优于只使用最后一个 item。

一个重要负面经验是：

> 不要默认把用户历史中出现过的 item 从候选集合中硬过滤掉。

因为目标 item 可能再次出现，硬过滤会直接损失有效召回。

---

## 7.4 Sequence source / recent window 搜索

测试过：

* `item_seq_raw`
* `item_seq_dedup`
* 不同 `recent_n`
* history filter 规则

本地结果显示某些场景下：

```text
item_seq_raw
+
recent window
+
no hard history filter
```

更稳定。

---

## 7.5 History frequency

除了“是否出现”，还加入：

```text
item 在用户历史中的出现次数
```

作为排序信号。

收益不大，但可以与 co-occurrence 互补。

---

## 7.6 Recency decay

为更近的交互赋予更大权重：

```text
w_t = decay ^ distance
```

后期证明轻量 recency decay 有小幅正向价值。

---

## 7.7 User profile prior

使用 `user.csv` 中匿名类别特征建立：

```text
P(target | user_profile)
```

例如：

```text
u_cat_01
u_cat_01 + u_cat_02
u_cat_01 + u_cat_02 + u_cat_03
```

再与全局 popularity / history signal 融合。

该方法对 `len=0` 用户尤其重要。

---

## 7.8 Prefix profile combination

Exp021/022 证明：

```text
prefix profile combination
```

比简单单字段 prior 更有效。

例如：

```text
P(target | cat1)
P(target | cat1, cat2)
P(target | cat1, cat2, cat3)
```

逐级融合，可以明显改善冷启动用户。

相比之下，暴力加入所有任意特征组合更容易过拟合，并没有稳定优于 prefix 设计。

---

## 7.9 Item feature transition

尝试利用 `item.csv` 的：

```text
i_cat_*
i_bucket_*
```

建立类别转移或 side-information 模型。

简单 item-category transition 没有明显提升，因此没有成为主干。

---

## 7.10 GRU4Rec

基础代码支持 GRU4Rec。

尝试过：

```text
sequence → embedding → GRU → target logits
```

也尝试与 heuristic 混合。

离线某些指标有改善，但早期线上结果显示：

```text
GRU hybrid < heuristic baseline
```

说明普通随机验证与真实测试短历史分布存在明显偏移。

---

## 7.11 SASRec

仓库也实现了 SASRec。

由于：

* 大量测试用户历史极短；
* 序列信号不足；
* 时间预算有限；

SASRec 并未在当前数据上形成稳定主路线。

---

## 7.12 Test-like Validation

这是 A2 最关键的方法论改进之一。

普通随机 holdout 会让验证集中的长历史用户比例过高，从而过度奖励强序列模型。

因此项目加入：

```text
framework/code/a2_offline_eval.py
```

按照真实测试集的历史长度分布对离线样本进行模拟 / 加权：

```text
len = 0
len = 1
len = 2-3
len = 4-10
len > 10
```

得到更接近线上分布的 weighted NDCG。

这一改变解释了为什么：

```text
离线更强的 GRU / multi-seed ranker
```

可能在线上反而下降。

---

## 7.13 Jaccard-normalized Co-occurrence

中期对 co-occurrence 公式做了重要升级。

不再只使用原始共现次数，而是采用类似：

```text
cooccur(a, target)
-------------------------
union / frequency normalization
```

的 Jaccard 风格归一化，以降低热门 item 的无条件支配。

该方法使 test-like weighted NDCG 明显提升，并成为后续 A2 强 heuristic 的组成部分。

---

## 7.14 Neural Feature Ranker

实现：

```text
framework/code/a2_feature_ranker.py
```

不是简单地用神经网络替代所有规则，而是让模型同时利用：

* sequence representation；
* heuristic score；
* popularity；
* co-occurrence；
* Jaccard；
* user profile；
* 其他候选级特征。

Exp044 是 A2 的重要突破点：

```text
A2 online ≈ 0.50230
```

相比此前约 `0.474x` 有明显提升。

---

## 7.15 Multi-seed Feature Ranker

Exp045 对 feature ranker 做了：

* 多 seed；
* 更高模型权重；
* 更强 offline ensemble。

离线 weighted NDCG 更高，但全量替换后线上反而下降。

这一实验形成了非常重要的经验：

> 一个“整体离线更高”的 ranker 不代表可以全量替换线上稳定 base；它可能只在某些用户桶上是真专家。

因此 Exp045 后来主要作为 **alternative expert** 使用。

---

## 7.16 Bucket Blend

实现：

```text
framework/code/a2_bucket_blend.py
```

按照用户历史长度：

```text
len=0
len=1
len=2-3
len=4-10
len>10
```

决定使用不同预测器。

实践中：

* 只替换 `len=0` 是低风险且有过线上正收益的操作；
* 扩大到更多桶时，榜单表现并不总是继续增加；
* hard replacement 的预测漂移较大。

---

## 7.17 Rank Fusion / RRF

实现：

```text
framework/code/a2_rank_fusion.py
```

尝试：

```text
base ranking
+
alternative ranking
→ Reciprocal Rank Fusion / soft fusion
```

也尝试加入 last-1 / last-2 / last-3 suffix transition。

离线存在局部信号，但 Exp069 一类线上实验说明简单 RRF / suffix 组合并不稳定，因此没有直接替代 Exp044/Exp066 稳定方案。

---

## 7.18 Empirical-Bayes Cold Start

实现：

```text
framework/code/a2_coldstart_eb.py
framework/code/a2_combo_eb.py
```

目标是解决 profile counter 在低频用户群上的高方差问题。

将：

```text
P(target | profile)
```

向全局分布做 shrinkage：

```text
P_EB
=
λ(group_count) * P_group
+
(1 - λ(group_count)) * P_global
```

局部冷启动验证中获得过较明显收益，但线上转换并不完全稳定。

---

## 7.19 Profile Group Gate

实现：

```text
framework/code/a2_profile_gate.py
```

思路：

不是让 alternative ranker 覆盖所有用户，而是只在离线证据较强的 profile group 中启用。

后续进一步做：

* strict gate；
* Leave-One-Split-Out / stability audit；
* group minimum support；
* 低漂移选择。

实验说明 profile gate 能减少风险，但过度依赖本地 group gain 仍容易过拟合。

---

## 7.20 Protected Blend

实现：

```text
framework/code/a2_protected_blend.py
```

为了降低线上漂移，引入：

```text
保留稳定 base Top1 / Top2
+
只重排 TopN 后部
```

例如：

```text
keep Top1
→ alternative model rerank positions 2-10
```

这种方式在稳定性方面优于全量替换，也是 Exp090 A2 稳定方案的一部分。

但 Exp111 也带来一个反面结论：

> 如果只在旧 Top10 内部重排，且候选集合几乎没有变化，线上可能完全没有收益。

因此后续 A2 必须同时考虑“排序质量”和“新候选召回”。

---

## 7.21 Item Feature Embedding

后期进一步训练利用 `item.csv` 匿名属性的 side-feature model。

本地能产生一定新信号，但总体没有超过强 Exp045 ranker，protected gain 也较小，因此未成为主模型。

---

## 7.22 Bucket-specific Heads / Experts

尝试让不同历史长度桶拥有：

* 单独输出 head；
* 单独 loss；
* 单独 expert；
* 单独模型权重。

该方向理论上合理，但当前数据中容易因桶内样本减少而过拟合，线上证据不足。

---

## 7.23 Sparse Short-history Ranker

实现：

```text
framework/code/a2_sparse_short_ranker.py
```

Exp112/113 专门针对：

```text
len=0
len=1
len=2-3
```

这些占测试主体的短历史用户。

当前生成了多种候选：

```text
pure
keep1_all
keep1_short
keep2_all
```

区别主要在于是否保留稳定 base 的 Top1 / Top2，以及 sparse ranker 能改动多少候选。

该方向具有更大的潜在上升空间，同时也伴随更大的 Top10 / Top1 漂移风险。

---

# 8. 关键实验阶段

与其逐条罗列 100+ 个实验，下面按阶段总结仓库中的主要演进。

| 阶段                  | 代表实验       | 核心内容                                                       | 主要结论                                |
| ------------------- | ---------- | ---------------------------------------------------------- | ----------------------------------- |
| Agent baseline      | Exp000-001 | 完整闭环、submission validator、offline evaluator                | 基础框架可运行，但预测器很弱                      |
| A2 heuristic        | Exp002-005 | history cooccur、window、profile prior                       | history + profile 明显强于 popularity   |
| 稳定 SAGE             | Exp006-010 | GraphSAGE、多 seed、LP                                        | class weight 无效，多 seed 有效           |
| Neural rec          | Exp012-015 | GRU4Rec / hybrid / frequency                               | 离线提升不一定转化为线上                        |
| C&S                 | Exp016-019 | Correct & Smooth                                           | A1 明确正收益                            |
| Test-like A2        | Exp020-030 | 测试分布加权、profile prefix、recency                              | 验证设计本身是核心模型组件                       |
| GAT exploration     | Exp031-040 | sparse GAT、ensemble、C&S                                    | 有信号，但被后续 SIGN 超越                    |
| Jaccard / ranker    | Exp041-045 | Jaccard cooccur、feature ranker                             | A2 突破 0.50                          |
| SIGN breakthrough   | Exp046-055 | multi-split、SIGN、label features                            | A1 从 0.69 段快速提升到 0.746              |
| SIGN refinement     | Exp056-061 | pseudo、structure、SVD、direction、config ensemble             | structure/direction 有效，SVD/pseudo 弱 |
| Joint stabilization | Exp062-070 | bucket blend、RRF、neighbor gate、EB                          | A1 0.759，A2 len0 替换有效               |
| Meta stack / gate   | Exp074-090 | A1 OOF stack、A2 group gate、protected blend                 | 稳定 best 到 0.6318                    |
| Robustness audits   | Exp094-111 | full-data ranker、item features、meta feature、bucket experts | 多个离线小增益未能转化线上                       |
| Short-history push  | Exp112-113 | sparse short-history ranker                                | 当前最激进的新方向，尚需线上验证                    |

---

# 9. 失败实验与经验总结

比赛代码中保留失败实验非常重要，因为 Agent 需要知道哪些方向已经被验证过。

## 9.1 A1

### `class_weight=balanced`

失败原因：

* 改变类别边界后验证集明显下降；
* 数据不平衡并不是当前主要瓶颈。

结论：

```text
不要因为类别计数不均衡就默认开启 balanced class weight。
```

### SVD / PCA 风格降维

在已有强 raw sparse feature + graph propagation 下，没有稳定收益。

结论：

```text
高维匿名特征并不意味着一定应该先做低秩压缩。
```

### Pseudo-label self-training

本地提升非常小，错误伪标签可能被图传播进一步放大。

结论：

```text
只有在 OOF 明显稳定提升时才值得承担风险。
```

### 简单 neighbor majority override

局部邻居标签很强，但 hard override 会伤害全局强模型。

结论：

```text
局部统计更适合作为 meta feature / soft expert，而不是无条件覆盖。
```

---

## 9.2 A2

### Hard history filtering

把用户看过的 item 全部从候选集中删除会降低分数。

原因：

```text
目标 item 可能是重复交互。
```

### GRU / SASRec 并非天然更强

序列模型在普通随机验证上容易占优，但真实测试包含大量极短历史用户。

结论：

```text
验证集分布比模型复杂度更重要。
```

### Item category transition

简单匿名 item 类别转移并没有提供足够增量。

### All-combination profile prior

任意组合所有 user categorical fields 会造成稀疏和过拟合，prefix hierarchy 更稳。

### 全量使用 Exp045

Exp045 离线更强，但线上下降。

结论：

```text
alternative expert 应该先做 bucket / gate / protected blend，而不是直接替换整个 test。
```

### 只在旧 Top10 内重排

Exp111 说明，如果候选集合几乎完全不变，仅做小范围位置调整可能没有线上收益。

结论：

```text
A2 后续不仅要 rerank，还必须关注 candidate recall。
```

---

## 9.3 关于基于榜单聚合反馈的实验

`record.md` 中曾出现过基于极少量 aggregate leaderboard score 做统计校准 / MAP 推断的探索性实验。

该路线：

* 约束远远不足；
* 极易过拟合；
* 实际线上表现也发生明显退化；
* 更重要的是，比赛规则禁止通过排行榜反馈反推测试答案。

因此这类分支应视为：

```text
DEPRECATED / ABANDONED
```

**不属于本项目推荐的合规复现路线，也不应在后续提交或 Agent 自动决策中使用。**

---

# 10. 仓库结构

```text
aliagent/
├── AGENTS.md
├── SPEC_v2.md
├── my_plan.md
├── plan.md
├── record.md
├── 竞赛说明.txt
└── framework/
    ├── agent/
    │   ├── config.py
    │   ├── orchestrator.py
    │   ├── phases.py
    │   ├── memory.py
    │   ├── llm_client.py
    │   ├── tools.py
    │   └── code_generator.py
    │
    ├── code/
    │   ├── models.py
    │   ├── datasets.py
    │   ├── train.py
    │   ├── infer.py
    │   ├── utils.py
    │   ├── validate_submission.py
    │   │
    │   ├── a1_correct_smooth.py
    │   ├── a1_ensemble_eval.py
    │   ├── a1_labelprop.py
    │   ├── a1_meta_bias_search.py
    │   ├── a1_mixed_ensemble_audit.py
    │   ├── a1_neighbor_gate.py
    │   ├── a1_sign_classwise_stack.py
    │   ├── a1_sign_config_ensemble_audit.py
    │   ├── a1_sign_infer.py
    │   ├── a1_sign_meta_stack.py
    │   ├── a1_sign_mlp.py
    │   ├── a1_sign_pseudo_audit.py
    │   │
    │   ├── a2_bucket_blend.py
    │   ├── a2_coldstart_eb.py
    │   ├── a2_combo_eb.py
    │   ├── a2_compare_submissions.py
    │   ├── a2_feature_ranker.py
    │   ├── a2_grid_search.py
    │   ├── a2_offline_eval.py
    │   ├── a2_profile_gate.py
    │   ├── a2_protected_blend.py
    │   ├── a2_rank_fusion.py
    │   ├── a2_sparse_short_ranker.py
    │   └── rec_heuristics.py
    │
    ├── data/
    │   ├── cls_data/
    │   └── rec_data/
    │
    ├── scripts/
    │   ├── run_exp*.sh
    │   └── build_exp*.sh
    │
    ├── tests/
    ├── BASELINE.md
    ├── config.yaml
    ├── main.py
    ├── repro_agent_step6.yaml
    ├── requirements.txt
    ├── .env.local.example
    └── run_agent_with_api.sh
```

说明：

* `agent/`：通用 Agent 编排；
* `code/`：预测器、评估、融合与专项实验工具；
* `scripts/`：按 Exp 编号保存的可复现实验入口；
* `record.md`：完整实验演化记录；
* `SPEC_v2.md`：Agent 架构规格；
* `my_plan.md` / `plan.md`：开发与比赛策略；
* `BASELINE.md`：基础预测器说明。

---

# 11. 环境安装

## 11.1 Python 依赖

主要技术栈：

```text
Python 3
PyTorch >= 2.0
NumPy >= 1.24
SciPy >= 1.10
Pandas >= 2.0
scikit-learn >= 1.3
PyYAML >= 6.0
requests >= 2.31
```

推荐：

* Python 3.9+；
* 有 GPU 时使用 CUDA；
* 强 SIGN / feature ranker / 多 seed 实验建议使用 NVIDIA GPU；
* 基础启发式、校验和部分图预处理可在 CPU 上运行。

## 11.2 安装

```bash
git clone https://github.com/iboy425/aliagent.git
cd aliagent

python -m venv .venv
```

Linux / macOS：

```bash
source .venv/bin/activate
```

Windows Git Bash：

```bash
source .venv/Scripts/activate
```

安装依赖：

```bash
pip install -r framework/requirements.txt
```

---

# 12. 数据准备

仓库按照以下目录读取竞赛数据：

```text
framework/data/
├── cls_data/
│   └── A1.npz
└── rec_data/
    ├── train.csv
    ├── test.csv
    ├── user.csv
    ├── item.csv
    ├── sample_submission.csv
    ├── metadata.json
    └── README.md
```

请使用比赛官方数据，并遵守比赛数据与外部资源规则。

不要将：

* 私有标签；
* 测试答案；
* 未授权外部数据；
* 真实 API Secret

提交到仓库。

---

# 13. 运行 Agent

## 13.1 配置 API

仓库提供：

```text
framework/.env.local.example
```

复制：

```bash
cd framework
cp .env.local.example .env.local
chmod 600 .env.local
```

编辑 `.env.local`：

```bash
DASHSCOPE_API_KEY="YOUR_KEY"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL="qwen-plus"

AGENT_OUTPUT_DIR="output/repro_agent_api_qwen_plus_round"
AGENT_BUDGET="1"
```

**不要提交 `.env.local` 或真实 API Key。**

---

## 13.2 一键运行

```bash
cd framework
./run_agent_with_api.sh
```

也可以直接运行：

```bash
python main.py --task 1 --task 2 --budget 10
```

仅运行 A1：

```bash
python main.py --task 1 --budget 5
```

从配置文件运行：

```bash
python main.py --config config.yaml
```

恢复历史 Agent memory：

```bash
python main.py --resume ./output/task1/research_memory.json
```

---

# 14. 独立训练与推理

基础训练入口仍保留在：

```text
framework/code/train.py
framework/code/infer.py
```

## 14.1 A1 GraphSAGE baseline

```bash
cd framework/code

python train.py \
  --task task1 \
  --data_path data/cls_data/A1.npz \
  --model_type sage \
  --hidden_dim 128 \
  --num_layers 2 \
  --lr 0.01 \
  --epochs 200 \
  --output_dir ../output/task1
```

推理：

```bash
python infer.py \
  --task task1 \
  --data_path data/cls_data/A1.npz \
  --checkpoint ../output/task1/best_model.pt \
  --output_path A1.csv
```

---

## 14.2 A2 GRU4Rec baseline

```bash
python train.py \
  --task task2 \
  --data_path data/rec_data/ \
  --model_type gru4rec \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --lr 0.001 \
  --epochs 50 \
  --output_dir ../output/task2
```

推理：

```bash
python infer.py \
  --task task2 \
  --data_path data/rec_data/ \
  --checkpoint ../output/task2/best_model.pt \
  --output_path A2.csv
```

> 注意：这些命令是通用 baseline 入口。当前历史强方案由 `SIGN + label features + ensemble/meta stack` 和 `feature ranker + controlled A2 fusion` 等专项脚本构成，不等价于单独运行上述 baseline。

---

# 15. 专项实验脚本

`framework/scripts/` 保存了大量按实验编号组织的脚本，例如：

```text
build_exp040_final_candidate.sh
build_exp051_a1_sign_ensemble_candidate.sh
build_exp054_a1_sign_label_ensemble_candidate.sh
build_exp059_a1_undir_reverse_candidate.sh
build_exp060_a1_config_ensemble_candidate.sh
build_exp063_a1_exp061_a2_len0_candidate.sh
build_exp066_a1_blend90_a2_len0_len1_candidate.sh
build_exp068_a1_blend90_a2_rrf_soft_candidate.sh
build_exp069_a1_gate_a2_rrf_suffix_candidate.sh
build_exp070_a1_exp066_a2_coldstart_eb_candidate.sh
build_exp074_a1_meta_a2_eb_candidate.sh
build_exp075_a1_meta_threshold_a2_eb_candidate.sh
build_exp078_a1_exp075_a2_profile_gate_candidate.sh
build_exp079_a1_best_a2_exp066_stable.sh
build_exp082_a1_best_a2_strict_profile_gate.sh
build_exp084_a1_best_a2_protected_blend.sh
build_exp085_a1_best_a2_strict_gate_keep_top1.sh
...
```

并有对应：

```text
run_exp*.sh
```

用于训练、审计或生成中间结果。

由于后期实验存在明确的依赖关系，例如：

```text
Exp061 checkpoints
    ↓
Exp063 / Exp066 candidate
    ↓
Exp075 meta stack
    ↓
Exp079 / Exp090 stable joint candidate
```

因此建议在复现某个高编号实验前先查看：

1. `record.md` 中该 Exp 的“输入/依赖”；
2. 对应 `run_exp*.sh`；
3. 对应 `build_exp*.sh`；
4. 所需 checkpoint / OOF probability 是否已经生成。

不要把 `build_expXXX` 理解为从零训练所有依赖的万能脚本。

---

# 16. 提交文件与校验

最终压缩包：

```text
prediction.zip
├── A1.csv
└── A2.csv
```

A1：

```csv
test_idx,label
18,3
19,7
```

A2：

```csv
uid,prediction
u000001,"i000001,i000002,i000003,i000004,i000005,i000006,i000007,i000008,i000009,i000010"
```

项目提供：

```text
framework/code/validate_submission.py
```

建议任何候选在提交前执行完整校验，包括：

* 文件名；
* 列名；
* 行数；
* UID 顺序；
* test_idx；
* 合法 label；
* item 是否来自候选集；
* Top10 长度；
* item 去重；
* 缺失值。

提交前的原则：

```text
训练成功 ≠ 可以提交
本地指标高 ≠ 可以提交
必须先完成格式校验和 prediction drift audit
```

---

# 17. 实验记录与可复现性

`record.md` 是本项目最重要的文件之一。

每轮实验应至少记录：

```text
Experiment ID
Hypothesis
Baseline
Changed Variables
Training / Validation Setup
Metrics
Runtime
Generated Files
Online Feedback（如有）
Conclusion
Next Action
```

推荐保持：

```text
ExpXXX
```

连续编号，并确保：

```text
record 中写的配置
=
脚本实际运行的配置
=
保存 checkpoint 的配置
=
最终 candidate 的配置
```

Agent memory 与人工可读日志都应服务于同一目标：

> **让任何一个线上结果都能追溯到真实实验过程，而不是赛后补写。**

---

# 18. 合规说明

本仓库的推荐复现原则：

1. 仅使用比赛允许的数据与服务；
2. 不使用隐藏测试标签；
3. 不使用私有答案库；
4. 不根据样本 ID 硬编码结果；
5. 不将固定最优答案写入 Agent prompt；
6. 不通过排行榜差分或反复特殊提交反推测试答案；
7. API Secret 仅通过环境变量或 `.env.local` 管理；
8. 过程日志应真实反映实际运行；
9. 强方案可以被 Agent 通过实验逐步发现，但不应把完整最终预测器作为“答案”直接预埋到提示词中。

仓库历史中存在的 aggregate leaderboard calibration 类探索已经被视为废弃分支，不应作为正式比赛方案复现。

---

# 19. 后续方向

结合当前实验记录，后续最值得继续研究的方向包括：

### A1

* 更稳健的 OOF meta stacking；
* 在不泄漏标签的前提下融合 LP / neighbor statistics；
* 更强但低方差的专家 gating；
* snapshot / seed / configuration soup；
* 面向 A/B 数据迁移的自动配置选择。

### A2

* 专门针对 `len=0/1/2-3` 的候选生成；
* sparse short-history ranker；
* ordered suffix transition；
* user-profile × last-item 交互；
* empirical-Bayes / hierarchical prior；
* 在保持稳定 Top1 的同时扩大候选 recall；
* bucket-wise calibration，而不是全局统一模型权重；
* 更严格的 LOSO / cross-split stability selection。

### Agent

* 将实验收益、运行时和不确定性共同纳入 acquisition function；
* 自动识别“离线涨、线上跌”的分布偏移方向；
* 自动维护失败实验 blacklist；
* 自动进行 drift audit；
* 根据剩余预算动态切换探索 / 利用；
* 生成更加完整、可审计的 trajectory。

---

# 最后的说明

这个仓库记录的不是一条从一开始就知道答案的“最优模型脚本”，而是一条真实的实验演化路径：

```text
GraphSAGE
→ Label Propagation
→ Correct & Smooth
→ GAT ensemble
→ SIGN
→ SIGN + label features
→ multi-direction / full-label retrain
→ OOF meta stack

Popularity
→ History Co-occurrence
→ User Profile
→ Test-like Validation
→ Jaccard
→ Feature Ranker
→ Bucket Blend
→ EB / Gate / Protected Blend
→ Sparse Short-history Ranker
```

其中既包含成功方法，也保留了大量失败方向。

对于“稀疏反馈下的自动化实验”这类任务，失败记录本身也是 Agent 的训练数据：真正有价值的不是一次性得到某个配置，而是让系统知道 **为什么上一轮没有工作、下一轮应该尝试什么，以及什么时候应该停止继续搜索**。
::: 
