> **说明**：这是项目最初的规划文档（2026 年 9 月，写 proposal 之前），保留在这里供参考。部分内容已过时：
> - 实际实验在 **NYU Torch 集群（单张 L40S）** 上运行，不是 Colab；
> - 第五节里的脚本示例已被 [`grpo_clipping_experiment.py`](../grpo_clipping_experiment.py) 取代；
> - 当前进度和结果见[分支首页 README](../../README.md)，实验细节见 [`course_project_grpo_clipping/README.md`](../README.md)。

---

# ARLArena 论文分析与课程项目规划

> 论文:*ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning*
> (Wang et al., ICML 2026, arXiv:2602.21534v3)
> 代码:https://github.com/WillDreamer/ARL-Arena.git

本文档分析该论文的研究方向、四维分解框架、创新点与 literature review 结构,并给出一个可落地的 NLP 课程项目规划(方向:稳定性分析 / 消融复现;算力:Colab 级)。

---

## 一、研究方向 (What it studies)

**核心主题:Agentic Reinforcement Learning (ARL) 的训练稳定性问题。**

背景链条:

1. 大模型(LLM)现在被越来越多地当作 **自主 agent** 来做多步交互任务(网页导航、具身环境、游戏、深度检索、工具调用)。
2. 这类任务需要 **多轮 (multi-turn)** 训练目标,而不是单轮问答。用 RL 做后训练(post-training)是主流手段(DeepSeek-R1、OpenAI o1 都是例子)。
3. **但 agentic RL(ARL)训练极不稳定,经常直接"崩溃"(training collapse)。** 原因是多轮交互带来的复合难题:无效动作、稀疏奖励、长程信用分配(credit assignment)、以及 agent-环境的非平稳性。早期一个小偏差会跨轮次级联放大,导致分布漂移、退化的 rollout。

**这篇论文要解决的问题**:让 ARL 训练变得稳定、可复现、可扩展。它不是提出一个"更强的模型",而是提出一个**分析框架 + 稳定训练配方 + 新算法**。

具体三件事:

- **ARLArena**:一个标准化、可复现的多轮 ARL 测试平台(testbed)+ 系统性分析方法论。
- 把 policy gradient(策略梯度)**分解成 4 个正交的设计维度**,逐一隔离分析每个维度对稳定性和性能的影响。
- **SAMPO**(Stable Agentic Multi-turn Policy Optimization):基于分析得出的一套统一的策略优化方法,直接针对分析中发现的主要不稳定来源。

---

## 二、四个分解维度(论文的分析骨架)

这是全文最核心的组织方式——把所有主流 RL 算法(GRPO、DAPO、GSPO、CISPO、SAPO、GiGPO、EMPG 等)沿着**同一套 4 个维度**对齐比较(见论文 Table 1):

| 维度 | 研究什么 | 关键发现 |
|---|---|---|
| **1. Loss Aggregation**(损失聚合) | token-mean vs. seq-mean-token-mean | 序列级聚合在长度差异大的任务(如数学)上会因 token 权重不均而损害训练 |
| **2. Importance Sampling (IS) Clipping**(重要性采样裁剪) | 容忍式裁剪 vs. 序列级裁剪 | **最敏感的维度**。宽松裁剪初期涨得快但会崩溃;序列级裁剪保证稳定提升(Finding 1) |
| **3. Trajectory Filtering / Resampling**(轨迹过滤/动态采样) | 过滤掉零梯度轨迹 | 与 GiGPO 结合比与 GRPO 结合更稳定(Finding 4) |
| **4. Advantage Design**(优势函数设计) | 引入环境级/全局-局部信息 | 引入细粒度环境优势能缓解奖励稀疏、提升性能(Finding 3) |

三个核心结论(论文摘要点明):

1. 容忍式裁剪 → 崩溃;序列级裁剪 → 稳定。
2. 把环境级信息加入 advantage 设计 → 稳定性和性能都提升。
3. 动态采样 + 细粒度 advantage 设计结合 → 进一步受益。

崩溃的**根因诊断**很值得学习:他们发现训练崩溃主要是由 **"负优势 + 低重要性比"的序列不断累积** 驱动的,并用 token 级/序列级 IS 分析(Figure 3 的 streamgraph)可视化验证。解决方法是 **sequence masking**(把这类有害序列屏蔽掉)。

---

## 三、创新点 (Novelty)

论文自己列的四条贡献:

1. **统一视角**:把策略梯度和 PO 方法沿 4 个维度做统一分类——这是"综述式重构 + 实验对齐",让原本各说各话的方法可比较。
2. **标准化、可复现的 testbed** 和多轮 ARL 稳定性诊断方法论(ARLArena)。
3. **任务鲁棒的发现和补救措施**——针对常见崩溃模式给出诊断和修复(比如 sequence masking)。
4. **SAMPO 新算法**:把序列级裁剪 + 细粒度 advantage(全局+局部)+ 动态过滤三者**统一到一个框架**,比 GRPO 基线平均提升 25.2%,并在 ALFWorld / WebShop / Sokoban / TIR Math 上做到稳定训练。

一个额外有意思的实验(Section 5.3):用 SAMPO 训练的小开源模型(Qwen3-4B)能在 agentic 任务上**超过 GPT-5.2 和 o3 的多 agent 系统**——论点是"在 agentic 任务上,规范的 RL 训练比推理时的复杂工程更有价值"。

**创新的方法论特征**(对写课程项目很有借鉴意义):

- 不是发明全新组件,而是**"解耦—隔离—重组"**:把已有方法拆成正交维度,逐维消融(ablation),再把每维最优的选择重组成新方法。这是一种非常稳、非常"可发表"的科研套路。
- 强调 **可复现性和公平比较**(标准化 testbed + 每个方法都做超参网格搜索)。

---

## 四、Literature Review 结构

论文的 related work 不是独立一节,而是**融进 Introduction 和 Table 1** 里(ML 会议常见写法)。引用的文献可分成几类:

- **RL post-training 基础**:PPO (Schulman 2017)、InstructGPT/RLHF (Ouyang 2022)、KL 近似 (Schulman 2017 blog)。
- **推理 RL**:DeepSeek-R1、OpenAI o1。
- **PO 算法族(核心对比对象)**:GRPO、DAPO (Yu 2025)、GSPO (Zheng 2025)、CISPO (Chen 2025)、SAPO (Gao 2025)、GiGPO (Feng 2025)、EMPG (Wang 2025)。
- **Agentic RL 框架/环境**:AgentGym、Agent-R1、AgentRL、RAGEN、SWEET-RL、verl(训练框架)。
- **Agent 环境/benchmark**:ALFWorld、WebShop、WebArena、Sokoban、TIR Math、ReAct、Voyager、API-Bank。
- **稳定性相关新工作**:CLEANER(自净化轨迹)、TIPS(turn-level reward shaping)、T²PO、Latent Poincaré shaping 等 2026 年的工作。

---

## 五、NLP 课程项目规划

**方向:稳定性分析 / 消融复现。算力:Colab 级(少量 GPU)。**

### 项目 scope:复现并验证一个 Finding

不要碰整个 ARLArena(4 环境 + Qwen3-4B/8B + H200,Colab 根本跑不动)。**聚焦复现论文 4 个维度中的 1 个,验证它的一条 Finding**,这就足够撑起一个课程项目。

**最推荐:复现 Finding 1(IS clipping 维度)—— "宽松裁剪导致崩溃 vs 序列级裁剪稳定"**

- 这是论文中**最敏感、信号最强**的维度,小规模也能看到明显的崩溃/稳定对比,画出来的曲线很有说服力。
- 崩溃现象(reward 突然掉、KL/gradient norm 爆炸)在小模型上一样会出现,容易观察。

**次选:Loss Aggregation 维度 —— token-mean vs seq-mean-token-mean**

- 实现最简单(就是 loss 归一化方式的差别),但信号弱一些,需要长度差异大的任务才明显。

### 具体可行的技术栈(Colab 级)

| 组件 | 建议 |
|---|---|
| 模型 | Qwen2.5-0.5B 或 1.5B-Instruct + **LoRA**(全参数训不动) |
| 框架 | **TRL** 的 `GRPOTrainer`(比 verl 轻,Colab 友好),或 verl 单卡模式 |
| 任务 | 二选一:①**GSM8K / 小数学题**(单轮、奖励好定义,先跑通 pipeline);②**极小 Sokoban**(3×3~4×4 网格,多轮但状态空间小) |
| 变量 | 只改 clipping 策略:标准 PPO clip(宽松)vs 序列级 clip(GSPO 式),其余固定 |
| 指标 | success rate / reward、off-policy KL、gradient norm、out-of-bounds token 比例——**照着论文 Figure 2 的 5 个 panel 画** |

先用数学任务跑通整条 RL pipeline(最省事),再决定要不要上多轮 agent 环境。多轮环境即使用最小 Sokoban 也会明显更难调。

### 项目叙事(报告怎么写)

1. **Related work / lit review**:直接用论文 Table 1 的框架——把 GRPO/DAPO/GSPO/CISPO 沿 4 维度对齐介绍(你只深入其中 1 维)。
2. **方法**:形式化你研究的那个维度,给出你对比的两个变体。
3. **实验**:小模型 + 小任务,画训练动态曲线,复现崩溃 vs 稳定。
4. **分析(你的"创新点"来源)**:课程项目不需要新算法。贡献可以是:
   - **"在更小规模上验证论文 Finding 是否成立"**(scale-down 复现本身就有价值);
   - 或提出一个小观察 / 小变体(比如换个 masking 阈值、换个任务看结论是否迁移)。

### 一个诚实的提醒

多轮 agentic RL 即使小规模,在 Colab 上也**很容易 OOM 或跑很久**(rollout 慢、环境交互开销大)。因此强烈建议:**先用单轮数学任务把"崩溃 vs 稳定"的对比做出来**,把它作为项目的稳妥核心;多轮 agent 环境作为 stretch goal,时间够再加。这样不会因为环境搭不起来而整个项目卡死。

---

## 六、Proposal 提交要求 (Assignment Instructions)

Please submit your Project Proposal for this assignment. Your proposal should present an initial plan for your project (which you may revise as your work progresses) and address the aspects outlined below.

**Writing Template:**
Please use the ACL conference template for the project submissions: https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj

**Project Proposal Outline:**

1. **Why Important? (Motivation):**
   Explain why this problem matters. What real-world or academic gap does it address?

2. **Related Work:**
   Briefly summarize the current state of the field and key prior work. Identify gaps or limitations you intend to address.

3. **Research Question or Goals:**
   State your central research question or hypothesis. Or list your main objectives. What do you aim to achieve?

4. **Methodology:**
   Describe your planned methods, models, datasets, or frameworks.

5. **Evaluation:**
   Explain how you will evaluate your approach (metrics, experiments, qualitative analysis, etc.).

6. **Tangible Outcomes / Contributions:**
   Outline your expected contributions (e.g., new algorithm, dataset, analysis, system, model, theoretical insight).
