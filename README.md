# Does Stable Agentic RL Scale Down?

**NYU NLP 课程项目 · Team 4（Yuxuan Feng、Yixuan Ma、Lilei Wu）**

在小模型上检验 ARLArena 关于 importance-sampling clipping 的结论：**tolerant token-level clipping 和 sequence-level clipping，哪种让 RL 训练更稳定？**

> 本分支基于 [ARLArena](https://github.com/WillDreamer/ARL-Arena)（Wang et al., ICML 2026）的 fork。我们的代码全部在 [`course_project_grpo_clipping/`](course_project_grpo_clipping/) 文件夹里，该文件夹的 README 记录了完整的实验细节和运行方法。最初的论文分析和项目规划见 [`docs/project_planning.md`](course_project_grpo_clipping/docs/project_planning.md)。仓库其余部分是 ARLArena 的原始代码，未做修改；原项目说明见[上游仓库](https://github.com/WillDreamer/ARL-Arena)。

---

## 一、项目要做什么

### 背景

用强化学习（RL）训练大语言模型做推理或 agent 任务，训练经常会**突然崩溃**：准确率先涨后断崖式下跌，KL 和梯度范数同时爆炸。ARLArena 把策略梯度方法拆成 4 个设计维度逐一研究，发现 **importance-sampling（IS）clipping 是对稳定性影响最大的维度**（Finding 1）：

> tolerant clipping yields fast early gains but causes training collapse, whereas sequence-level clipping ensures stable improvement.

但这个结论是在 4B–8B 模型、H200/B200 GPU 上得到的，大多数学生和小实验室复现不了。

### 研究问题

ARLArena 的 Finding 1 在 **0.5B 模型 + LoRA + 单张 GPU** 的条件下还成立吗？

### 对比与假设

| | (a) Tolerant token-level clipping | (b) Sequence-level clipping（GSPO） |
|---|---|---|
| 做法 | 每个 token 一个 ratio，clip 上界放宽（ε_high = 0.6） | 每个回答一个 ratio（几何平均），整体 clip 一次 |
| 预测 | 前期提升更快，之后崩溃 | 稳定提升 |

**Hypothesis**：tolerant clipping will produce faster early improvement followed by collapse, while sequence-level clipping will be more stable.

### 方法与评估

- **方法**：用 Hugging Face TRL 的 GRPOTrainer，在 GSM8K 上训练 Qwen2.5-0.5B-Instruct + LoRA。两组之间**只有 clipping 不同**，其余全部固定。
- **评估**：held-out 准确率随训练步数的曲线；稳定性指标包括 KL、梯度范数、被 clip 的比例、格式正确率。"崩溃"定义为准确率从峰值显著下降，同时 KL 或梯度范数出现尖峰。

---

## 二、已经完成的

| 状态 | 内容 |
|---|---|
| ✅ | **训练 pipeline**：基于 TRL GRPOTrainer + LoRA，支持切换 clipping 粒度（token / sequence）和宽度（ε） |
| ✅ | **NYU Torch 集群环境**：Python 环境、依赖版本固定、模型和数据预下载、Slurm 提交脚本 |
| ✅ | **两轮冒烟测试**：修复环境问题；确认需要 `num_iterations = 2`，否则 ratio 恒为 1、clipping 永远不触发 |
| ✅ | **v1 pilot（4 组 × 150 步）**：发现 **reward hacking**，模型不再推理、只输出答案标签。改用 Qwen 原生的 `\boxed{}` 格式后解决 |
| ✅ | **v2 pilot（4 组 × 300 步，单个 seed）**：训练正常，结果见下 |
| ✅ | 代码、配置和逐步日志已上传到本仓库 |
| ✅ | Midterm presentation |

### v2 pilot 的主要发现

| | (a) tolerant（token_wide） | (b) sequence-level（seq_narrow） |
|---|---|---|
| 测试准确率（第 50 步 → 第 300 步） | 35.9% → 39.1% | 33.1% → 36.9% |
| 被 clip 的比例 | 0.02% | 33.3% |
| 最终 KL | 0.0032 | 0.0022 |
| 是否崩溃 | 否 | 否 |

- **Pipeline 可行**：单张 L40S 每组约 1.1–1.4 小时。
- **两组都没有崩溃**，准确率差距（约 3 个点）在噪声范围内。Hypothesis 目前**既未被支持，也未被推翻**。
- **tolerant 组的 clipping 几乎从不触发**：当前设置下模型每步变化很小（KL 约 0.003），放宽的上界根本用不到。
- **要真正检验 hypothesis，必须提高 off-policy 程度**，这是下一步的重点。

详细结果和训练曲线见 [`course_project_grpo_clipping/README.md`](course_project_grpo_clipping/README.md)。

---

## 三、未来要做的

| 优先级 | 任务 | 目的 |
|---|---|---|
| 1 | **压力测试**：`num_iterations` 提高到 4 / 8，或 lr 提高到 5e-5 | 找到 clipping 真正起作用的设置，看 tolerant 组是否开始崩溃 |
| 2 | **多 seed**：每组跑 3 个 seed，报告均值 ± 标准差 | 区分真实差异和随机噪声（proposal 要求） |
| 3 | **崩溃机制诊断**：按 advantage 正负，统计越界 IS ratio 的比例；尝试 sequence masking | 检验论文的说法：崩溃由"负 advantage + 低 ratio"的序列累积造成 |
| 4 | **加入 CISPO / SAPO 式的 tolerant 变体**：参考本仓库 `verl/trainer/ppo/core_algos.py` 的官方实现 | ARLArena 原文的 tolerant 指 CISPO/SAPO（保留越界 token 的梯度），与我们放宽 ε 的做法不同 |
| 5 | **模型规模**：在 Qwen2.5-1.5B 上重复关键实验 | 检验效应是否随规模增强（proposal 中的 robustness 部分） |
| 6 | *Stretch*：极小的 Sokoban 多轮环境 | 从单轮数学任务扩展到真正的多轮 agent 场景 |
| 7 | Final report | 汇总所有结果 |

### 已知局限

- 目前每组只有 1 个 seed。
- 约 15% 的回答达到 400 token 上限被截断（两组比例相同）。
- 300 步只用了约 8% 的训练集。
- GSM8K 是单轮任务，不是真正的多轮 agent 场景。

---

## 引用

本项目复现和检验的原始工作：

```bibtex
@article{wang2026arlarena,
  title={ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning},
  author={Wang, Xiaoxuan and Zhang, Han and Wang, Haixin and others},
  journal={arXiv preprint arXiv:2602.21534},
  year={2026}
}
```

ARLArena 代码遵循其原始 [LICENSE](LICENSE)。
