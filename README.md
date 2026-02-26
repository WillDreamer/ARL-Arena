<!-- <div align="center">
 👋 Hi, everyone! 
    We will tell the world open-source agentic RL area is suffering!
    <br>
    <br>
</div> -->


<h1 style="text-align: center;"> 🤖 ARLArena: A Unified Framework for Stable
Agentic Reinforcement Learning </h1>

<div align="center">

<a href="https://huggingface.co/papers/2602.21534"><img src="https://img.shields.io/static/v1?label=Agentic RL&message=Paper&color=red"></a>
<a href="https://huggingface.co/UCLA-SCAI/models"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Models&color=yellow"></a>
<img src="https://img.shields.io/badge/Num_Tasks-16-blue">
[![GitHub Repo stars](https://img.shields.io/github/stars/WillDreamer/ARL-Arena)](https://github.com/WillDreamer/ARL-Arena/stargazers)


</div>


## Abstract
Agentic reinforcement learning (ARL) has rapidly gained attention as a promising paradigm for training agents to solve complex, multi-step interactive tasks.
Despite encouraging early results, ARL remains highly unstable, often leading to training collapse. This instability limits scalability to larger environments and longer interaction horizons, and constrains systematic exploration of algorithmic design choices.
In this paper, we first propose ARLArena, a stable training recipe and systematic analysis framework that examines training stability in a controlled and reproducible setting. ARLArena first constructs a clean and standardized testbed. Then, we decompose policy gradient into four core design dimensions and assess the performance and stability of each dimension. Through this fine-grained analysis, we distill a unified perspective on ARL and propose SAMPO, a stable agentic policy optimization method designed to mitigate the dominant sources of instability in ARL.
Empirically, SAMPO achieves consistently stable training and strong performance across diverse agentic tasks. Overall, this study provides a unifying policy gradient perspective for ARL and offers practical guidance for building stable and reproducible LLM-based agent training pipelines.

<p align="center"><img src="public/arlarena_overview.png" width="800px" alt="ARLArena Framework" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
Figure 1: Overview of our framework
</p>

## Our Framework Design

- ARLArena profoundly analyzes the existing sufferings of Agentic RL from the perspective of policy gradient.
- ARLArena comprehensively compares existing Agentic RL algorithms and provides a systematic discussion and analysis across multiple dimensions.
- ARLArena universally provides experimental results and findings on multiple agentic tasks.

### 🔥 Key Features

- ✅ Support Training Multi-turn Math+Code Interpreter Agents
- ✅ Support Training Multi-turn Embody Agents
- ✅ Support Training Multi-turn Multi-modal Game Agents
- ✅ Support Training Multi-turn Web Agents
- ✅ Support Training Multi-turn Search Agents

### 🔧 Upcoming Features and Changes

- ➡️ Support Software Enginnering Agents

### 📅 TODO

- [ ] Cross-domain agentic reasoning
- [ ] Multiple tool integration reasoning

## 💡 Getting Started


ARLArena is based on the following main dependencies:

```python
Python=3.11, VeRL=0.4.0, PyTorch=2.6.0, and vLLM=0.8.5
```

## 🚀 Existing Support

> ### 🛒 **Web Agents**


```bash
# 1. Build the webshop environments
bash prepare_all_web.sh

# 2. Run the demo code with:
conda activate agentrl_web.sh
bash examples/shop_agent_trainer/train_xxxx.sh
```

> ### 🤖 **Embodied Agents**
```bash
# 1. Build the environments
bash prepare_all_embody.sh

# 2. Run the demo code with:
bash examples/world_agent_trainer/train_xxx.sh

```

> ### 🧮 **Math+CI**

1. We use Sandbox Fusion as an asynchronous code interpreter. You can follow the [Guidance](sandbox/README.md) to run the CI.

2. The training datasets are Math3-5 from SimpleRL and Deepscaler in `datasets`.

```bash
# 3. Install the requirements
bash prepare_all_science.sh

# 4. Run the demo code with:
conda activate agentrl_science
bash examples/simpletir_trainer/train_grpo.sh
```

> ### 🎮 **OpenAI Game Agents**

```bash
# 1. Install the requirements
bash prepare_all_game.sh

# 2. Run the demo code with:
bash examples/game_agent_trainer/train_xxx.sh
```

> ### 🕸️ **Search Agents**

```bash
#! 1. Build the RAG server environments
bash prepare_all_search.sh


# 2. Run the demo code with:
bash examples/search_agent_trainer/train_xxx.sh
```

## 🌊 Easy Extension

🔹 All of the methods utilized is in `recipe`, you can warp the verl worker for your code to join our codebase. The folder under `recipe` can represent either a method for different tasks or a series methods for one task. You can refer to [Easy Extension](docs/extension.md) for examples.

🔹 All of the environments utilized is in `agent_system`, you can warp the env for your code to join our codebase.

🔹 Add specific dependencies to `requirements_xxx.txt`

🔹 Feel free to add the folder of the third-party tools, e.g., `sandbox` for code implementation.


## 📊 Further Details

<p align="center"><img src="public/sum.png" width="900px" alt="table" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
Figure 2: A summary of policy optimization methods studied in ARLArena.
</p>

<p align="center"><img src="public/table.png" width="900px" alt="table" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
Figure 3: Performance comparison of policy optimization methods across four agentic tasks, evaluated on the SFT version of Qwen3-4B.
</p>


<p align="center"><img src="public/combined_success_rate.png" width="900px" alt="success" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
Figure 4: Training curves on ALFWorld (left) and Sokoban (right). 
</p>


## ✍️ Citation
```bibtex
@misc{wang2026arlarenaunifiedframeworkstable,
      title={ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning}, 
      author={Xiaoxuan Wang and Han Zhang and Haixin Wang and Yidan Shi and Ruoyan Li and Kaiqiao Han and Chenyi Tong and Haoran Deng and Renliang Sun and Alexander Taylor and Yanqiao Zhu and Jason Cong and Yizhou Sun and Wei Wang},
      year={2026},
      eprint={2602.21534},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2602.21534}, 
}
```


<!-- 

```bash
# Install requiremnet
pip install mlflow

# Start server
mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:////tmp/mlruns.db \
  --default-artifact-root /tmp/mlruns

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

actor_rollout_ref.rollout.trace.backend: mlflow  # or weave
actor_rollout_ref.rollout.trace.token2text: True
trainer.logger: ['console', 'mlflow']
``` -->

## 🎆 Awesome work for reference

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of **DeepSeek R1 Zero** recipe for reasoning tasks ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
- [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team. ![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/simpleRL-reason)
- [Easy-R1](https://github.com/hiyouga/EasyR1): **Multi-modal** RL training framework ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tunning framework for multiple agent environments. ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
- [rllm](https://github.com/agentica-project/rllm): async RL training with [verl-pipeline](https://github.com/agentica-project/verl-pipeline) ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/rllm)
- [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning **agent** training framework ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
- [Logic-RL](https://github.com/Unakar/Logic-RL): a reproduction of DeepSeek R1 Zero on 2K Tiny Logic Puzzle Dataset. ![GitHub Repo stars](https://img.shields.io/github/stars/Unakar/Logic-RL)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and **searching (tool-call)** interleaved LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): RL Training of **Search Agent** with **Search/Retrieval Outcome** ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
- [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to **Re**ason with **Search** for LLMs via Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
- [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for **Code** with Reliable Rewards ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
- [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1): Skywork open reaonser series ![GitHub Repo stars](https://img.shields.io/github/stars/SkyworkAI/Skywork-OR1)
- [ToRL](https://github.com/GAIR-NLP/ToRL): Scaling tool-integrated RL ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/ToRL)
- [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for **long-horizon LLM/VLM agents**, along with a new algorithm **GiGPO** ![GitHub Repo stars](https://img.shields.io/github/stars/langfengQ/verl-agent)
- [PF-PPO](https://arxiv.org/abs/2409.06957): Policy Filtration for PPO based on the reliability of reward signals for more efficient and robust RLHF.
- [GUI-R1](https://github.com/ritzz-ai/GUI-R1): **GUI-R1**: A Generalist R1-style Vision-Language Action Model For **GUI Agents** ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)
- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)
- [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN)
- [ReTool](https://retool-rl.github.io/): ReTool: reinforcement learning for strategic tool use in LLMs
- [Seed-Coder](https://github.com/ByteDance-Seed/Seed-Coder): RL training of Seed-Coder boosts performance on competitive programming ![GitHub Repo stars](https://img.shields.io/github/stars/ByteDance-Seed/Seed-Coder)
- [all-hands/openhands-lm-32b-v0.1](https://www.all-hands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model): A strong, open coding agent model, trained with [multi-turn fine-tuning](https://github.com/volcengine/verl/pull/195)
- [RM-R1](https://arxiv.org/abs/2505.02387): RL training of reasoning reward models ![GitHub Repo stars](https://img.shields.io/github/stars/RM-R1-UIUC/RM-R1)
- [Absolute Zero Reasoner](https://arxiv.org/abs/2505.03335): A no human curated data self-play framework for reasoning![GitHub Repo stars](https://img.shields.io/github/stars/LeapLabTHU/Absolute-Zero-Reasoner)
- [LUFFY](https://arxiv.org/pdf/2504.14945): Learning to Reason under Off-Policy Guidance![GitHub Repo stars](https://img.shields.io/github/stars/ElliottYan/LUFFY)
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool): An unified and easy-to-extend tool-agent training framework based on verl![GitHub Repo stars](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)
- [DeepMath](https://github.com/zwhe99/DeepMath): DeepMath-103K data and series models for math reasoning![GitHub Repo stars](https://img.shields.io/github/stars/zwhe99/DeepMath)
