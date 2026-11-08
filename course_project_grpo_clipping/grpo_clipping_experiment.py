# ============================================================
# GRPO clipping study: ratio GRANULARITY (token vs sequence) x clip WIDTH (narrow vs wide)
# Task : GSM8K single-turn math (verifiable reward)
# Model: Qwen2.5-0.5B-Instruct + LoRA, single Colab GPU (L4 / A100 recommended; T4 works, slower)
#
# Usage:
#   python grpo_clipping_experiment.py --variant token_wide --seed 0 --max_steps 80
#   python grpo_clipping_experiment.py --list            # show the 4 arms
#
# Changes vs v0 (grpo_clipping_experiment_v0.py):
#   1. 2x2 design. v0's two arms changed BOTH granularity and epsilon_high at once
#      (confounded). Now granularity and width are separate factors.
#   2. Off-policy updates. With TRL's default num_iterations=1 the policy is updated
#      once per rollout batch -> old policy == current policy -> IS ratio == 1 and
#      clipping NEVER fires, so all arms would be identical. We default to
#      num_iterations=2 so each rollout batch is reused, making ratios != 1.
#   3. LR 1e-6 -> 1e-5. 1e-6 is a full-finetune LR; with LoRA it barely moves the
#      policy, so neither collapse nor learning would be visible in 100 steps.
#   4. Numeric answer matching ("18.0" == "18"), and all logs dumped to
#      <output_dir>/log_history.json for plot_pilot.py.
# ============================================================

import argparse
import inspect
import json
import os
import re

# ---------------- the 4 arms ----------------
# Sequence-level ratios are a geometric mean over tokens, so they sit MUCH closer
# to 1 than token ratios. A width of 0.2 at sequence level would basically never
# clip, so each granularity needs its own "narrow" scale. Narrow sequence bounds
# follow the GSPO paper / TRL docs (3e-4, 4e-4).
VARIANTS = {
    # baseline: standard GRPO / PPO clip
    "token_narrow": dict(importance_sampling_level="token", epsilon=0.2, epsilon_high=0.2),
    # tolerant arm (ARLArena Finding 1 predicts collapse)
    "token_wide": dict(importance_sampling_level="token", epsilon=0.2, epsilon_high=0.6),
    # GSPO-style stable arm
    "seq_narrow": dict(importance_sampling_level="sequence", epsilon=3e-4, epsilon_high=4e-4),
    # sequence-level but tolerant, separates granularity from width
    "seq_wide": dict(importance_sampling_level="sequence", epsilon=3e-3, epsilon_high=1e-2),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="token_wide", choices=list(VARIANTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--num_iterations", type=int, default=2,
                   help="policy updates per rollout batch; MUST be >1 for clipping to matter")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient to reference policy")
    p.add_argument("--num_generations", type=int, default=8, help="group size G")
    p.add_argument("--prompts_per_step", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=400)
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--eval_batch_size", type=int, default=32,
                   help="completions per eval batch (multiple of G); 8 made each eval ~20 min")
    p.add_argument("--output_root", default="runs")
    p.add_argument("--report_to", default="none", help="none | tensorboard | wandb")
    p.add_argument("--list", action="store_true")
    return p.parse_args()


# ==================== DATA ====================
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
# v1 used <think>/<answer> tags. The 0.5B model rarely followed them (~7% start accuracy)
# and the format reward was hacked: completions shrank to "<answer>N</answer>" with no
# reasoning. \boxed{} is Qwen2.5's native answer format, so we use it instead.


def extract_gold(text):
    # GSM8K gold answers end with "#### <number>"
    return text.split("####")[-1].strip().replace(",", "")


def build_prompt(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "gold": extract_gold(example["answer"]),
    }


# ==================== REWARDS ====================
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _text(comp):
    return comp[-1]["content"] if isinstance(comp, list) else comp


def _boxed_number(text):
    """Number inside the LAST \\boxed{...}, or None."""
    boxes = _BOXED_RE.findall(text)
    if not boxes:
        return None
    nums = _NUM_RE.findall(boxes[-1])
    return nums[-1] if nums else None


def _same_number(pred, gold):
    try:
        return abs(float(pred.replace(",", "").rstrip(".")) - float(gold)) < 1e-6
    except ValueError:
        return False


def correctness_reward(prompts, completions, gold, **kwargs):
    out = []
    for comp, g in zip(completions, gold):
        pred = _boxed_number(_text(comp))
        out.append(1.0 if (pred is not None and _same_number(pred, g)) else 0.0)
    return out


def format_reward(prompts, completions, **kwargs):
    # small shaping term (0.1) for producing a parsable \\boxed{number}
    return [0.1 if _boxed_number(_text(c)) is not None else 0.0 for c in completions]


# ==================== CONFIG (version-tolerant) ====================
def build_config(GRPOConfig, wanted):
    """Pass only the kwargs this TRL version supports; fail loudly on the critical ones."""
    supported = set(inspect.signature(GRPOConfig.__init__).parameters)
    critical = {"importance_sampling_level", "epsilon", "epsilon_high", "num_iterations"}
    missing = critical - supported
    if missing:
        raise RuntimeError(
            f"This TRL version lacks {sorted(missing)}. Install trl>=0.20 "
            "(importance_sampling_level was added in 0.20)."
        )
    dropped = sorted(k for k in wanted if k not in supported)
    if dropped:
        print(f"[warn] TRL does not support these args, skipped: {dropped}")
    return GRPOConfig(**{k: v for k, v in wanted.items() if k in supported})


def main():
    args = parse_args()
    if args.list:
        for k, v in VARIANTS.items():
            print(f"{k:13s} {v}")
        return

    import torch
    import trl
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    cfg = VARIANTS[args.variant]
    out_dir = os.path.join(args.output_root, f"{args.variant}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    train_ds = (load_dataset("openai/gsm8k", "main", split="train")
                .shuffle(seed=args.seed).map(build_prompt))
    eval_ds = (load_dataset("openai/gsm8k", "main", split="test")
               .select(range(args.n_eval)).map(build_prompt))

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    G = args.num_generations
    wanted = dict(
        output_dir=out_dir,
        seed=args.seed,
        # --- the experiment variables ---
        **cfg,
        num_iterations=args.num_iterations,
        # --- rollouts: each optimizer step sees prompts_per_step prompts x G completions ---
        num_generations=G,
        per_device_train_batch_size=G,
        gradient_accumulation_steps=args.prompts_per_step,
        per_device_eval_batch_size=args.eval_batch_size,
        max_prompt_length=384,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        # --- optimization, FIXED across arms ---
        learning_rate=args.lr,
        beta=args.beta,
        max_steps=args.max_steps,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        bf16=bf16,
        fp16=False,           # GRPO + fp16 on T4 is numerically fragile; T4 falls back to fp32
        gradient_checkpointing=True,
        report_to=args.report_to,
        log_completions=False,
    )
    targs = build_config(GRPOConfig, wanted)

    run_meta = {"trl_version": trl.__version__, "variant": args.variant, **vars(args),
                "grpo_config": {k: str(v) for k, v in targs.to_dict().items()}}
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[info] TRL {trl.__version__} | arm={args.variant} {cfg} | "
          f"num_iterations={args.num_iterations} | lr={args.lr} | bf16={bf16}")

    trainer = GRPOTrainer(
        model=args.model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[correctness_reward, format_reward],
        peft_config=lora,
    )

    try:
        trainer.train()
    finally:
        # always dump logs, even if the run crashes or is interrupted mid-way
        with open(os.path.join(out_dir, "log_history.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=1)
        print(f"[info] logs -> {out_dir}/log_history.json")


if __name__ == "__main__":
    main()
