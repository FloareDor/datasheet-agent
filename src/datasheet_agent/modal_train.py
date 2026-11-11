"""GRPO training on Modal.

Trains Qwen2.5-1.5B to drive the three datasheet tools, using TRL's
`environment_factory` multi-turn rollouts with vLLM colocated on the training GPUs.

Run detached, because a training run outlives any terminal:
    modal run --detach -m datasheet_agent.modal_train

The datasheets live on a Modal Volume rather than in the image: 300 training PDFs are
far too large to bake in, and a volume lets the splits change without a rebuild.
Upload them first with `scripts/upload_training_data.py`.

One training row per (part, field group). Grouping gives three shorter episodes per
datasheet instead of one long one, which is both cheaper to roll out and closer to
what the deployed multi-agent pipeline actually runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

MINUTES = 60
HOURS = 60 * MINUTES

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/checkpoints")

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "torch",
        "transformers>=5.2.0",
        "trl==1.17.1",
        "peft",
        "accelerate",
        "datasets",
        "vllm==0.32.0",
        "wandb",
        "pymupdf>=2.1",
        "datasheetindex @ git+https://github.com/Infineon/datasheetindex",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_python_source("datasheet_agent")
)

data_volume = modal.Volume.from_name("datasheet-training-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("datasheet-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("datasheet-hf-cache", create_if_missing=True)

app = modal.App("datasheet-agent-train")


def _build_dataset(split: str) -> object:
    """One row per (part, field group), with the labels that group is graded on."""
    import json

    from datasets import Dataset

    from datasheet_agent.extract import SYSTEM
    from datasheet_agent.schema import GROUPS, fields_in_group

    rows: list[dict[str, object]] = []
    path = DATA_DIR / "splits" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            pdf_path = DATA_DIR / "pdfs" / Path(record["pdf_path"]).name
            if not pdf_path.exists():
                continue
            for group in GROUPS:
                labels = {
                    spec.name: record["labels"].get(spec.name, "")
                    for spec in fields_in_group(group)
                }
                if not any(labels.values()):
                    continue  # nothing to grade: this episode could only teach noise
                rows.append(
                    {
                        # A conversational seed. TRL indexes `prompt[-1]["content"]`,
                        # so the column cannot be empty even though the environment's
                        # reset() appends the real task text to this user turn.
                        "prompt": [
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": (
                                    f"Extract the {group} specs for part "
                                    f"{record['part_number']} from its datasheet."
                                ),
                            },
                        ],
                        "part_number": record["part_number"],
                        "pdf_path": str(pdf_path),
                        "group": group,
                        "labels": labels,
                    }
                )
    return Dataset.from_list(rows)


@app.function(
    image=train_image,
    gpu="H100:2",
    volumes={
        str(DATA_DIR): data_volume,
        str(OUTPUT_DIR): ckpt_volume,
        "/root/.cache/huggingface": hf_cache,
    },
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("wandb-api-key"),
    ],
    timeout=8 * HOURS,
)
def train(
    steps: int = 300,
    num_generations: int = 8,
    learning_rate: float = 5e-6,
    max_tool_iterations: int = 8,
    run_name: str = "grpo-qwen1.5b",
    tool_call_penalty: float | None = None,
) -> dict[str, object]:
    """Run GRPO and save the trained adapter to the checkpoints volume."""
    import functools

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from datasheet_agent.rl_env import DatasheetEnv, extraction_reward

    os.environ.setdefault("WANDB_PROJECT", "datasheet-agent")
    if tool_call_penalty is not None:
        os.environ["TOOL_CALL_PENALTY"] = str(tool_call_penalty)

    dataset = _build_dataset("train")
    print(f"training rows (part x field group): {len(dataset)}", flush=True)

    output_dir = OUTPUT_DIR / run_name
    config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=steps,
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        max_completion_length=1024,
        # Colocated vLLM: rollouts and training share the two H100s. Server mode
        # would need a third GPU doing nothing while the trainer steps.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        max_tool_calling_iterations=max_tool_iterations,
        logging_steps=1,
        save_steps=50,
        report_to="wandb",
        run_name=run_name,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=BASE_MODEL,
        args=config,
        train_dataset=dataset,
        reward_funcs=extraction_reward,
        # The cache lives on the data volume so an index built in one episode is
        # reused by every later episode on the same datasheet.
        environment_factory=functools.partial(DatasheetEnv, cache_dir=DATA_DIR / "cache"),
        peft_config=LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()

    final = output_dir / "final"
    trainer.save_model(str(final))
    ckpt_volume.commit()
    print(f"saved adapter to {final}", flush=True)
    return {"output_dir": str(final), "steps": steps, "rows": len(dataset)}


@app.local_entrypoint()
def main(
    steps: int = 400,
    run_name: str = "grpo-qwen1.5b",
    learning_rate: float = 5e-6,
    wait: bool = False,
) -> None:
    """Launch training.

    Spawned, not called. `.remote()` keeps the local process attached for the whole
    run, and `--detach` did not save it: when the launching shell was killed, the
    training job was cancelled with it at step ~250 of 400. `.spawn()` returns a
    handle immediately and the job runs on Modal independently of anything local.
    """
    handle = train.spawn(steps=steps, learning_rate=learning_rate, run_name=run_name)
    print(f"spawned {run_name}: call id {handle.object_id}")
    print("follow with: modal app logs <app-id>, or the W&B run")
    if wait:
        print(handle.get())
