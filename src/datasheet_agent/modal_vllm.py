"""vLLM inference servers on Modal, OpenAI-compatible.

Three servers, because the evaluation needs three distinct roles:

- **small** - Qwen2.5-1.5B-Instruct, the model the plan trains with GRPO. The system
  under test, and later the thing served with speculative decoding.
- **large** - Qwen2.5-32B-Instruct, the strong reference. Claude fills this role when
  an `ANTHROPIC_API_KEY` is available; this is the open stand-in when it is not.
- **verifier** - Phi-4, which checks extracted values against their evidence.

The verifier is deliberately *not* a Qwen model. A model asked to check its own work
tends to agree with it, so the verifier comes from a different family than either
agent it audits. Phi-4 is ungated and MIT-licensed, so it needs no license handshake.

Deploy:
    modal deploy -m datasheet_agent.modal_vllm

Only `small` and `large` enable tool calling; the verifier reads evidence and returns
a judgement, so it needs no tools.
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

SMALL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LARGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
VERIFIER_MODEL = "microsoft/phi-4"

VLLM_PORT = 8000
MINUTES = 60
MAX_MODEL_LEN = 32_768
# Phi-4 has max_position_embeddings=16384; asking for more fails at startup. The
# verifier only ever sees one field, one value and one evidence line, so this is ample.
VERIFIER_MAX_MODEL_LEN = 16_384

# A CUDA *devel* base, not `debian_slim`: vLLM 0.32 compiles kernels at engine start
# and dies with "Could not find nvcc" on a runtime-only image.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install("vllm==0.32.0", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
)

TRAINED_ADAPTER = "datasheet-grpo"
CHECKPOINTS = "/checkpoints"

hf_cache = modal.Volume.from_name("datasheet-hf-cache", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("datasheet-checkpoints", create_if_missing=True)
vllm_cache = modal.Volume.from_name("datasheet-vllm-cache", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache}

app = modal.App("datasheet-agent-vllm")


def serve_vllm(
    model: str,
    gpu_count: int = 1,
    tool_parser: str | None = None,
    max_model_len: int = MAX_MODEL_LEN,
    extra_args: list[str] | None = None,
) -> None:
    """Exec the vLLM OpenAI server so Modal's web proxy can reach it on VLLM_PORT."""
    command = [
        "vllm",
        "serve",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        model,
        "--enable-prefix-caching",
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(gpu_count),
    ]
    if tool_parser:
        command += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
    command += extra_args or []
    print("launching:", " ".join(command), flush=True)
    # Start and return. `@modal.web_server` proxies to the port once it is listening;
    # blocking here (e.g. on .wait()) leaves the container forever "starting", and
    # every request is answered with a 303 redirect loop instead of being proxied.
    subprocess.Popen(command)


@app.function(
    image=vllm_image,
    gpu="L40S",
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=60 * MINUTES,
    scaledown_window=10 * MINUTES,
    max_containers=4,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def small() -> None:
    """The 1.5B agent model. Serves a LoRA-merged checkpoint when SMALL_MODEL is set."""
    serve_vllm(os.environ.get("SMALL_MODEL", SMALL_MODEL), tool_parser="hermes")


@app.function(
    image=vllm_image,
    gpu="H100:2",
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=60 * MINUTES,
    scaledown_window=10 * MINUTES,
    max_containers=1,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def large() -> None:
    """The strong open reference model, standing in for Claude."""
    serve_vllm(os.environ.get("LARGE_MODEL", LARGE_MODEL), gpu_count=2, tool_parser="hermes")


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=60 * MINUTES,
    scaledown_window=10 * MINUTES,
    max_containers=1,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def verifier() -> None:
    """Phi-4: a different family from the agents it audits, so it can disagree."""
    serve_vllm(
        os.environ.get("VERIFIER_MODEL", VERIFIER_MODEL),
        max_model_len=VERIFIER_MAX_MODEL_LEN,
    )


@app.function(
    image=vllm_image,
    gpu="L40S",
    volumes={**VOLUMES, CHECKPOINTS: ckpt_volume},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=60 * MINUTES,
    scaledown_window=10 * MINUTES,
    max_containers=4,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def trained() -> None:
    """The GRPO-trained policy: the base model with the LoRA adapter attached.

    Served as an adapter rather than merged weights so the before/after comparison is
    exactly one variable -- same base, same server, same tool schemas, adapter on or
    off. Request it under the model name `datasheet-grpo`.
    """
    run_name = os.environ.get("RUN_NAME", "grpo-qwen1.5b-v4-nocost")
    adapter_path = f"{CHECKPOINTS}/{run_name}/final"
    serve_vllm(
        os.environ.get("SMALL_MODEL", SMALL_MODEL),
        tool_parser="hermes",
        extra_args=[
            "--enable-lora",
            "--max-lora-rank",
            "32",
            "--lora-modules",
            f"{TRAINED_ADAPTER}={adapter_path}",
        ],
    )


@app.function(
    image=vllm_image,
    gpu="L40S",
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=60 * MINUTES,
    scaledown_window=10 * MINUTES,
    max_containers=2,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def small_spec() -> None:
    """The small model with n-gram speculative decoding.

    N-gram speculation fits this task unusually well. The agent's output quotes the
    datasheet back verbatim -- the evidence field is a line copied out of a table it
    just read -- so the draft tokens are very often already present in the prompt.
    That is exactly the case prompt-lookup decoding is built for, and it needs no
    draft model, so it costs no extra GPU memory.
    """
    serve_vllm(
        os.environ.get("SMALL_MODEL", SMALL_MODEL),
        tool_parser="hermes",
        extra_args=[
            "--speculative-config",
            json.dumps(
                {
                    "method": "ngram",
                    "num_speculative_tokens": 5,
                    "prompt_lookup_max": 4,
                    "prompt_lookup_min": 2,
                }
            ),
        ],
    )
