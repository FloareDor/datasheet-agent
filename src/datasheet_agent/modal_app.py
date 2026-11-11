"""The demo service: upload a datasheet, get a spec sheet back.

A FastAPI app on Modal that runs the same pipeline the evaluation measures -- the
parallel field agents, the code-and-model verifier, one retry -- against the vLLM
server deployed by `modal_vllm.py`. Nothing here is demo-only: a number shown on the
page is a number the eval would have scored.

Deploy:
    modal deploy -m datasheet_agent.modal_app
"""

# Two deliberate choices about annotations in this module.
#
# No `from __future__ import annotations`: it turns FastAPI's parameter hints into
# strings, and Pydantic then cannot resolve them.
#
# FastAPI is imported at module level, not inside `web()`: FastAPI resolves a route's
# annotations against its function's *module globals*, so `UploadFile` imported into a
# local scope is invisible to it and every request fails with a ForwardRef error.
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

MINUTES = 60
SMALL_URL = "https://sairavig--datasheet-agent-vllm-small.us-east.modal.run/v1"
VERIFIER_URL = "https://sairavig--datasheet-agent-vllm-verifier.us-east.modal.run/v1"

image = (
    modal.Image.debian_slim(python_version="3.13")
    # datasheetindex installs from git, which needs git in the image.
    .apt_install("git")
    .pip_install(
        "fastapi[standard]",
        "openai",
        "anthropic",
        "pydantic>=2.7",
        "langgraph>=0.2.40",
        "rapidfuzz",
        "pymupdf>=2.1",
        "datasheetindex @ git+https://github.com/Infineon/datasheetindex",
    )
    .add_local_python_source("datasheet_agent")
    .add_local_dir("web", remote_path="/root/web")
)

cache_volume = modal.Volume.from_name("datasheet-artifact-cache", create_if_missing=True)
CACHE_DIR = Path("/cache")

app = modal.App("datasheet-agent-demo")


@app.function(
    image=image,
    volumes={str(CACHE_DIR): cache_volume},
    timeout=15 * MINUTES,
    max_containers=4,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web() -> Any:
    from datasheet_agent.llm import OpenAIBackend
    from datasheet_agent.modal_vllm import SMALL_MODEL, VERIFIER_MODEL
    from datasheet_agent.pipeline import PipelineConfig, extract_multi_agent

    api = FastAPI(title="DatasheetAgent", docs_url="/docs")

    def build_config(use_verifier: bool) -> PipelineConfig:
        return PipelineConfig(
            backend=OpenAIBackend(
                model=os.environ.get("SMALL_MODEL", SMALL_MODEL),
                base_url=SMALL_URL,
                api_key="modal",
            ),
            cache_dir=CACHE_DIR,
            verifier_backend=(
                OpenAIBackend(model=VERIFIER_MODEL, base_url=VERIFIER_URL, api_key="modal")
                if use_verifier
                else None
            ),
            use_verifier=use_verifier,
        )

    @api.get("/")
    def index() -> FileResponse:
        return FileResponse("/root/web/index.html")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/extract")
    async def extract(
        file: UploadFile = File(...),  # noqa: B008 - FastAPI dependency style
        part_number: str = Form(...),  # noqa: B008
        verifier: bool = Form(default=True),  # noqa: B008
    ) -> JSONResponse:
        """Extract a spec sheet from an uploaded datasheet PDF."""
        if not part_number.strip():
            raise HTTPException(status_code=400, detail="part_number is required")
        contents = await file.read()
        if not contents.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="that file is not a PDF")

        workdir = Path(tempfile.mkdtemp(prefix="datasheet-"))
        pdf_path = workdir / "datasheet.pdf"
        try:
            pdf_path.write_bytes(contents)
            result = extract_multi_agent(build_config(verifier), part_number.strip(), pdf_path)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return JSONResponse(
            {
                "part_number": part_number.strip(),
                "values": result.extraction.values,
                "evidence": result.extraction.evidence,
                "rejected": [
                    {"field": r.field, "reason": r.reason, "stage": r.stage}
                    for r in result.verification.rejections
                ],
                "repaired_fields": list(result.repaired_fields),
                "tool_calls": result.extraction.tool_call_count,
                "usage": result.extraction.usage.to_dict(),
                "error": result.extraction.error,
            }
        )

    return api
