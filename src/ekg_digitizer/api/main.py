# ==============================================================================
# File Name : main.py
# Purpose   : FastAPI app: upload an EKG image, get the digitized time series
# ==============================================================================
"""Run locally:

    uvicorn ekg_digitizer.api.main:app --reload

The model is loaded once at startup, from the W&B registry's ``production`` alias
(needs ``WANDB_API_KEY``) or from ``EKG_CHECKPOINT`` when set. Interactive docs
are served at ``/docs``; the upload page at ``/``.
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ekg_digitizer.api.schemas import (
    DigitizeResponse,
    ErrorResponse,
    Health,
    LeadSignal,
    Ready,
)
from ekg_digitizer.api.service import (
    Digitizer,
    InvalidImageError,
    ModelOutputError,
    UnsupportedImageTypeError,
    decode_image,
)
from ekg_digitizer.api.settings import Settings
from ekg_digitizer.config import load_config
from ekg_digitizer.evaluation.metrics import LEADS
from ekg_digitizer.inference import default_device, load_model
from ekg_digitizer.serving import load_production_model

logger = logging.getLogger("ekg_digitizer.api")
STATIC_DIR = Path(__file__).parent / "static"


def build_digitizer(settings: Settings) -> Digitizer:
    """Load the model named by the settings (local file, else registry production alias)."""
    config = load_config(str(settings.config_path))
    device = default_device()

    if settings.checkpoint:
        model, version = load_model(settings.checkpoint, config, device), f"local:{settings.checkpoint.name}"
    else:
        # load_production_model takes the config and builds the model itself
        served = load_production_model(config, device)
        model, version = served.model.to(device), served.version

    logger.info("Loaded model %s on %s", version, device)
    return Digitizer(model, config, device, version, settings.max_concurrency)


def create_app(settings: Settings | None = None, digitizer: Digitizer | None = None) -> FastAPI:
    """App factory. Pass ``digitizer`` to skip model loading (used by tests)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast: a service without a model should not pass its startup probe
        app.state.digitizer = digitizer or await run_in_threadpool(build_digitizer, settings)
        yield
        app.state.digitizer = None

    app = FastAPI(
        title="EKG Digitizer",
        description="Extract 12-lead time-series signals from an image of an EKG record.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info("%s %s -> %d (%.0f ms)", request.method, request.url.path,
                    response.status_code, (time.perf_counter() - start) * 1000)
        return response

    @app.exception_handler(ModelOutputError)
    async def model_output_handler(request: Request, exc: ModelOutputError):
        logger.error("%s", exc)
        return JSONResponse({"detail": "Inference failed"}, status_code=500)

    # ---- Probes -----------------------------------------------------------

    @app.get("/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        """Liveness: the process is up."""
        return Health()

    @app.get("/ready", response_model=Ready, responses={503: {"model": Ready}}, tags=["ops"])
    def ready(request: Request, response: Response) -> Ready:
        """Readiness: the model is loaded. Also reports which version is being served."""
        served: Digitizer | None = getattr(request.app.state, "digitizer", None)
        if served is None:
            response.status_code = 503
            return Ready(status="loading")
        return Ready(status="ready", model_version=served.version, device=str(served.device))

    # ---- Inference --------------------------------------------------------

    @app.post(
        "/api/v1/digitize",
        response_model=DigitizeResponse,
        responses={
            200: {"content": {"text/csv": {}}, "description": "JSON, or CSV with ?format=csv"},
            400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
            415: {"model": ErrorResponse}, 503: {"model": ErrorResponse},
        },
        tags=["inference"],
    )
    async def digitize(
        request: Request,
        file: Annotated[UploadFile, File(description="PNG or JPEG image of an EKG record")],
        fs: Annotated[int | None, Query(ge=100, le=1000, description="Output sampling rate in Hz "
                                        "(default: the model's native rate)")] = None,
        format: Annotated[Literal["json", "csv"], Query()] = "json",
    ):
        served: Digitizer | None = request.app.state.digitizer
        if served is None:
            raise HTTPException(503, "Model not loaded")

        # Reject early from the header, then enforce the limit on what is actually read
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + 4096:
            raise HTTPException(413, f"File exceeds {settings.max_upload_bytes // 2**20} MB limit")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"File exceeds {settings.max_upload_bytes // 2**20} MB limit")

        try:
            image = await run_in_threadpool(decode_image, data, settings.max_image_pixels)
        except UnsupportedImageTypeError as err:
            raise HTTPException(415, str(err)) from err
        except InvalidImageError as err:
            raise HTTPException(400, str(err)) from err

        out_fs = fs or served.base_fs
        record = await run_in_threadpool(served.digitize, image, fs)

        if format == "csv":
            # Same layout as write_record: one column per lead, shorter leads padded empty
            frame = pd.DataFrame({lead: pd.Series(record[lead]) for lead in LEADS if lead in record})
            buffer = io.StringIO()
            frame.to_csv(buffer, index=False, float_format="%.6f")
            return Response(
                buffer.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="ekg_signals.csv"',
                         "X-Model-Version": served.version, "X-Sampling-Rate": str(out_fs)},
            )

        return DigitizeResponse(
            model_version=served.version,
            fs=out_fs,
            leads={lead: LeadSignal(n_samples=len(record[lead]), duration_s=len(record[lead]) / out_fs,
                                    values=record[lead].tolist())
                   for lead in LEADS if lead in record},
        )

    # ---- Web page ---------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()