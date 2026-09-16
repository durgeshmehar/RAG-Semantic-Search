import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.v1.router import router as v1_router
from .core import config
from .core.error_handlers import register_error_handlers
from .core.logging_config import setup_logging
from .core.openapi import register_openapi
from .db import db
from .repositories import vector_repository
from .tasks import job_queue, worker

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.init_db()

    # Fail fast on a misconfigured QDRANT_URL rather than only discovering it
    # on the first upload's indexing attempt.
    try:
        vector_repository.get_client().get_collections()
        logger.info("connected to Qdrant at %s", config.QDRANT_URL)
    except Exception:
        logger.exception(
            "could not reach Qdrant at %s -- is the qdrant service up?",
            config.QDRANT_URL,
        )
        raise

    # Anything left mid-flight by an unclean shutdown goes back on the queue.
    recovered = job_queue.recover_stuck_jobs()
    if recovered:
        logger.info("recovered %d chunks left in-flight by a previous run", recovered)

    worker.start_workers()
    logger.info("started %d indexing workers", config.WORKER_COUNT)

    yield

    worker.stop_workers()
    db.close_connection()


app = FastAPI(
    title="Large File Processing & Search",
    version="1.0.0",
    lifespan=lifespan,
    description=(
        "Upload text files up to 10 GB on a 4 GB machine, resume interrupted "
        "uploads, and search their contents in natural language.\n\n"
        "Every request takes an `X-User-Id` header identifying the caller "
        "(any string; omit it and requests share one anonymous identity). "
        "Files are scoped to their owner -- listing, status, search, and "
        "delete only see files created under the same `X-User-Id`. Click "
        "**Authorize** below and set it once to have it applied to every "
        "request tried from this page.\n\n"
        "**Upload flow** (every path below is under `/v1`)\n"
        "1. `POST /v1/files` to register the upload and get a `file_id`.\n"
        "2. `PUT /v1/files/{file_id}/chunk?offset=N` repeatedly with raw chunk bodies.\n"
        "3. `POST /v1/files/{file_id}/complete` once every chunk has been sent.\n"
        "4. `GET /v1/files/{file_id}/status` to watch progress -- or, after an "
        "interruption, to learn the offset to resume from.\n"
        "5. `POST /v1/files/{file_id}/search` once `searchable` is true.\n\n"
        "Indexing runs during the upload, so passages become searchable before "
        "`complete` is even called."
    ),
    # Keeps the X-User-Id value entered via "Authorize" (below) filled in
    # across page reloads and every "Try it out" call, instead of it
    # resetting per request or per endpoint.
    swagger_ui_parameters={"persistAuthorization": True},
)


app.include_router(v1_router, prefix="/v1")


@app.get("/health", tags=["meta"], summary="Liveness, worker, and Qdrant state")
def health() -> dict:
    try:
        vector_repository.get_client().get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "workers_running": worker._pool.running if worker._pool else False,
        "queue_depth": job_queue.pending_count(),
        "qdrant_reachable": qdrant_ok,
    }


register_error_handlers(app)
register_openapi(app)



"""Application main file - entry point.

Startup order matters: the schema must exist before jobs can be recovered, and
recovery must run before workers start, or a worker could claim a row that
recovery is about to reset.
"""
