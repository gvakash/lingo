"""
tasks.py — Celery task definitions for async pipeline processing
"""
from celery import Celery
from app.config import settings

celery_app = Celery(
    "lingo_pipeline",
    broker=settings.CELERY_BROKER,
    backend=settings.CELERY_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
)

app = celery_app


@celery_app.task(bind=True, name="tasks.run_pipeline")
def run_pipeline_task(self, audio_path: str, config_dict: dict):
    """Async pipeline task — called by API for long-running jobs."""
    from app.core.pipeline import SpeechPipeline, PipelineConfig

    self.update_state(state="PROGRESS", meta={"stage": "starting"})
    pipeline = SpeechPipeline()

    config = PipelineConfig(**config_dict)
    self.update_state(state="PROGRESS", meta={"stage": "processing"})

    result = pipeline.run(audio_path, config)
    return {
        "job_id": result.job_id,
        "status": result.status,
        "output_files": result.output_files,
        "total_time_secs": result.total_time_secs,
    }
