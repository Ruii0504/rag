from pathlib import Path

from fastapi import FastAPI

from rag_customer_service.api import create_app
from rag_customer_service.bootstrap import get_container
from rag_customer_service.config import Settings


def create_production_app(base_dir: Path | None = None) -> FastAPI:
    """创建同时提供 API 和前端静态资源的生产应用。"""
    project_dir = (
        Path(base_dir).resolve()
        if base_dir is not None
        else Path(__file__).resolve().parents[2]
    )
    settings = Settings.from_env(project_dir)
    container = get_container(project_dir)
    return create_app(
        container,
        settings,
        frontend_dir=project_dir / "frontend" / "dist",
    )
