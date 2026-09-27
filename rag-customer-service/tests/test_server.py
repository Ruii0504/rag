from types import SimpleNamespace
from pathlib import Path

from fastapi.testclient import TestClient


def test_production_factory_loads_project_settings_container_and_frontend(tmp_path, monkeypatch):
    from rag_customer_service import server

    frontend = tmp_path / "frontend" / "dist"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text("<main>production</main>", encoding="utf-8")
    settings = SimpleNamespace()
    container = SimpleNamespace()
    calls = []
    monkeypatch.setattr(
        server.Settings,
        "from_env",
        classmethod(lambda cls, base_dir=None: settings),
    )
    monkeypatch.setattr(
        server,
        "get_container",
        lambda base_dir=None: calls.append(base_dir) or container,
    )

    app = server.create_production_app(tmp_path)

    with TestClient(app) as client:
        assert client.get("/").text == "<main>production</main>"
    assert calls == [tmp_path]


def test_production_factory_defaults_to_project_root_instead_of_current_directory(
    tmp_path,
    monkeypatch,
):
    from rag_customer_service import server

    settings = SimpleNamespace()
    container = SimpleNamespace()
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        server.Settings,
        "from_env",
        classmethod(lambda cls, base_dir=None: settings),
    )
    monkeypatch.setattr(
        server,
        "get_container",
        lambda base_dir=None: calls.append(base_dir) or container,
    )

    server.create_production_app()

    assert calls == [Path(server.__file__).resolve().parents[2]]
