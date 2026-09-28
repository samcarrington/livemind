"""Static HTML/SVG page routes."""

import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from starlette.responses import RedirectResponse

router = APIRouter()

_STATIC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)


def _html(name: str) -> FileResponse:
    return FileResponse(os.path.join(_STATIC, name))


@router.get("/livemind-favicon.svg")
async def serve_favicon():
    return FileResponse(
        os.path.join(_STATIC, "livemind-favicon.svg"), media_type="image/svg+xml"
    )


@router.get("/livemind-logo-clean.svg")
async def serve_logo():
    return FileResponse(
        os.path.join(_STATIC, "livemind-logo-clean.svg"), media_type="image/svg+xml"
    )


@router.get("/", response_class=HTMLResponse)
async def serve_main():
    return _html("live-mindmap.html")


@router.get("/monitor", response_class=HTMLResponse)
async def serve_monitor():
    return _html("monitor.html")


@router.get("/doc", response_class=HTMLResponse)
async def serve_doc():
    return _html("doc.html")


@router.get("/doc/admin", response_class=HTMLResponse)
async def serve_doc_admin():
    return _html("doc-admin.html")


@router.get("/sessions", response_class=HTMLResponse)
async def serve_sessions():
    return _html("sessions.html")


@router.get("/sessions/archive", response_class=HTMLResponse)
async def serve_archive():
    return _html("sessions.html")


@router.get("/admin/sessions")
async def redirect_admin_sessions():
    return RedirectResponse("/sessions")


@router.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    return _html("admin.html")
