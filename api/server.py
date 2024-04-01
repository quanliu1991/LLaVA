from fastapi import FastAPI
from api.routes.router import router
from api.config import EnvVar
from api.event_handlers import start_app_handler, stop_app_handler, exception_handler
import os
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

path = os.path.dirname(os.path.realpath(__file__))
static_dir = os.path.join(path, 'static')


def get_app() -> FastAPI:
    fast_app = FastAPI(
        title=EnvVar.APP_NAME,
        version=EnvVar.APP_VERSION,
        debug=EnvVar.IS_DEBUG,
    )
    fast_app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @fast_app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=fast_app.openapi_url,
            title=fast_app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @fast_app.get(fast_app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
    async def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    @fast_app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=fast_app.openapi_url,
            title=fast_app.title + " - ReDoc",
            redoc_js_url="/static/redoc.standalone.js",
        )

    fast_app.include_router(router, prefix=EnvVar.API_PREFIX)

    fast_app.add_event_handler("startup", start_app_handler(fast_app))
    fast_app.add_event_handler("shutdown", stop_app_handler(fast_app))
    fast_app.add_exception_handler(Exception, exception_handler)

    return fast_app


app = get_app()
