"""Customizes the generated OpenAPI schema: an Authorize button for X-User-Id,
and a hand-written requestBody for the one endpoint FastAPI can't infer a
schema for on its own.
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def register_openapi(app: FastAPI) -> None:
    def _custom_openapi() -> dict:
        """Register X-User-Id as a real security scheme, and document the one
        endpoint whose body FastAPI can't infer a schema for on its own.

        Without the security scheme, Swagger has no "Authorize" button for a
        plain header parameter -- every endpoint would need X-User-Id typed in
        separately, by hand, per try. Declaring it as an apiKey security scheme
        applied to every route gives Swagger one Authorize dialog that then
        auto-fills the header on every request tried from the page.

        PUT /v1/files/{file_id}/chunk reads its body via the raw Request rather
        than a typed parameter (see app/api/v1/upload.py's upload_chunk docstring for why:
        FastAPI 0.115's Body(media_type=...) 422s on real clients' differing
        default Content-Type headers). A raw Request is invisible to FastAPI's
        schema generator, so without this, Swagger's "Try it out" for that
        endpoint shows no body field at all. The requestBody below is added by
        hand purely for documentation -- it does not change how the endpoint
        parses the request at runtime.
        """
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema["components"]["securitySchemes"] = {
            "UserId": {
                "type": "apiKey",
                "in": "header",
                "name": "X-User-Id",
                "description": "Any string identifying the caller. Files are scoped to it.",
            }
        }
        for path in schema["paths"].values():
            for operation in path.values():
                operation.setdefault("security", []).append({"UserId": []})

        schema["paths"]["/v1/files/{file_id}/chunk"]["put"]["requestBody"] = {
            "required": True,
            "content": {
                "application/octet-stream": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                        "description": "Raw chunk bytes -- not JSON, not base64, "
                        "not wrapped in a field. The exact bytes of the file at "
                        "[offset, offset+len(body)).",
                    }
                }
            },
        }

        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = _custom_openapi
