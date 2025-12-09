from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from typing import Optional
import os

from api.v1 import auth, users, roles, patients, organizations, research, admin, audit, icd11, projects, ml_training
from api.v1 import ml_training_train, ml_execute

app = FastAPI(
    title="National Registry API",
    version="1.0.0",
    description="""
    National Cancer Registry Management System API
    
    ## Authentication
    
    To use this API:
    1. Login via `/api/v1/auth/login` with your email and password
    2. Receive JWT `access_token` and `refresh_token`
    3. Use the `access_token` in the Authorization header: `Authorization: Bearer <access_token>`
    4. When access_token expires, use `/api/v1/auth/refresh` with your `refresh_token`
    
    All protected endpoints require the Authorization header with a valid JWT token.
    """,
    swagger_ui_parameters={
        "persistAuthorization": True,
    }
)


def custom_openapi():
    """
    Custom OpenAPI schema generator that ONLY includes HTTPBearer authentication.
    Completely removes OAuth2PasswordBearer and any OAuth2 schemes.
    """
    # Force regeneration - clear cache
    app.openapi_schema = None
    
    # Get the OpenAPI schema from FastAPI
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # AGGRESSIVE CLEANUP: Remove ALL security schemes first
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    # DELETE everything in securitySchemes
    if "securitySchemes" in openapi_schema["components"]:
        del openapi_schema["components"]["securitySchemes"]
    
    # CREATE ONLY HTTPBearer - nothing else
    openapi_schema["components"]["securitySchemes"] = {
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /api/v1/auth/login endpoint. Use format: Bearer <token>"
        }
    }
    
    # Update ALL endpoints to use ONLY HTTPBearer
    for path, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if isinstance(operation, dict) and method.lower() in ["get", "post", "put", "delete", "patch"]:
                # If endpoint has security, replace ALL with HTTPBearer only
                if "security" in operation:
                    # Replace entire security array with ONLY HTTPBearer
                    operation["security"] = [{"HTTPBearer": []}]
    
    # FINAL VERIFICATION: Force only HTTPBearer exists
    if "components" in openapi_schema and "securitySchemes" in openapi_schema["components"]:
        schemes = openapi_schema["components"]["securitySchemes"]
        # Remove any OAuth2 keys
        cleaned_schemes = {}
        for key, value in schemes.items():
            if "oauth" not in key.lower() and "OAuth" not in key:
                if key == "HTTPBearer":
                    cleaned_schemes[key] = value
        
        # If HTTPBearer doesn't exist, create it
        if "HTTPBearer" not in cleaned_schemes:
            cleaned_schemes["HTTPBearer"] = {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token obtained from /api/v1/auth/login endpoint. Use format: Bearer <token>"
            }
        
        # Set ONLY HTTPBearer
        openapi_schema["components"]["securitySchemes"] = cleaned_schemes
    
    # Cache the cleaned schema
    app.openapi_schema = openapi_schema
    
    return openapi_schema


# Configure CORS - Allow frontend origins with credentials
# IMPORTANT: Must be added BEFORE routes are included

# Get allowed origins from environment variable or use defaults
ALLOWED_ORIGINS_STR = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3001,http://127.0.0.1:3001,http://localhost:3000,http://127.0.0.1:3000"
)

# Check if wildcard is used
if ALLOWED_ORIGINS_STR.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,  # Allow credentials (cookies, authorization headers)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, PATCH, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
    max_age=3600,  # Cache preflight requests for 1 hour
)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
app.include_router(roles.router, prefix="/api/v1")
app.include_router(patients.router, prefix="/api/v1")
app.include_router(organizations.router, prefix="/api/v1")
app.include_router(research.router, prefix="/api/v1")
app.include_router(projects.router, prefix="/api/v1")
app.include_router(ml_training.router, prefix="/api/v1")
app.include_router(ml_training_train.router, prefix="/api/v1")
app.include_router(ml_execute.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
app.include_router(icd11.router, prefix="/api/v1")

# Legacy endpoints for WHO ECT widget compatibility
@app.get("/api/token", dependencies=[])
def legacy_token_endpoint():
    """
    Legacy endpoint for WHO ECT widget compatibility.
    Redirects to /api/v1/icd11/token
    """
    from api.v1.icd11 import get_who_token
    try:
        token = get_who_token()
        return {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": 3600,
        }
    except Exception as e:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get token: {str(e)}"
        )

@app.get("/api/icd/{code}", dependencies=[])
def legacy_icd_endpoint(code: str):
    """
    Legacy endpoint for WHO ECT widget compatibility.
    Calls the same logic as /api/v1/icd11/{code}
    """
    from fastapi import HTTPException, status
    from api.v1.icd11 import get_who_token, parse_icd11_response
    import httpx
    
    if not code or not code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ICD-11 code is required"
        )
    
    code = code.strip()
    
    try:
        token = get_who_token()
        
        # WHO API entity endpoint
        entity_url = f"https://id.who.int/icd/release/11/entity/{code}"
        
        response = httpx.get(
            entity_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "API-Version": "v2",
            },
            timeout=15.0,
        )
        
        if response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ICD-11 code '{code}' not found"
            )
        
        if response.status_code == 401:
            # Token expired - clear cache and retry once
            from api.v1.icd11 import _token_cache
            _token_cache.clear()
            token = get_who_token()
            response = httpx.get(
                entity_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "API-Version": "v2",
                },
                timeout=15.0,
            )
        
        if response.status_code != 200:
            error_detail = response.text[:200] if response.text else "Unknown error"
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"WHO API lookup failed (HTTP {response.status_code}): {error_detail}"
            )
        
        raw_data = response.json()
        
        # Parse and extract relevant fields
        parsed = parse_icd11_response(raw_data, code)
        
        return {
            "code": code,
            "raw": raw_data,
            "parsed": parsed,
            "auto_fill_fields": parsed,  # For backward compatibility
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ICD code: {str(e)}"
        )

# Set custom OpenAPI schema AFTER all routes are registered
# This ensures we can properly remove OAuth2 from the schema
app.openapi = custom_openapi
