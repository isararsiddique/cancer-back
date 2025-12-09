"""
ICD-11 API endpoints for WHO API integration.
Provides token management, search, and code lookup functionality.
"""
from fastapi import APIRouter, HTTPException, status, Query
from typing import Dict, Any
import httpx
from datetime import datetime, timedelta
import logging

from core.config import settings

router = APIRouter(prefix="/icd11", tags=["ICD-11"])

logger = logging.getLogger(__name__)

# WHO API Configuration
WHO_API_BASE = "https://id.who.int"
WHO_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"  # Correct token endpoint
WHO_API_URL = f"{WHO_API_BASE}/icd/release/11"

# Use provided credentials or fallback to environment variables
WHO_CLIENT_ID = "ebea7984-077e-4366-a655-65531bdb26c5_c389da01-7ffe-42ed-b382-23370ef4ab1f"
WHO_CLIENT_SECRET = "3SyTLj3I9SH6WQa3XOtnv6NmSAS1oRKryRt7xoIVUTQ="

# Token cache (in-memory, for production use Redis)
_token_cache: Dict[str, Any] = {}


def get_who_token() -> str:
    """
    Get WHO API access token.
    Uses cached token if still valid, otherwise fetches new one.
    """
    # Check cache
    if _token_cache.get("token") and _token_cache.get("expires_at"):
        if datetime.now() < _token_cache["expires_at"]:
            return _token_cache["token"]
    
    # Get credentials from settings or use defaults
    client_id = settings.who_client_id or WHO_CLIENT_ID
    client_secret = settings.who_client_secret or WHO_CLIENT_SECRET
    
    try:
        # Request token from WHO API
        # WHO API uses form data with client_id and client_secret as form fields
        response = httpx.post(
            WHO_TOKEN_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
                "scope": "icdapi_access",
            },
            timeout=15.0,
        )
        
        if response.status_code != 200:
            logger.error(f"WHO API token request failed: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to get WHO API token: {response.text}"
            )
        
        token_data = response.json()
        access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)  # Default 1 hour
        
        # Cache token with 5-minute buffer
        _token_cache["token"] = access_token
        _token_cache["expires_at"] = datetime.now() + timedelta(seconds=expires_in - 300)
        
        logger.info("WHO API token obtained successfully")
        return access_token
        
    except httpx.RequestError as e:
        logger.error(f"WHO API request error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cannot connect to WHO API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting WHO token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting WHO API token: {str(e)}"
        )


@router.get("/token", dependencies=[])
def get_token():
    """
    Get WHO API access token (public endpoint, no auth required).
    Returns token for frontend to use with WHO Embedded Coding Tool.
    """
    try:
        token = get_who_token()
        return {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": 3600,  # Approximate
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /token endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get token: {str(e)}"
        )


@router.get("/search", dependencies=[])
def search_icd11(
    q: str = Query(..., description="Search query")
):
    """
    Search ICD-11 codes by keyword.
    Public endpoint - no authentication required.
    """
    if not q or len(q.strip()) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query must be at least 2 characters"
        )
    
    try:
        token = get_who_token()
        
        # WHO API search endpoint
        search_url = f"{WHO_API_URL}/mms/search"
        
        response = httpx.get(
            search_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "API-Version": "v2",
            },
            params={
                "q": q.strip(),
                "useFlexisearch": "true",
                "flatResults": "true",
            },
            timeout=15.0,
        )
        
        if response.status_code == 404:
            # No results found
            return {
                "query": q,
                "results": [],
                "total": 0,
            }
        
        if response.status_code == 401:
            # Token expired or invalid - clear cache and retry once
            logger.warning("WHO API token expired, refreshing...")
            _token_cache.clear()
            token = get_who_token()
            response = httpx.get(
                search_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "API-Version": "v2",
                },
                params={
                    "q": q.strip(),
                    "useFlexisearch": "true",
                    "flatResults": "true",
                },
                timeout=15.0,
            )
        
        if response.status_code != 200:
            logger.error(f"WHO API search failed: {response.status_code} - {response.text}")
            error_detail = response.text[:200] if response.text else "Unknown error"
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"WHO API search failed (HTTP {response.status_code}): {error_detail}"
            )
        
        data = response.json()
        
        # Parse WHO API response
        results = []
        if "destinationEntities" in data:
            for entity in data["destinationEntities"]:
                code = entity.get("code") or entity.get("@id", "").split("/")[-1]
                title = entity.get("title", {})
                if isinstance(title, dict):
                    title = title.get("@value") or title.get("value") or ""
                
                results.append({
                    "code": code,
                    "title": title or "No description",
                    "uri": entity.get("@id") or entity.get("uri", ""),
                })
        
        return {
            "query": q,
            "results": results,
            "total": len(results),
        }
        
    except httpx.RequestError as e:
        logger.error(f"WHO API request error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cannot connect to WHO API: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/{code}", dependencies=[])
def get_icd11_code(
    code: str
):
    """
    Get detailed information for a specific ICD-11 code.
    Public endpoint - no authentication required.
    """
    if not code or not code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ICD-11 code is required"
        )
    
    code = code.strip()
    
    try:
        token = get_who_token()
        
        # WHO API entity endpoint
        entity_url = f"{WHO_API_URL}/entity/{code}"
        
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
            # Token expired or invalid - clear cache and retry once
            logger.warning("WHO API token expired, refreshing...")
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
            logger.error(f"WHO API entity lookup failed: {response.status_code} - {response.text}")
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
        
    except httpx.RequestError as e:
        logger.error(f"WHO API request error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cannot connect to WHO API: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in code lookup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code lookup failed: {str(e)}"
        )


def parse_icd11_response(raw_data: Dict[str, Any], code: str) -> Dict[str, Any]:
    """
    Parse WHO API response and extract fields for patient schema.
    """
    parsed = {
        "icd11_main_code": code,
        "icd11_description": "",
        "icd11_composite_expression": "",
        "icd11_topography_code": None,
        "icd11_topography": None,
        "icd11_morphology_code": None,
        "icd11_morphology": None,
        "icd11_behavior_code": None,
        "icd11_stage_code": None,
        "laterality": None,
        "icd11_manifestation_code": None,
        "manifestation": None,
    }
    
    # Extract title/description
    title = raw_data.get("title", {})
    if isinstance(title, dict):
        parsed["icd11_description"] = title.get("@value") or title.get("value") or ""
    elif isinstance(title, str):
        parsed["icd11_description"] = title
    
    # Extract properties
    properties = raw_data.get("properties", [])
    for prop in properties:
        prop_type = prop.get("type", "")
        prop_value = prop.get("value", {})
        
        if isinstance(prop_value, dict):
            prop_code = prop_value.get("code") or prop_value.get("@id", "").split("/")[-1]
            prop_title = prop_value.get("title", {})
            if isinstance(prop_title, dict):
                prop_title = prop_title.get("@value") or prop_title.get("value") or ""
            elif isinstance(prop_title, str):
                pass  # Already a string
            else:
                prop_title = ""
        else:
            prop_code = str(prop_value)
            prop_title = ""
        
        # Map property types to fields
        if "Topography" in prop_type or "XA" in prop_code:
            parsed["icd11_topography_code"] = prop_code
            parsed["icd11_topography"] = prop_title or prop_code
        elif "Morphology" in prop_type or "XM" in prop_code:
            parsed["icd11_morphology_code"] = prop_code
            parsed["icd11_morphology"] = prop_title or prop_code
        elif "Laterality" in prop_type or "XK" in prop_code:
            parsed["laterality"] = prop_title or prop_code
        elif "Manifestation" in prop_type or "MG" in prop_code:
            parsed["icd11_manifestation_code"] = prop_code
            parsed["manifestation"] = prop_title or prop_code
    
    # Extract code components (for behavior/stage codes)
    code_components = raw_data.get("code", "")
    if isinstance(code_components, str) and "/" in code_components:
        parts = code_components.split("/")
        for part in parts:
            if part.startswith("XS"):
                parsed["icd11_behavior_code"] = part
            elif part.startswith("XK"):
                parsed["laterality"] = part
            elif part.startswith("MG"):
                parsed["icd11_manifestation_code"] = part
    
    # Keep None for empty fields (don't set to "NA")
    # Frontend will handle converting None/null to "null" string for display
    # Backend will store None in database
    
    return parsed

