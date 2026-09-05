"""
Copernicus OAuth2 Client Credentials Authentication Service.
Handles token retrieval, automatic renewal, token caching, and resilient error reporting.
"""

import os
import time
import json
import logging
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any, Tuple

from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

logger = logging.getLogger("satquery.copernicus.auth")

# Endpoints for Copernicus Data Space & Sentinel Hub
CDSE_AUTH_URL = os.getenv(
    "COPERNICUS_AUTH_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
SENTINEL_HUB_AUTH_URL = os.getenv(
    "SENTINEL_HUB_AUTH_URL",
    "https://services.sentinel-hub.com/oauth/token"
)


class CopernicusAuthService:
    """
    Manages OAuth2 token lifecycle for Copernicus Data Space & Sentinel Hub.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        auth_url: Optional[str] = None
    ):
        self._client_id = client_id or os.getenv("COPERNICUS_CLIENT_ID")
        self._client_secret = client_secret or os.getenv("COPERNICUS_CLIENT_SECRET")
        self._auth_url = auth_url or CDSE_AUTH_URL
        self._sentinel_hub_auth_url = SENTINEL_HUB_AUTH_URL

        # Token cache: (token_str, expiry_epoch)
        self._cached_token: Optional[str] = None
        self._token_expiry_epoch: float = 0.0
        self._last_error: Optional[str] = None

    @property
    def has_credentials(self) -> bool:
        """Returns True if client credentials are provided."""
        return bool(self._client_id and self._client_secret)

    @property
    def client_id(self) -> Optional[str]:
        return self._client_id

    def get_token(self, force_refresh: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Retrieves a valid OAuth2 Bearer token.
        Automatically uses cached token if valid; otherwise fetches a new one.
        
        Returns:
            (access_token, error_message)
        """
        if not self.has_credentials:
            err = "Copernicus credentials not configured. Please set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in .env."
            self._last_error = err
            return None, err

        # Check in-memory cache (refresh if within 60 seconds of expiry)
        now = time.time()
        if not force_refresh and self._cached_token and now < (self._token_expiry_epoch - 60):
            return self._cached_token, None

        # Attempt token exchange
        for auth_url in [self._auth_url, self._sentinel_hub_auth_url]:
            try:
                token, expires_in = self._request_oauth_token(auth_url)
                if token:
                    self._cached_token = token
                    self._token_expiry_epoch = now + expires_in
                    self._last_error = None
                    logger.info(f"Successfully obtained Copernicus OAuth token from {auth_url} (expires in {expires_in}s).")
                    return token, None
            except Exception as e:
                logger.warning(f"Failed OAuth token request against {auth_url}: {e}")
                self._last_error = str(e)

        # In case CDSE OAuth server is offline, return descriptive diagnostic
        diagnostic = f"Copernicus Authentication Failed: {self._last_error or 'Unable to establish OAuth2 handshake'}"
        return None, diagnostic

    def _request_oauth_token(self, auth_url: str) -> Tuple[str, int]:
        """Performs form-encoded POST request to OAuth2 token endpoint."""
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }).encode("utf-8")

        req = urllib.request.Request(
            auth_url,
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "SatQuery-AI/1.0",
                "Accept": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            token = data.get("access_token")
            expires_in = int(data.get("expires_in", 3600))
            if not token:
                raise ValueError("Response did not contain an access_token.")
            return token, expires_in

    def invalidate_token(self) -> None:
        """Clears cached token to force immediate re-authentication."""
        self._cached_token = None
        self._token_expiry_epoch = 0.0


# Global singleton instance
copernicus_auth = CopernicusAuthService()
