"""Authenticate against Microsoft Graph Education API using MSAL device code flow."""
import os
import atexit

import msal
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["EduAssignments.ReadBasic", "User.Read"]

# Cache file keeps the signed-in session so teachers don't log in every run
TOKEN_CACHE_PATH = os.path.join(os.path.dirname(__file__), "token_cache.bin")


def _load_cache() -> msal.SerializableTokenCache:
    # Load token cache from disk if it exists
    cache = msal.SerializableTokenCache()
    if os.path.exists(TOKEN_CACHE_PATH):
        with open(TOKEN_CACHE_PATH, "r", encoding="utf-8") as f:
            cache.deserialize(f.read())

    def save_cache():
        if cache.has_state_changed:
            with open(TOKEN_CACHE_PATH, "w", encoding="utf-8") as f:
                f.write(cache.serialize())

    atexit.register(save_cache)
    return cache


def get_access_token() -> str:
    """Return a valid access token, signing in via device code flow if needed."""
    if not CLIENT_ID or not TENANT_ID:
        raise RuntimeError("AZURE_CLIENT_ID / AZURE_TENANT_ID missing from .env")

    cache = _load_cache()
    app = msal.PublicClientApplication(
        client_id=CLIENT_ID, authority=AUTHORITY, token_cache=cache
    )

    # Try silent auth first using a cached account
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            return result["access_token"]

    # No cached session: fall back to device code flow
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise RuntimeError(f"Failed to start device flow: {flow}")

    print(flow["message"])  # tells the teacher which URL and code to enter

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(
            f"Authentication failed: {result.get('error_description', result)}"
        )

    return result["access_token"]


if __name__ == "__main__":
    token = get_access_token()
    print("Login succeeded. Access token acquired (not printed for safety).")
