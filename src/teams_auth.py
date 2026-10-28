"""Authenticate against Microsoft Graph Education API using MSAL device code flow."""
import os
import sys
import atexit

import msal
from dotenv import load_dotenv

# Inside the frozen exe, read the .env sitting next to the exe so it works no matter which
# folder the exe was launched from (shortcut, Start menu, etc.)
if getattr(sys, "frozen", False):
    load_dotenv(os.path.join(os.path.dirname(sys.executable), ".env"))
else:
    load_dotenv()

CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["EduAssignments.ReadBasic", "EduRoster.ReadBasic", "Files.Read.All", "User.Read"]

# Cache file keeps the signed-in session so teachers don't log in every run.
# Inside the frozen exe __file__ lives in a temp folder that's deleted on exit, so keep
# the cache next to the exe instead.
if getattr(sys, "frozen", False):
    _CACHE_DIR = os.path.dirname(sys.executable)
else:
    _CACHE_DIR = os.path.dirname(__file__)
TOKEN_CACHE_PATH = os.path.join(_CACHE_DIR, "token_cache.bin")


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


def _make_app() -> tuple[msal.PublicClientApplication, msal.SerializableTokenCache]:
    if not CLIENT_ID or not TENANT_ID:
        raise RuntimeError("AZURE_CLIENT_ID / AZURE_TENANT_ID missing from .env")

    cache = _load_cache()
    app = msal.PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY, token_cache=cache)
    return app, cache


def get_cached_token() -> str | None:
    """Return an access token from a previous sign-in, or None if the teacher must sign in."""
    app, _ = _make_app()
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            return result["access_token"]
    return None


def start_sign_in() -> dict:
    """Start a device code sign-in; the returned flow holds the user_code and verification_uri
    to show the teacher, and is then passed to finish_sign_in()."""
    app, _ = _make_app()
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise RuntimeError(f"Failed to start device flow: {flow}")
    return flow


def finish_sign_in(flow: dict) -> str:
    """Block until the teacher enters the code and signs in, then return the access token."""
    app, cache = _make_app()
    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(
            f"Authentication failed: {result.get('error_description', result)}"
        )

    # Save right away: the Streamlit server keeps running, so atexit alone would lose
    # the session if the exe is closed by killing its console window
    if cache.has_state_changed:
        with open(TOKEN_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(cache.serialize())

    return result["access_token"]


def get_access_token() -> str:
    """Return a valid access token, signing in via device code flow in the console if needed."""
    token = get_cached_token()
    if token:
        return token

    flow = start_sign_in()
    print(flow["message"])  # tells the user which URL and code to enter
    return finish_sign_in(flow)


def sign_out() -> None:
    """Forget the cached Microsoft sign-in so the next login starts fresh.

    Writes the cache to disk immediately (rather than relying on the atexit-based save
    in _load_cache, which only fires when the whole process exits) since the Streamlit
    server process keeps running across a sign-out.
    """
    if not CLIENT_ID or not TENANT_ID or not os.path.exists(TOKEN_CACHE_PATH):
        return

    cache = msal.SerializableTokenCache()
    with open(TOKEN_CACHE_PATH, "r", encoding="utf-8") as f:
        cache.deserialize(f.read())

    app = msal.PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY, token_cache=cache)
    for account in app.get_accounts():
        app.remove_account(account)

    with open(TOKEN_CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(cache.serialize())


if __name__ == "__main__":
    token = get_access_token()
    print("Login succeeded. Access token acquired (not printed for safety).")
