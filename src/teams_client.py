"""Client for Microsoft Graph Education API: list classes, assignments, and submissions."""
import requests

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def _get(url: str, token: str) -> dict:
    # Shared GET helper with bearer auth and error handling
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        raise RuntimeError(
            f"Graph API request failed ({response.status_code}): {response.text}"
        )
    return response.json()


def get_my_classes(token: str) -> list[dict]:
    """Return the classes the signed-in teacher belongs to."""
    data = _get(f"{GRAPH_BASE}/education/me/classes", token)
    return [{"id": c["id"], "displayName": c["displayName"]} for c in data.get("value", [])]


def get_assignments(token: str, class_id: str) -> list[dict]:
    """Return assignments for a given class."""
    data = _get(f"{GRAPH_BASE}/education/classes/{class_id}/assignments", token)
    return [
        {
            "id": a["id"],
            "displayName": a["displayName"],
            "status": a.get("status"),
            "dueDateTime": a.get("dueDateTime"),
        }
        for a in data.get("value", [])
    ]


def get_class_members(token: str, class_id: str) -> list[dict]:
    """Return the roster of a class (teachers and students), including display names."""
    data = _get(f"{GRAPH_BASE}/education/classes/{class_id}/members", token)
    return [{"id": m["id"], "displayName": m.get("displayName")} for m in data.get("value", [])]


def get_submissions(token: str, class_id: str, assignment_id: str) -> list[dict]:
    """Return student submissions for a given assignment."""
    data = _get(
        f"{GRAPH_BASE}/education/classes/{class_id}/assignments/{assignment_id}/submissions",
        token,
    )
    return [
        {
            "id": s["id"],
            "recipientId": s.get("recipient", {}).get("userId"),
            "status": s.get("status"),
            "submittedDateTime": s.get("submittedDateTime"),
        }
        for s in data.get("value", [])
    ]


def get_submission_resources(
    token: str, class_id: str, assignment_id: str, submission_id: str
) -> list[dict]:
    """Return downloadable file resources attached to a submission (link-only resources are skipped)."""
    data = _get(
        f"{GRAPH_BASE}/education/classes/{class_id}/assignments/{assignment_id}"
        f"/submissions/{submission_id}/resources",
        token,
    )
    resources = []
    for r in data.get("value", []):
        resource = r.get("resource", {})
        file_url = resource.get("fileUrl")
        if file_url:
            resources.append({"displayName": resource.get("displayName"), "fileUrl": file_url})
    return resources


def download_resource(token: str, file_url: str, save_path: str) -> str:
    """Download a submission resource's file content and save it to save_path."""
    response = requests.get(f"{file_url}/content", headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        raise RuntimeError(
            f"Resource download failed ({response.status_code}): {response.text}"
        )
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path
