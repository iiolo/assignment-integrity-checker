"""Local Streamlit UI for checking student assignments for plagiarism and AI-generated text."""
import os
import tempfile

import streamlit as st

from src.ai_detection_check import check_ai_generated
from src.extract_text import extract_text
from src.similarity import calculate_similarities
from src.teams_auth import get_access_token
from src.teams_client import (
    download_resource,
    get_assignments,
    get_class_members,
    get_my_classes,
    get_submission_resources,
    get_submissions,
)

st.set_page_config(page_title="Assignment Integrity Checker", layout="wide")
st.title("Assignment Integrity Checker")


def ai_score_to_grade(score: float) -> str:
    """Map an AI-generated probability (0~100) to a Low/Medium/High grade."""
    if score < 34:
        return "Low"
    if score < 67:
        return "Medium"
    return "High"


def save_upload_to_temp(uploaded_file) -> str:
    """Save an in-memory uploaded file to disk so extract_text() can read it by path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def render_report(student_texts: dict[str, str], threshold: int) -> None:
    """Run similarity + AI detection on the given texts and render the report."""
    st.subheader("Similarity between submissions")
    similarity_results = calculate_similarities(student_texts, threshold=threshold)
    if similarity_results:
        st.table(
            [
                {"Student A": a, "Student B": b, "Similarity (%)": f"{score:.1f}"}
                for a, b, score in similarity_results
            ]
        )
    else:
        st.write(f"No pairs found at or above the {threshold}% threshold.")

    st.subheader("AI-generated text likelihood")
    with st.spinner("Running AI detection model..."):
        ai_rows = []
        for student_name, text in student_texts.items():
            score = check_ai_generated(text)
            ai_rows.append(
                {
                    "Student": student_name,
                    "AI-generated probability (%)": f"{score:.1f}",
                    "Grade": ai_score_to_grade(score),
                }
            )
    st.table(ai_rows)

    st.warning(
        "These results are for reference only and should not be used as the sole basis "
        "for a plagiarism or AI-use decision."
    )

    st.subheader("Extracted text (for verification)")
    st.caption("Check what was actually extracted and compared for each submission.")
    for student_name, text in student_texts.items():
        with st.expander(student_name):
            st.text(text if text else "(no text extracted)")


threshold = st.slider("Similarity alert threshold (%)", min_value=0, max_value=100, value=40)

mode = st.radio("Get submissions from", ["Upload files", "Fetch from Teams"])

if mode == "Upload files":
    st.caption(
        "Upload each student's assignment file. The file name (without extension) "
        "is used as the student's name."
    )
    uploaded_files = st.file_uploader(
        "Student assignment files (.docx, .pdf)",
        type=["docx", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("Run check", disabled=not uploaded_files):
        student_texts: dict[str, str] = {}

        # Extract text from every uploaded file, keyed by student name (file name)
        with st.spinner("Extracting text from files..."):
            for uploaded_file in uploaded_files:
                student_name = os.path.splitext(uploaded_file.name)[0]
                temp_path = save_upload_to_temp(uploaded_file)
                try:
                    student_texts[student_name] = extract_text(temp_path)
                except (ValueError, FileNotFoundError) as exc:
                    st.error(f"Failed to read {uploaded_file.name}: {exc}")
                finally:
                    os.remove(temp_path)

        render_report(student_texts, threshold)

else:
    if "graph_token" not in st.session_state:
        st.info(
            "Signing in will print a login code to the console window behind this browser "
            "tab. Open https://login.microsoft.com/device and enter that code to continue."
        )
        if st.button("Sign in with Microsoft"):
            with st.spinner("Waiting for sign-in to complete..."):
                st.session_state.graph_token = get_access_token()
            st.rerun()
    else:
        token = st.session_state.graph_token

        # Cache the class list in session state; it rarely changes within one session
        if "classes" not in st.session_state:
            with st.spinner("Loading classes..."):
                st.session_state.classes = get_my_classes(token)
        classes = st.session_state.classes

        if not classes:
            st.warning("No classes found for this account.")
        else:
            class_choice = st.selectbox("Class", classes, format_func=lambda c: c["displayName"])
            assignments = get_assignments(token, class_choice["id"])

            if not assignments:
                st.warning("No assignments found for this class.")
            else:
                assignment_choice = st.selectbox(
                    "Assignment", assignments, format_func=lambda a: a["displayName"]
                )

                if st.button("Fetch submissions"):
                    members = get_class_members(token, class_choice["id"])
                    name_by_id = {m["id"]: m["displayName"] for m in members}

                    submissions = get_submissions(token, class_choice["id"], assignment_choice["id"])
                    submitted = [s for s in submissions if s["submittedDateTime"]]

                    student_texts: dict[str, str] = {}
                    with st.spinner(f"Downloading {len(submitted)} submissions..."):
                        for sub in submitted:
                            student_name = name_by_id.get(sub["recipientId"], sub["recipientId"])
                            resources = get_submission_resources(
                                token, class_choice["id"], assignment_choice["id"], sub["id"]
                            )

                            texts = []
                            for resource in resources:
                                suffix = os.path.splitext(resource["displayName"])[1]
                                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                    temp_path = tmp.name
                                try:
                                    download_resource(token, resource["fileUrl"], temp_path)
                                    texts.append(extract_text(temp_path))
                                except (ValueError, FileNotFoundError, RuntimeError) as exc:
                                    st.error(f"Failed to read {resource['displayName']}: {exc}")
                                finally:
                                    os.remove(temp_path)

                            if texts:
                                student_texts[student_name] = "\n".join(texts)

                    render_report(student_texts, threshold)
