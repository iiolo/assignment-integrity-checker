"""Local Streamlit UI for checking student assignments for plagiarism and AI-generated text."""
import os
import tempfile

import streamlit as st

from src.ai_detection_check import check_ai_generated
from src.extract_text import extract_text
from src.similarity import calculate_similarities

st.set_page_config(page_title="Assignment Integrity Checker", layout="wide")
st.title("Assignment Integrity Checker")
st.caption(
    "Upload each student's assignment file. The file name (without extension) "
    "is used as the student's name."
)


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


uploaded_files = st.file_uploader(
    "Student assignment files (.docx, .pdf)",
    type=["docx", "pdf"],
    accept_multiple_files=True,
)

threshold = st.slider("Similarity alert threshold (%)", min_value=0, max_value=100, value=40)

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
