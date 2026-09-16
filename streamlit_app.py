"""Local Streamlit UI for checking student assignments for plagiarism and AI-generated text."""
import html
import os
import tempfile

import streamlit as st

from src.ai_detection_check import check_ai_generated_document, class_median_score, grade_ai_scores
from src.extract_text import extract_text
from src.similarity import calculate_similarities, looks_like_prose, strip_shared_lines
from src.teams_auth import get_access_token, sign_out
from src.teams_client import (
    download_resource,
    get_assignments,
    get_class_members,
    get_my_classes,
    get_submission_resources,
    get_submissions,
)

st.set_page_config(page_title="Assignment Integrity Checker", page_icon="🎓", layout="wide")

# Keep the intro/setup area narrower than the full wide layout so it doesn't look
# sparse; the report further down uses the full width, where tables need the room.
_intro_left, INTRO_COL, _intro_right = st.columns([1, 3, 1])
with INTRO_COL:
    st.title("🎓 Assignment Integrity Checker")
    st.caption(
        "Check student submissions for similarity between each other and for AI-generated "
        "text — everything runs locally on this computer."
    )
    st.caption("1) Provide submissions　→　2) Automatic analysis　→　3) Review the report")


def cluster_similar_students(similarity_results: list[tuple[str, str, float, list]]) -> list[list[str]]:
    """Group students into clusters connected by any flagged similarity pair (transitively):
    if A matches B and B matches C, all three belong in one group even if A and C weren't
    directly flagged against each other."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for student_a, student_b, _, _ in similarity_results:
        union(student_a, student_b)

    groups: dict[str, list[str]] = {}
    for name in parent:
        groups.setdefault(find(name), []).append(name)
    return list(groups.values())


def highlight_text(
    text: str, ai_passages: list[tuple[str, float]], similarity_passages: list[tuple[str, str, float]]
) -> str:
    """Wrap AI-suspect and similarity-matched passages in colored <mark> tags.

    Each mark gets a title attribute (shown as a tooltip on hover) explaining why it's
    highlighted: which student it matched for a similarity passage, or the AI-likeness
    score for an AI-suspect passage.

    The text is HTML-escaped first, and every passage is escaped the same way before
    being searched for, so a passage matches correctly even if it contains characters
    like '&' or '<', and no literal HTML/script in a submission can affect the page.
    """
    escaped = html.escape(text)
    for passage, other_name, score in similarity_passages:
        escaped_passage = html.escape(passage)
        if escaped_passage:
            tooltip = html.escape(f"Matched with {other_name} ({score:.0f}%)")
            escaped = escaped.replace(
                escaped_passage,
                f'<mark title="{tooltip}" style="background:#ff8a80;color:#000">{escaped_passage}</mark>',
            )
    for passage, score in ai_passages:
        escaped_passage = html.escape(passage)
        if escaped_passage:
            tooltip = html.escape(f"{score:.0f}% AI-like")
            escaped = escaped.replace(
                escaped_passage,
                f'<mark title="{tooltip}" style="background:#ffcc80;color:#000">{escaped_passage}</mark>',
            )
    # Each remaining line is one paragraph's worth of content; giving each its own
    # block with spacing reads far better than one wall of text joined by plain line breaks
    paragraphs = [line for line in escaped.split("\n") if line]
    return "".join(f'<p style="margin:0 0 0.8em 0">{paragraph}</p>' for paragraph in paragraphs)


def save_upload_to_temp(uploaded_file) -> str:
    """Save an in-memory uploaded file to disk so extract_text() can read it by path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def render_report(student_texts: dict[str, str], threshold: int) -> None:
    """Run similarity + AI detection on the given texts and render a risk-sorted report."""
    with st.spinner("Comparing submissions for similarity..."):
        similarity_results = calculate_similarities(student_texts, threshold=threshold)

    # Each student's highest similarity score against anyone else
    best_similarity: dict[str, tuple[str, float]] = {}
    for student_a, student_b, score, _ in similarity_results:
        for name, other in ((student_a, student_b), (student_b, student_a)):
            if name not in best_similarity or score > best_similarity[name][1]:
                best_similarity[name] = (other, score)

    names = list(student_texts.keys())

    # Shared boilerplate (assignment instructions, a standard academic-integrity
    # declaration, etc.) shouldn't count toward a student's own AI-generated score
    texts_for_ai_check = strip_shared_lines(student_texts)

    scores: dict[str, float] = {}
    paragraphs_by_student: dict[str, list[tuple[str, float]]] = {}
    progress = st.progress(0.0, text="Running AI detection...")
    for i, student_name in enumerate(names):
        score, paragraphs = check_ai_generated_document(texts_for_ai_check[student_name])
        scores[student_name] = score
        paragraphs_by_student[student_name] = paragraphs
        progress.progress((i + 1) / len(names), text=f"Running AI detection... ({i + 1}/{len(names)})")
    progress.empty()

    # Graded relative to the class, not by a fixed cutoff alone: see grade_ai_scores()
    grades = grade_ai_scores(scores)
    ai_data = {name: (scores[name], grades[name], paragraphs_by_student[name]) for name in names}

    # The raw score alone is misleading (formal writing scores almost as high as real AI
    # text), so the UI leads with how far above the class's own baseline a score sits
    class_median = class_median_score(scores)

    def ai_display(name: str) -> str:
        score = ai_data[name][0]
        if class_median is None:
            return f"{score:.1f}% (raw score, too few submissions for a class comparison)"
        delta = score - class_median
        return f"{delta:+.0f}%p vs class median (raw score {score:.1f}%)"

    # Rubrics, checklists, and other table-heavy documents produce misleading scores
    # if scored as if they were free-written prose, so flag them instead of hiding it
    non_prose_students = {name for name, text in student_texts.items() if not looks_like_prose(text)}
    if non_prose_students:
        st.warning(
            "These submissions don't look like free-written text (they read more like a "
            "table, checklist, or form) so their similarity/AI scores below may be misleading: "
            + ", ".join(sorted(non_prose_students))
        )

    def risk_level(name: str) -> str:
        # A flagged similarity pair or a High AI grade needs a look; Medium AI needs a
        # second glance; anything else is routine and doesn't need to take up space
        if name in best_similarity or ai_data[name][1] == "High":
            return "red"
        if ai_data[name][1] == "Medium":
            return "yellow"
        return "green"

    order = {"red": 0, "yellow": 1, "green": 2}
    sorted_names = sorted(
        names,
        key=lambda n: (order[risk_level(n)], -ai_data[n][0], -best_similarity.get(n, ("", 0))[1]),
    )
    badge = {"red": "🔴", "yellow": "🟡", "green": "🟢"}

    flagged_names = [n for n in sorted_names if risk_level(n) != "green"]
    routine_names = [n for n in sorted_names if risk_level(n) == "green"]

    # Students linked by any chain of flagged similarity pairs (e.g. A-B and B-C flagged
    # means A, B, C are shown together as one group instead of two separate pair rows)
    similarity_groups = [g for g in cluster_similar_students(similarity_results) if len(g) >= 2]
    grouped_names = {name for group in similarity_groups for name in group}

    def group_pairs(group: list[str]) -> list[tuple[str, str, float, list]]:
        group_set = set(group)
        return [r for r in similarity_results if r[0] in group_set and r[1] in group_set]

    tab_summary, tab_similarity, tab_ai, tab_original = st.tabs(
        ["🔍 Summary", "📑 Similarity details", "🤖 AI detection details", "📄 Original text"]
    )

    with tab_summary:
        st.caption(
            "AI grade is relative to this class's own scores, not a fixed cutoff: this "
            "detector also scores formal or non-native-English writing almost as high as "
            "real AI text, so only students who score well above their classmates are flagged."
        )

        if not flagged_names:
            st.success("No students were flagged for similarity or AI-generated text.")

        ai_only_flagged = [n for n in flagged_names if n not in grouped_names]

        if similarity_groups:
            st.markdown("##### 🔴 Similarity groups")
            for group in sorted(similarity_groups, key=lambda g: -max(score for _, _, score, _ in group_pairs(g))):
                with st.expander(f"Group: {', '.join(group)} ({len(group)} students)"):
                    st.markdown("**Why flagged:**")
                    for student_a, student_b, score, _ in sorted(group_pairs(group), key=lambda r: -r[2]):
                        st.write(f"- {student_a} <-> {student_b}: {score:.1f}% similarity")
                    st.markdown("**AI detection for this group:**")
                    for name in group:
                        st.write(f"- {name}: {ai_data[name][1]} ({ai_display(name)})")
                    prose_flagged = [name for name in group if name in non_prose_students]
                    if prose_flagged:
                        st.write("⚠️ Not free-written text: " + ", ".join(prose_flagged))
                    st.markdown(
                        "**Worth a closer look:** the matching sentences (Similarity details tab) "
                        "and the highlighted passages (Original text tab)."
                    )

        if ai_only_flagged:
            st.markdown("##### 🟡 AI-flagged (no similarity match)")
            for name in ai_only_flagged:
                score, grade, paragraphs = ai_data[name]
                with st.expander(f"{badge[risk_level(name)]} {name}"):
                    st.markdown("**Why flagged:**")
                    st.write(f"- AI grade {grade}: {ai_display(name)}")
                    if name in non_prose_students:
                        st.write("⚠️ Not free-written text (table/checklist/form) - this score may be misleading.")
                    st.markdown("**Worth a closer look:** the suspect passages (AI detection details tab).")

        if routine_names:
            with st.expander(f"🟢 No issues found ({len(routine_names)})"):
                for name in routine_names:
                    _, grade, _ = ai_data[name]
                    st.write(f"{name}: AI {grade} ({ai_display(name)})")

        st.warning(
            "These results are for reference only and should not be used as the sole basis "
            "for a plagiarism or AI-use decision."
        )

    with tab_similarity:
        if similarity_results:
            st.caption(
                "Students linked by any flagged pair (even indirectly, e.g. A-B and B-C) "
                "are grouped together, one foldable section per group."
            )
            for group in sorted(similarity_groups, key=lambda g: -max(score for _, _, score, _ in group_pairs(g))):
                pairs = sorted(group_pairs(group), key=lambda r: -r[2])
                with st.expander(f"Group: {', '.join(group)} ({len(group)} students)"):
                    st.table(
                        [
                            {"Student A": a, "Student B": b, "Similarity (%)": f"{score:.1f}"}
                            for a, b, score, _ in pairs
                        ]
                    )
                    for student_a, student_b, score, matches in pairs:
                        st.markdown(f"**{student_a} <-> {student_b} ({score:.1f}%): matching sentences**")
                        if matches:
                            for sentence_a, sentence_b, sentence_score in matches:
                                st.markdown(f"- {sentence_score:.0f}% match")
                                st.write(f"　{student_a}: {sentence_a}")
                                st.write(f"　{student_b}: {sentence_b}")
                        else:
                            st.write("No single matching sentence stands out; the overlap is spread across the text.")
        else:
            st.write(f"No pairs found at or above the {threshold}% threshold.")

    with tab_ai:
        if class_median is not None:
            st.caption(f"Class median raw AI score: {class_median:.1f}%. Scores below compare each student to it.")
        st.table(
            [
                {
                    "Student": name,
                    "Grade": ai_data[name][1],
                    "vs class median / raw score": ai_display(name),
                }
                for name in names
            ]
        )
        st.caption("Only Medium/High grade students are expanded below, to keep this section short.")
        for name in names:
            score, grade, paragraphs = ai_data[name]
            if grade == "Low":
                continue
            with st.expander(f"{name} ({grade}): most AI-like passages"):
                if paragraphs:
                    for paragraph, p_score in paragraphs:
                        st.markdown(f"**{p_score:.0f}% AI-like**")
                        st.write(paragraph)
                else:
                    st.write("No paragraph was long enough to score individually.")

    with tab_original:
        st.caption(
            "Shared boilerplate (assignment instructions, a standard declaration, etc.) is "
            "removed so only each student's own writing is shown. &nbsp;&nbsp; "
            "🟧 orange = AI-suspect passage (Medium/High grade students only) &nbsp;&nbsp; "
            "🟥 red = matched with another student's submission (hover to see who)"
        )
        for name in names:
            score, grade, paragraphs = ai_data[name]
            ai_passages = paragraphs if grade != "Low" else []
            similarity_passages = []
            for student_a, student_b, pair_score, matches in similarity_results:
                if name not in (student_a, student_b):
                    continue
                other_name = student_b if student_a == name else student_a
                for sentence_a, sentence_b, _ in matches:
                    own_sentence = sentence_a if student_a == name else sentence_b
                    similarity_passages.append((own_sentence, other_name, pair_score))

            with st.expander(f"{badge[risk_level(name)]} {name}"):
                text = texts_for_ai_check[name]
                if not text:
                    st.write("(no student-written text remains after removing shared boilerplate)")
                else:
                    st.markdown(highlight_text(text, ai_passages, similarity_passages), unsafe_allow_html=True)


_setup_left, SETUP_COL, _setup_right = st.columns([1, 3, 1])
with SETUP_COL:
    mode = st.radio(
        "How would you like to get submissions?",
        ["📤 Upload files", "🔗 Fetch from Teams"],
        horizontal=True,
    )

    with st.expander("⚙️ Advanced settings"):
        threshold = st.slider("Similarity alert threshold (%)", min_value=0, max_value=100, value=40)
        st.caption("Student pairs at or above this similarity score are flagged. 40% is a reasonable default.")

st.divider()

if mode == "📤 Upload files":
    _up_left, UP_COL, _up_right = st.columns([1, 3, 1])
    with UP_COL:
        st.caption(
            "Upload each student's assignment file. The file name (without extension) "
            "is used as the student's name."
        )
        uploaded_files = st.file_uploader(
            "Student assignment files (.docx, .pptx, .pdf, .odt, .hwpx, .rtf, .txt)",
            type=["docx", "pptx", "pdf", "odt", "hwpx", "rtf", "txt"],
            accept_multiple_files=True,
        )
        run_clicked = st.button("Run check", disabled=not uploaded_files)

    if run_clicked:
        student_texts: dict[str, str] = {}
        failures: list[str] = []

        # Extract text from every uploaded file, keyed by student name (file name)
        progress = st.progress(0.0, text="Extracting text from files...")
        for i, uploaded_file in enumerate(uploaded_files):
            student_name = os.path.splitext(uploaded_file.name)[0]
            temp_path = save_upload_to_temp(uploaded_file)
            try:
                student_texts[student_name] = extract_text(temp_path)
            except (ValueError, FileNotFoundError) as exc:
                failures.append(f"{uploaded_file.name}: {exc}")
            finally:
                os.remove(temp_path)
            progress.progress(
                (i + 1) / len(uploaded_files), text=f"Extracting text from files... ({i + 1}/{len(uploaded_files)})"
            )
        progress.empty()

        if failures:
            with st.expander(f"⚠️ {len(failures)} file(s) could not be read"):
                for failure in failures:
                    st.write(failure)

        render_report(student_texts, threshold)

else:
    _t_left, TEAMS_COL, _t_right = st.columns([1, 3, 1])

    if "graph_token" not in st.session_state:
        with TEAMS_COL:
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
        class_choice = None
        assignment_choice = None
        fetch_clicked = False

        with TEAMS_COL:
            status_col, signout_col = st.columns([4, 1])
            with status_col:
                st.caption("✅ Signed in to Microsoft")
            with signout_col:
                if st.button("Sign out", use_container_width=True):
                    sign_out()
                    st.session_state.pop("graph_token", None)
                    st.session_state.pop("classes", None)
                    st.rerun()

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

                    preview_col, fetch_col = st.columns(2)
                    with preview_col:
                        preview_clicked = st.button("🔍 Preview one submission", use_container_width=True)
                    with fetch_col:
                        fetch_clicked = st.button("▶️ Fetch submissions", type="primary", use_container_width=True)

                    if preview_clicked:
                        preview_submissions = get_submissions(token, class_choice["id"], assignment_choice["id"])
                        preview_submitted = [s for s in preview_submissions if s["submittedDateTime"]]
                        if not preview_submitted:
                            st.warning("No submissions found for this assignment yet.")
                        else:
                            with st.spinner("Downloading a sample submission..."):
                                preview_resources = get_submission_resources(
                                    token, class_choice["id"], assignment_choice["id"], preview_submitted[0]["id"]
                                )
                                preview_texts = []
                                for resource in preview_resources:
                                    suffix = os.path.splitext(resource["displayName"])[1]
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                        temp_path = tmp.name
                                    try:
                                        download_resource(token, resource["fileUrl"], temp_path)
                                        preview_texts.append(extract_text(temp_path))
                                    except (ValueError, FileNotFoundError, RuntimeError) as exc:
                                        st.error(f"Failed to read {resource['displayName']}: {exc}")
                                    finally:
                                        os.remove(temp_path)

                            if not preview_texts:
                                st.warning("This submission had no readable file attached.")
                            else:
                                preview_text = "\n".join(preview_texts)
                                if looks_like_prose(preview_text):
                                    st.success(
                                        "This looks like free-written text (essay-style) - good for this tool."
                                    )
                                else:
                                    st.warning(
                                        "This doesn't look like free-written text (reads more like a table, "
                                        "checklist, or form, e.g. a marking guide) - similarity/AI results on "
                                        "this assignment may be misleading."
                                    )
                                snippet = preview_text[:1500]
                                st.text(snippet + ("..." if len(preview_text) > len(snippet) else ""))

        if fetch_clicked:
            members = get_class_members(token, class_choice["id"])
            name_by_id = {m["id"]: m["displayName"] for m in members}

            submissions = get_submissions(token, class_choice["id"], assignment_choice["id"])
            submitted = [s for s in submissions if s["submittedDateTime"]]

            student_texts: dict[str, str] = {}
            failures: list[str] = []
            progress = st.progress(0.0, text=f"Downloading 0/{len(submitted)} submissions...")
            for i, sub in enumerate(submitted):
                recipient_id = sub["recipientId"]
                # Roster lookup misses students who submitted but later left the class
                student_name = name_by_id.get(
                    recipient_id, f"Unknown student (not in roster, ...{recipient_id[-6:]})"
                )
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
                        failures.append(f"{student_name} - {resource['displayName']}: {exc}")
                    finally:
                        os.remove(temp_path)

                if texts:
                    student_texts[student_name] = "\n".join(texts)

                progress.progress(
                    (i + 1) / len(submitted), text=f"Downloading {i + 1}/{len(submitted)} submissions..."
                )
            progress.empty()

            if failures:
                with st.expander(f"⚠️ {len(failures)} file(s) could not be read"):
                    for failure in failures:
                        st.write(failure)

            render_report(student_texts, threshold)
