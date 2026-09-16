import re
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Apache-2.0 licensed (commercial use allowed), small and widely used for sentence similarity
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# A line shared by at least this fraction of submissions is treated as shared template
# text (e.g. assignment instructions) rather than a sign of copying, and is stripped
# before comparison. Kept high so a small group of students copying each other isn't
# mistaken for everyone sharing a template.
COMMON_LINE_THRESHOLD = 0.7

# Fragments shorter than this rarely explain a similarity score on their own
# (e.g. a lone "Introduction" heading), so they are excluded as match evidence.
SENTENCE_MIN_LENGTH = 15

# How many matching sentence pairs to surface as evidence per flagged student pair
EVIDENCE_TOP_N = 3

# Semantic (meaning-based) similarity runs hotter than literal TF-IDF overlap even
# between unrelated sentences, so it needs a higher floor before being shown as evidence
MIN_EVIDENCE_SCORE = 55.0

# Below this share of non-empty lines being real sentences, a submission is probably a
# table, checklist, or rubric-style form rather than free-written prose (e.g. a trainer's
# marking guide), which produces misleading similarity and AI-detection scores
MIN_PROSE_SENTENCE_RATIO = 0.2

# A sentence this semantically close to another counts as "the same sentence" when
# checking how many students share it (catches assignment questions students copy into
# their own submission, even if formatting/wording differs slightly between them)
COMMON_SENTENCE_SIMILARITY = 0.9

# Share of students needed for a sentence to count as a shared prompt rather than
# evidence of copying between them. Lower than COMMON_LINE_THRESHOLD: in practice,
# students inconsistently delete the assignment brief from their own document before
# submitting (observed as low as ~55% in a real class), so requiring the same 70%
# bar as an exact-line match would let a widely-shared prompt through as "evidence"
COMMON_SENTENCE_THRESHOLD = 0.4


def _load_semantic_model() -> SentenceTransformer:
    # Prefer the local cache first: some antivirus filter drivers (e.g. Norton) block the
    # lock file used during the online freshness check, even when the model is already cached
    try:
        return SentenceTransformer(SEMANTIC_MODEL_NAME, local_files_only=True)
    except OSError:
        # Not cached yet (first run): fall back to a normal online download
        return SentenceTransformer(SEMANTIC_MODEL_NAME)


# Load the model once at import time so repeated calls don't reload it
_semantic_model = _load_semantic_model()


def strip_shared_lines(student_texts: dict[str, str]) -> dict[str, str]:
    """Remove lines that most submissions share verbatim (e.g. assignment instructions or a
    standard academic-integrity declaration), so downstream analysis only sees each
    student's own content. Public so both similarity comparison and AI detection can use it."""
    lines_by_student = {
        name: [line.strip() for line in text.splitlines() if line.strip()]
        for name, text in student_texts.items()
    }

    # Skip with too few submissions: not enough students to tell a shared template
    # apart from a small group copying each other
    if len(lines_by_student) < 3:
        return student_texts

    line_counts: dict[str, int] = {}
    for lines in lines_by_student.values():
        for line in set(lines):
            line_counts[line] = line_counts.get(line, 0) + 1

    n = len(lines_by_student)
    common_lines = {line for line, count in line_counts.items() if count / n >= COMMON_LINE_THRESHOLD}

    return {
        name: "\n".join(line for line in lines if line not in common_lines)
        for name, lines in lines_by_student.items()
    }


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks, dropping fragments too short to be useful evidence.

    Titles and headings (e.g. "The Importance of Recycling in Modern Society") are excluded by
    requiring terminal punctuation: on a shared assignment topic, everyone's title tends to be
    structurally similar, which would otherwise look like a strong semantic match on its own.
    """
    pieces = (piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", text))
    return [piece for piece in pieces if len(piece) >= SENTENCE_MIN_LENGTH and piece.endswith((".", "!", "?"))]


def looks_like_prose(text: str) -> bool:
    """Heuristic check for whether a submission is free-written prose.

    A rubric, checklist, or table-heavy form (e.g. a trainer's marking guide) has few
    real sentences relative to its line count, and produces misleading similarity and
    AI-detection scores if treated like a normal essay.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return len(_split_sentences(text)) / len(lines) >= MIN_PROSE_SENTENCE_RATIO


def _embed_students(student_texts: dict[str, str]) -> dict[str, tuple[list[str], object]]:
    """Split each student's text into sentences and embed them once, for reuse across every pair."""
    embedded = {}
    for name, text in student_texts.items():
        sentences = _split_sentences(text)
        embeddings = _semantic_model.encode(sentences, normalize_embeddings=True) if sentences else None
        embedded[name] = (sentences, embeddings)
    return embedded


def _remove_common_sentences(
    sentence_data: dict[str, tuple[list[str], object]],
) -> dict[str, tuple[list[str], object]]:
    """Drop sentences that show up near-identically across most students.

    Students commonly copy the assignment's own questions/checklist items into their
    submission before answering them. Those repeated prompts are identical in meaning
    across everyone's submission (even if formatting differs slightly), so left in place
    they look exactly like copying between students.
    """
    students = list(sentence_data.keys())
    if len(students) < 3:
        return sentence_data

    entries: list[tuple[str, int]] = []  # (student name, index within that student's sentence list)
    embeddings_list = []
    for name in students:
        sentences, embeddings = sentence_data[name]
        for idx in range(len(sentences)):
            entries.append((name, idx))
            embeddings_list.append(embeddings[idx])

    if not embeddings_list:
        return sentence_data

    matrix = np.vstack(embeddings_list)
    sim_matrix = matrix @ matrix.T
    n_students = len(students)

    keep_indices: dict[str, list[int]] = {name: [] for name in students}
    for i, (owner, local_idx) in enumerate(entries):
        matching_students = {entries[j][0] for j in range(len(entries)) if sim_matrix[i][j] >= COMMON_SENTENCE_SIMILARITY}
        if len(matching_students) / n_students < COMMON_SENTENCE_THRESHOLD:
            keep_indices[owner].append(local_idx)

    result = {}
    for name in students:
        sentences, embeddings = sentence_data[name]
        kept = keep_indices[name]
        result[name] = (
            [sentences[i] for i in kept],
            embeddings[kept] if kept else None,
        )
    return result


def _semantic_similarity(
    sentences_a: list[str], embeddings_a, sentences_b: list[str], embeddings_b
) -> tuple[float, list[tuple[str, str, float]]]:
    """Meaning-based similarity between two already-embedded texts, robust to paraphrasing.

    Returns an overall percentage (how much of each text has a close semantic match in the
    other) plus the sentence pairs that matched most closely, as evidence.
    """
    if embeddings_a is None or embeddings_b is None:
        return 0.0, []

    # Embeddings are normalized, so the dot product is the cosine similarity
    sim_matrix = embeddings_a @ embeddings_b.T

    best_for_a = sim_matrix.max(axis=1)
    best_for_b = sim_matrix.max(axis=0)
    overall_score = float((best_for_a.mean() + best_for_b.mean()) / 2 * 100)

    matches = [
        (sentences_a[i], sentences_b[j], float(sim_matrix[i][j] * 100))
        for i, j in enumerate(sim_matrix.argmax(axis=1))
    ]
    matches = [match for match in matches if match[2] >= MIN_EVIDENCE_SCORE]
    matches.sort(key=lambda match: match[2], reverse=True)
    return overall_score, matches[:EVIDENCE_TOP_N]


def calculate_similarities(
    student_texts: dict[str, str], threshold: float = 40.0
) -> list[tuple[str, str, float, list[tuple[str, str, float]]]]:
    """Compare every pair of students' texts and return pairs with similarity >= threshold(%),
    each with its strongest matching sentences as evidence.

    The reported score is the higher of two measures: literal TF-IDF overlap (catches
    copy-pasted text) and semantic sentence similarity (catches paraphrased copying that
    TF-IDF would miss), so a pair is flagged if either method finds a strong match.
    """
    students = list(student_texts.keys())

    # No comparison needed with fewer than 2 students
    if len(students) < 2:
        return []

    # Vectorize all students' (line-stripped) texts using TF-IDF
    stripped_texts = strip_shared_lines(student_texts)
    texts = [stripped_texts[name] for name in students]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    tfidf_sim = cosine_similarity(tfidf_matrix)

    # Built from the original text, not the line-stripped version: stripping lines
    # first would remove a shared sentence for whichever students had it on a line by
    # itself, leaving only a minority remnant and understating how widely it's shared
    sentence_data = _remove_common_sentences(_embed_students(student_texts))

    results: list[tuple[str, str, float, list[tuple[str, str, float]]]] = []

    # Iterate over every unique student pair (A-B) and keep only pairs at or above the threshold
    for i, j in combinations(range(len(students)), 2):
        name_a, name_b = students[i], students[j]
        literal_score = tfidf_sim[i][j] * 100

        sentences_a, embeddings_a = sentence_data[name_a]
        sentences_b, embeddings_b = sentence_data[name_b]
        semantic_score, evidence = _semantic_similarity(sentences_a, embeddings_a, sentences_b, embeddings_b)

        score = max(literal_score, semantic_score)
        if score >= threshold:
            results.append((name_a, name_b, score, evidence))

    return results
