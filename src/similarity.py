import re
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

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
MIN_EVIDENCE_SCORE = 85.0

# Literal copying is measured with word n-gram "shingles" (Broder 1997) and the containment
# of the shorter text in the longer one. Same-topic answers share vocabulary but rarely share
# N words in a row, so unlike unigram TF-IDF this does not score honest answers as copying.
NGRAM_SIZE = 5

# A shingle found in at least this share of submissions is shared wording (quoted question,
# stock phrase), not evidence of copying between two particular students
COMMON_NGRAM_THRESHOLD = 0.4

# Below this many shingles a text is too short for the containment ratio to mean anything
MIN_SHINGLES = 30

# Runs of shared shingles shorter than this many words aren't shown as copied passages
MIN_PASSAGE_WORDS = 8

# Paraphrase check needs at least this many sentences in the shorter text to be meaningful
MIN_SENTENCES_FOR_PARAPHRASE = 5

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


def find_shared_lines(student_texts: dict[str, str]) -> set[str]:
    """Lines that most submissions share verbatim (assignment instructions, a standard
    academic-integrity declaration). Public so the report can show the teacher exactly what
    was excluded: automatic removal can be wrong, so a human should be able to check it."""
    lines_by_student = {
        name: [line.strip() for line in text.splitlines() if line.strip()]
        for name, text in student_texts.items()
    }

    # Skip with too few submissions: not enough students to tell a shared template
    # apart from a small group copying each other
    if len(lines_by_student) < 3:
        return set()

    line_counts: dict[str, int] = {}
    for lines in lines_by_student.values():
        for line in set(lines):
            line_counts[line] = line_counts.get(line, 0) + 1

    n = len(lines_by_student)
    return {line for line, count in line_counts.items() if count / n >= COMMON_LINE_THRESHOLD}


def strip_shared_lines(student_texts: dict[str, str]) -> dict[str, str]:
    """Remove lines that most submissions share verbatim, so downstream analysis only sees each
    student's own content. Public so both similarity comparison and AI detection can use it."""
    common_lines = find_shared_lines(student_texts)
    if not common_lines:
        return student_texts

    return {
        name: "\n".join(line for line in (l.strip() for l in text.splitlines()) if line and line not in common_lines)
        for name, text in student_texts.items()
    }


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def strip_template(student_texts: dict[str, str], template_text: str) -> dict[str, str]:
    """Remove the teacher's blank template (questions, instructions, declarations) from every
    submission so only the students' own answers are compared.

    Unlike strip_shared_lines(), which has to guess what is shared from the submissions
    themselves, this knows exactly what the teacher wrote, so it works for any class size
    and still catches questions a student reworded slightly.
    """
    all_lines = [_normalize_line(line) for line in template_text.splitlines()]
    all_lines = [line for line in all_lines if line]
    # Short template lines (field labels like "Date" or a unit code) only match whole lines;
    # as substrings they'd cut words out of real answers
    exact = set(all_lines)
    template_lines = [line for line in all_lines if len(line) >= SENTENCE_MIN_LENGTH]
    if not template_lines:
        return student_texts

    # Longest first so a long template line is removed before any shorter line inside it
    contained = sorted(set(template_lines), key=len, reverse=True)
    template_embeddings = _semantic_model.encode(template_lines, normalize_embeddings=True)

    result = {}
    for name, text in student_texts.items():
        lines = [_normalize_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]

        # Template wording typed on the same line as the answer ("Q1. Why? Because...")
        cleaned = []
        for line in lines:
            if line in exact:
                continue
            for template_line in contained:
                if template_line in line:
                    line = line.replace(template_line, " ")
            line = _normalize_line(line)
            if line:
                cleaned.append(line)

        # Reworded copies of a template line (e.g. a question retyped with minor edits)
        candidates = [i for i, line in enumerate(cleaned) if len(line) >= SENTENCE_MIN_LENGTH]
        drop = set()
        if candidates:
            embeddings = _semantic_model.encode([cleaned[i] for i in candidates], normalize_embeddings=True)
            best = (embeddings @ template_embeddings.T).max(axis=1)
            drop = {candidates[k] for k, score in enumerate(best) if score >= COMMON_SENTENCE_SIMILARITY}

        result[name] = "\n".join(line for i, line in enumerate(cleaned) if i not in drop)
    return result


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


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WORD_RE = re.compile(r"[A-Za-z0-9’']+")


def _tokenize(text: str) -> tuple[str, list[tuple[str, int, int]]]:
    """Lower-cased words with their positions in the URL-stripped text. URLs and reference
    lists are shared whenever two students cite the same source, so they are ignored."""
    cleaned = _URL_RE.sub(" ", text)
    return cleaned, [(m.group().lower(), m.start(), m.end()) for m in _WORD_RE.finditer(cleaned)]


def _shingles(tokens: list[tuple[str, int, int]]) -> list[tuple[str, ...]]:
    words = [t[0] for t in tokens]
    return [tuple(words[i : i + NGRAM_SIZE]) for i in range(len(words) - NGRAM_SIZE + 1)]


def _shared_passages(
    cleaned_a: str, tokens_a, shingles_a, cleaned_b: str, tokens_b, shingles_b, shared: set
) -> list[tuple[str, str, float]]:
    """Merge overlapping shared shingles into the longest runs of identical wording
    (the seed-extension step used by PAN text-alignment systems), as copied-passage evidence."""
    first_in_b: dict[tuple[str, ...], int] = {}
    for j, sh in enumerate(shingles_b):
        first_in_b.setdefault(sh, j)

    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(shingles_a):
        if shingles_a[i] in shared:
            j = i
            while j + 1 < len(shingles_a) and shingles_a[j + 1] in shared:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1

    passages = []
    for start, end in runs:
        n_words = end - start + NGRAM_SIZE
        if n_words < MIN_PASSAGE_WORDS:
            continue
        b_start = first_in_b[shingles_a[start]]
        text_a = cleaned_a[tokens_a[start][1] : tokens_a[end + NGRAM_SIZE - 1][2]]
        text_b = cleaned_b[tokens_b[b_start][1] : tokens_b[b_start + n_words - 1][2]]
        passages.append((n_words, text_a, text_b))
    passages.sort(reverse=True)
    return [(a, b, 100.0) for _, a, b in passages[:EVIDENCE_TOP_N]]


def _paraphrase_similarity(
    sentences_a: list[str], embeddings_a, sentences_b: list[str], embeddings_b
) -> tuple[float, list[tuple[str, str, float]]]:
    """Share of the shorter text's sentences that have a near-identical-meaning sentence in
    the other text (catches reworded copying that word matching misses), plus those pairs."""
    if embeddings_a is None or embeddings_b is None:
        return 0.0, []
    if min(len(sentences_a), len(sentences_b)) < MIN_SENTENCES_FOR_PARAPHRASE:
        return 0.0, []

    # Embeddings are normalized, so the dot product is the cosine similarity
    sim_matrix = embeddings_a @ embeddings_b.T
    axis = 1 if len(sentences_a) <= len(sentences_b) else 0
    best = sim_matrix.max(axis=axis)
    score = float((best * 100 >= MIN_EVIDENCE_SCORE).mean() * 100)

    matches = [
        (sentences_a[i], sentences_b[j], float(sim_matrix[i][j] * 100))
        for i, j in enumerate(sim_matrix.argmax(axis=1))
        if sim_matrix[i][j] * 100 >= MIN_EVIDENCE_SCORE
    ]
    matches.sort(key=lambda match: match[2], reverse=True)
    return score, matches[:EVIDENCE_TOP_N]


def calculate_similarities(
    student_texts: dict[str, str], threshold: float = 10.0
) -> list[tuple[str, str, float, list[tuple[str, str, float]]]]:
    """Compare every pair of students' texts and return pairs scoring >= threshold(%),
    each with its strongest matching passages as evidence.

    The score is the higher of two measures: containment of word 5-grams (how much of the
    shorter text is wording also found in the other: copy-paste) and the share of sentences
    that are reworded copies (paraphrase). Wording shared by many students is ignored.
    """
    students = list(student_texts.keys())

    # No comparison needed with fewer than 2 students
    if len(students) < 2:
        return []

    tokenized = {name: _tokenize(student_texts[name]) for name in students}
    shingles = {name: _shingles(tokenized[name][1]) for name in students}
    shingle_sets = {name: set(sh) for name, sh in shingles.items()}

    common: set = set()
    if len(students) >= 3:
        counts: dict[tuple[str, ...], int] = {}
        for sh_set in shingle_sets.values():
            for sh in sh_set:
                counts[sh] = counts.get(sh, 0) + 1
        # At least 3 students: in a small class 40% is only 2, which is exactly what two
        # students copying from each other look like, and would erase the copied passage
        min_students = max(3, COMMON_NGRAM_THRESHOLD * len(students))
        common = {sh for sh, c in counts.items() if c >= min_students}

    sentence_data = _remove_common_sentences(_embed_students(student_texts))

    results: list[tuple[str, str, float, list[tuple[str, str, float]]]] = []

    for i, j in combinations(range(len(students)), 2):
        name_a, name_b = students[i], students[j]
        set_a, set_b = shingle_sets[name_a] - common, shingle_sets[name_b] - common

        literal_score = 0.0
        passages: list[tuple[str, str, float]] = []
        if min(len(set_a), len(set_b)) >= MIN_SHINGLES:
            shared = set_a & set_b
            literal_score = len(shared) / min(len(set_a), len(set_b)) * 100
            if shared:
                passages = _shared_passages(
                    tokenized[name_a][0], tokenized[name_a][1], shingles[name_a],
                    tokenized[name_b][0], tokenized[name_b][1], shingles[name_b], shared,
                )

        sentences_a, embeddings_a = sentence_data[name_a]
        sentences_b, embeddings_b = sentence_data[name_b]
        paraphrase_score, paraphrase_evidence = _paraphrase_similarity(
            sentences_a, embeddings_a, sentences_b, embeddings_b
        )

        score = max(literal_score, paraphrase_score)
        if score >= threshold:
            results.append((name_a, name_b, score, passages if literal_score >= paraphrase_score else paraphrase_evidence))

    return results
