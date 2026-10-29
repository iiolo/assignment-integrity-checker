from transformers import pipeline

# Apache-2.0 licensed (commercial use allowed), maintained by Fakespot (Mozilla)
MODEL_NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"

# Detectors need real context: the literature finds statistical and fine-tuned detectors need
# roughly 100-200 words to reach their potential, and accuracy drops sharply on sentence-length
# input (Fraser, Dawkins & Kiritchenko, JAIR 2025). Short paragraphs are therefore merged
# (consecutive ones, same student) into chunks of about this many words before scoring.
CHUNK_TARGET_WORDS = 150

# A trailing chunk shorter than this is folded into the previous one instead of scored alone
CHUNK_MIN_WORDS = 60

# A submission with fewer words than this is not graded at all: any score would be noise
MIN_WORDS_FOR_SCORE = 100

# Below this many submissions, comparing a student against classmates isn't meaningful
MIN_CLASS_SIZE_FOR_RELATIVE_GRADING = 3

# A score needs to clear this floor before being graded Medium/High at all, regardless
# of how it compares to classmates
MEDIUM_SCORE_FLOOR = 34.0
HIGH_SCORE_FLOOR = 60.0

# How many points above the class median a score needs to be to count as "standing out"
OUTLIER_MARGIN = 15.0


def _load_classifier():
    # Prefer the local cache first: some antivirus filter drivers (e.g. Norton) block the
    # lock file used during the online freshness check, even when the model is already cached
    kwargs = dict(
        task="text-classification",
        model=MODEL_NAME,
        top_k=None,
        truncation=True,
        max_length=512,
    )
    try:
        return pipeline(local_files_only=True, **kwargs)
    except OSError:
        # Not cached yet (first run): fall back to a normal online download
        return pipeline(**kwargs)


# Load the model once at import time so repeated calls don't reload it
_classifier = _load_classifier()


def check_ai_generated(text: str) -> float:
    """Run the local RoBERTa classifier and return the AI-generated probability (0~100%)."""
    scores = _classifier(text)[0]

    # Find the "AI" label's score among the returned label/score pairs
    ai_score = next(item["score"] for item in scores if item["label"] == "AI")

    return ai_score * 100


def count_words(text: str) -> int:
    return len(text.split())


def _chunk_paragraphs(text: str) -> list[str]:
    """Merge consecutive paragraphs into chunks of about CHUNK_TARGET_WORDS words, so the
    classifier always sees enough context (short answers scored one by one are unreliable,
    while concatenating a student's short passages is known to improve detection)."""
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: list[list[str]] = []
    current: list[str] = []
    words = 0
    for paragraph in paragraphs:
        current.append(paragraph)
        words += count_words(paragraph)
        if words >= CHUNK_TARGET_WORDS:
            chunks.append(current)
            current, words = [], 0
    if current:
        if chunks and words < CHUNK_MIN_WORDS:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return ["\n".join(chunk) for chunk in chunks]


def check_ai_generated_document(text: str, top_n: int = 3) -> tuple[float, list[tuple[str, float]]]:
    """Score a whole submission for AI-generated likelihood.

    Scoring is done per chunk of about CHUNK_TARGET_WORDS words rather than running the
    classifier once on the full text (the model truncates input at 512 tokens, so a single
    full-text pass would silently ignore anything after the first part of a long submission),
    and the document score is the average across that student's chunks.

    Callers should skip submissions under MIN_WORDS_FOR_SCORE words (see count_words): the
    score of a very short text is not reliable.

    Returns the document-level score plus the top_n most AI-like chunks, so the report can
    show exactly which passages scored highest even though they're averaged into the overall
    score.
    """
    chunks = _chunk_paragraphs(text)
    if not chunks:
        return check_ai_generated(text), []

    scored = [(chunk, check_ai_generated(chunk)) for chunk in chunks]
    average_score = sum(score for _, score in scored) / len(scored)
    scored.sort(key=lambda item: item[1], reverse=True)
    return average_score, scored[:top_n]


def _grade_absolute(score: float) -> str:
    if score < MEDIUM_SCORE_FLOOR:
        return "Low"
    if score < HIGH_SCORE_FLOOR + OUTLIER_MARGIN:
        return "Medium"
    return "High"


def class_median_score(scores: dict[str, float]) -> float | None:
    """Return the class's median AI score, or None if there aren't enough submissions
    for a class baseline to mean anything (see MIN_CLASS_SIZE_FOR_RELATIVE_GRADING)."""
    if len(scores) < MIN_CLASS_SIZE_FOR_RELATIVE_GRADING:
        return None
    sorted_scores = sorted(scores.values())
    return sorted_scores[len(sorted_scores) // 2]


def grade_ai_scores(scores: dict[str, float]) -> dict[str, str]:
    """Grade each student's AI-generated probability as Low/Medium/High, relative to
    their own class rather than a fixed cutoff alone.

    This classifier scores confident, formal, or non-native-English writing almost as
    high as real AI-generated text, so a class-wide fixed threshold flags a class doing
    a formal assignment almost entirely, regardless of actual AI use. Comparing each
    score against the class's own median instead means a uniformly formal assignment
    doesn't get everyone flagged, while a student who stands out well above their
    classmates still does. Ties are broken toward Low: commercial detectors (e.g.
    Muhayu's GPT Killer) are deliberately tuned to miss some AI use rather than falsely
    accuse a student who simply writes formally, and this follows the same principle.
    """
    median = class_median_score(scores)
    if median is None:
        return {name: _grade_absolute(score) for name, score in scores.items()}

    grades = {}
    for name, score in scores.items():
        stands_out = score - median >= OUTLIER_MARGIN
        if score >= HIGH_SCORE_FLOOR and stands_out:
            grades[name] = "High"
        elif score >= MEDIUM_SCORE_FLOOR and stands_out:
            grades[name] = "Medium"
        else:
            grades[name] = "Low"
    return grades
