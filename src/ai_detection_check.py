from transformers import pipeline

# Apache-2.0 licensed (commercial use allowed), maintained by Fakespot (Mozilla)
MODEL_NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"

# Paragraphs shorter than this give the classifier too little context for a reliable score
PARAGRAPH_MIN_LENGTH = 40

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


def check_ai_generated_document(text: str, top_n: int = 3) -> tuple[float, list[tuple[str, float]]]:
    """Score a whole submission for AI-generated likelihood.

    Scoring is done per paragraph rather than running the classifier once on the full
    text (the model truncates input at 512 tokens, so a single full-text pass would
    silently ignore anything after the first paragraph or two of a long submission),
    and the document score is the average across all of that student's paragraphs.

    Returns the document-level score plus the top_n most AI-like paragraphs, so the
    report can show exactly which passages scored highest even though they're averaged
    into the overall score.
    """
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= PARAGRAPH_MIN_LENGTH]
    if not paragraphs:
        return check_ai_generated(text), []

    scored = [(paragraph, check_ai_generated(paragraph)) for paragraph in paragraphs]
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
