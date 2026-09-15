from transformers import pipeline

# Apache-2.0 licensed (commercial use allowed), maintained by Fakespot (Mozilla)
MODEL_NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"


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
