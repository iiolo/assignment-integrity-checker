from transformers import pipeline

# Apache-2.0 licensed (commercial use allowed), maintained by Fakespot (Mozilla)
MODEL_NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"

# Load the model once at import time so repeated calls don't reload it
_classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None,
    truncation=True,
    max_length=512,
)


def check_ai_generated(text: str) -> float:
    """Run the local RoBERTa classifier and return the AI-generated probability (0~100%)."""
    scores = _classifier(text)[0]

    # Find the "AI" label's score among the returned label/score pairs
    ai_score = next(item["score"] for item in scores if item["label"] == "AI")

    return ai_score * 100
