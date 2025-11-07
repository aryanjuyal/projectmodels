# ml_api/moderation.py
from transformers import pipeline

# 💡 Lightweight sentiment-based moderation model
moderator = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None
)

def analyze_post(content: str) -> dict:
    """
    Analyze the given text for negativity/toxicity.
    Returns a dictionary with score and moderation status.
    """
    result = moderator(content)[0]           # [{'label': 'NEGATIVE', 'score': 0.93}, ...]
    top = max(result, key=lambda x: x["score"])
    label = top["label"]
    score = top["score"]

    # Decide moderation status
    status = "block" if label == "NEGATIVE" and score > 0.8 else "safe"

    return {
        "text": content,
        "label": label,
        "score": round(score, 3),
        "status": status
    }
