# ml_api/moderation.py

from transformers import pipeline
import json
import os

moderator = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

def analyze_post(content: str) -> dict:
    """
    Analyze the given text for toxicity.
    Returns a dictionary with toxicity score and label.
    """
    result = moderator(content)[0]
    
    top = max(result, key=lambda x: x["score"])
    label = top["label"]
    score = top["score"]

    if label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] and score > 0.4:
        status = "block" if score > 0.75 else "warning"
    else:
        status = "safe"

    return {
        "text": content,
        "label": label,
        "score": round(score, 3),
        "status": status
    }
