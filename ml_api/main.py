import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from ml_api.recommendation import recommend_posts
from ml_api.moderation import analyze_post
import pandas as pd
import uvicorn

app = FastAPI(title="College Community ML API")

class PostInput(BaseModel):
    text: str

class RecommendationInput(BaseModel):
    user_id: int
    users: List[Dict]
    posts: List[Dict]
    interactions: List[Dict]

@app.get("/")
def root():
    return {"message": "Welcome to the College Community ML API 🚀"}

@app.post("/moderate-post/")
def moderate_post(input: PostInput):
    return analyze_post(input.text)

@app.post("/recommend-posts/")
def get_recommendations(data: RecommendationInput):
    users_df = pd.DataFrame(data.users)
    posts_df = pd.DataFrame(data.posts)
    interactions_df = pd.DataFrame(data.interactions)
    recs = recommend_posts(data.user_id, users_df, posts_df, interactions_df)
    return {"user_id": data.user_id, "recommendations": recs}


