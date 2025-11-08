
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

def build_user_profiles(users_df, posts_df, interactions_df):
    post_embeddings = model.encode(posts_df["content"].tolist(), convert_to_numpy=True)
    user_profiles = {}

    for user_id in users_df["id"]:
        user_interactions = interactions_df[interactions_df["user_id"] == user_id]
        if user_interactions.empty:
            user_profiles[user_id] = np.zeros(post_embeddings.shape[1])
            continue
        post_ids = user_interactions["post_id"].tolist()
        user_vectors = post_embeddings[[pid - 1 for pid in post_ids]]
        user_profiles[user_id] = np.mean(user_vectors, axis=0)

    return user_profiles, post_embeddings


def recommend_posts(user_id, users_df, posts_df, interactions_df, top_k=3):
    user_profiles, post_embeddings = build_user_profiles(users_df, posts_df, interactions_df)
    user_vec = user_profiles.get(user_id)
    if user_vec is None:
        return []

    sims = cosine_similarity([user_vec], post_embeddings)[0]
    seen = interactions_df[interactions_df["user_id"] == user_id]["post_id"].tolist()

    recommendations = [
        (pid + 1, score)
        for pid, score in enumerate(sims)
        if (pid + 1) not in seen
    ]
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_posts = recommendations[:top_k]

    return posts_df[posts_df["id"].isin([p[0] for p in top_posts])][["id", "content"]].to_dict(orient="records")
