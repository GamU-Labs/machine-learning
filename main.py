from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="GamU Recommendation API")

print("Memuat model...")
try:
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    df = pd.read_pickle('models/clean_games_df.pkl')
    print("Model berhasil dimuat")
except Exception as e:
    print(f"Gagal load model: {e}")

@app.get("/")
def home():
    return {"message": "Selamat datang di API GamU!"}

@app.get("/recommend")
def get_rekomendasi(judul: str):
    try:
        idx = df[df['title'].str.lower() == judul.lower()].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Game '{judul}' tidak ditemukan.")

    # Hitung similarity
    game_vector = tfidf_matrix[idx]
    sim_scores = cosine_similarity(game_vector, tfidf_matrix).flatten()
    
    # Ambil Top 5
    similar_indices = sim_scores.argsort()[::-1][1:6]
    
    rekomendasi = df.iloc[similar_indices][['title', 'rating', 'desc_sentence', 'tags_clean']].to_dict('records')
    
    return {
        "input_game": judul,
        "status": "success",
        "recommendations": rekomendasi
    }