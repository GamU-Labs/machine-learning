from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai.errors import ClientError
import os
import logging
import time

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-3.1-flash-lite"
TOP_N           = 2
RETRY_DELAYS    = [5, 10, 20]   # detik jeda antar retry saat semua key kena rate limit

GEMINI_API_KEYS = [
    k for k in [
        "API KEY",
    ]
    if not k.startswith("GANTI")
]

if not GEMINI_API_KEYS:
    logger.warning("Tidak ada API key Gemini yang valid. Fitur LLM tidak akan berfungsi.")

current_key_index = 0

# ── Load Model ────────────────────────────────────────────────────────────────
logger.info("Memuat model...")
try:
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    df = pd.read_pickle("models/clean_games_df.pkl")
    logger.info("Model berhasil dimuat.")
except Exception as e:
    logger.error(f"Gagal load model: {e}")
    tfidf = tfidf_matrix = df = None


# ── Rekomendasi ───────────────────────────────────────────────────────────────
def get_recommendations(judul: str, top_n: int = TOP_N) -> list | None:
    """Kembalikan list rekomendasi game berdasarkan kemiripan TF-IDF tags."""
    matches = df[df["title"].str.lower() == judul.lower()]
    if matches.empty:
        return None

    idx         = matches.index[0]
    sim_scores  = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][1 : top_n + 1]

    results = df.iloc[top_indices][["title", "rating", "desc_sentence", "tags_clean"]].copy()
    results["similarity_score"] = sim_scores[top_indices]
    return results.to_dict("records")


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(input_game: str, recommendations: list) -> str:
    rec_lines = "\n".join(
        f"{i+1}. {r['title']} (Rating: {r['rating']}, Similarity: {r['similarity_score']:.2f})\n"
        f"   Tags: {r['tags_clean']}\n"
        f"   Deskripsi: {r['desc_sentence']}"
        for i, r in enumerate(recommendations)
    )
    return (
        f'Kamu adalah asisten rekomendasi game yang ramah dan informatif.\n\n'
        f'Pengguna menyukai game bernama "{input_game}".\n'
        f'Sistem rekomendasi telah menemukan {len(recommendations)} game berikut yang mirip:\n\n'
        f'{rec_lines}\n\n'
        f'Tugasmu:\n'
        f'1. Berikan pengantar singkat mengapa game-game ini cocok untuk penggemar "{input_game}".\n'
        f'2. Jelaskan masing-masing game secara ringkas dan menarik (1-2 kalimat per game).\n'
        f'3. Tutup dengan satu kalimat rekomendasi terbaik pilihan kamu.\n\n'
        f'Gunakan Bahasa Indonesia yang santai dan engaging.'
    )


# ── Gemini Client ─────────────────────────────────────────────────────────────
def get_gemini_response(prompt: str) -> str:
    """
    Kirim prompt ke Gemini.
    - Fallback otomatis ke key berikutnya jika kena rate limit (429)
    - Retry dengan jeda jika semua key habis limit sekaligus
    """
    global current_key_index

    if not GEMINI_API_KEYS:
        return "(Tidak ada API key Gemini yang tersedia.)"

    for retry in range(len(RETRY_DELAYS) + 1):
        all_rate_limited = True

        for i in range(len(GEMINI_API_KEYS)):
            idx     = (current_key_index + i) % len(GEMINI_API_KEYS)
            api_key = GEMINI_API_KEYS[idx]

            try:
                logger.info(f"Mencoba API key ke-{idx + 1} dari {len(GEMINI_API_KEYS)}...")
                client   = genai.Client(api_key=api_key)
                response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)

                current_key_index = idx
                logger.info(f"Berhasil menggunakan API key ke-{idx + 1}.")
                return response.text

            except ClientError as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err.upper():
                    logger.warning(f"API key ke-{idx + 1} habis limit (429).")
                elif "401" in err or "403" in err or "API_KEY_INVALID" in err.upper():
                    logger.error(f"API key ke-{idx + 1} tidak valid (401/403).")
                    all_rate_limited = False
                else:
                    logger.error(f"API key ke-{idx + 1} ClientError: {err}")
                    all_rate_limited = False

            except Exception as e:
                logger.error(f"API key ke-{idx + 1} error: {type(e).__name__}: {e}")
                all_rate_limited = False

        if all_rate_limited and retry < len(RETRY_DELAYS):
            jeda = RETRY_DELAYS[retry]
            logger.warning(f"Semua key kena rate limit. Menunggu {jeda} detik (retry {retry + 1})...")
            time.sleep(jeda)
        else:
            break

    logger.error("Semua API key Gemini gagal.")
    return "(Gagal menghubungi Gemini. Periksa log server untuk detail error.)"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return jsonify({"message": "Selamat datang di API GamU!", "status": "ok"})


@app.route("/recommend", methods=["GET"])
def recommend():
    if df is None:
        return jsonify({"error": "Model belum dimuat. Periksa file model."}), 503

    judul = request.args.get("judul", "").strip()
    if not judul:
        return jsonify({"error": "Parameter 'judul' wajib diisi."}), 400

    recommendations = get_recommendations(judul)
    if recommendations is None:
        return jsonify({"error": f"Game '{judul}' tidak ditemukan di database."}), 404

    llm_output = get_gemini_response(build_prompt(judul, recommendations))

    return jsonify({
        "input_game"     : judul,
        "status"         : "success",
        "recommendations": recommendations,
        "llm_response"   : llm_output,
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
