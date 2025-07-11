from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

app = Flask(__name__)

# === Load movie data ===
df = pd.read_csv("movies.csv")

# Handle missing description
if 'description' not in df.columns:
    if 'overview' in df.columns:
        df['description'] = df['overview']
    else:
        df['description'] = "No description available."

df['genres'] = df['genres'].apply(lambda x: x.split(','))
df['main_genre'] = df['genres'].apply(lambda x: x[0])
df['combined'] = df['title'] + " " + df['description']

# === Train movie genre model (for similarity only) ===
if not os.path.exists("tfidf_vectorizer.pkl"):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=1)
    X = tfidf_vectorizer.fit_transform(df['combined'])
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
else:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# === Train mood-to-genre model ===
mood_df = pd.read_csv("mood_genre.csv")  # should contain columns: mood, genre

if not os.path.exists("mood_genre_model.pkl"):
    mood_vectorizer = TfidfVectorizer()
    X_mood = mood_vectorizer.fit_transform(mood_df['mood'])
    y_mood = mood_df['genre']

    mood_model = LogisticRegression()
    mood_model.fit(X_mood, y_mood)

    joblib.dump(mood_model, 'mood_genre_model.pkl')
    joblib.dump(mood_vectorizer, 'mood_vectorizer.pkl')
else:
    mood_model = joblib.load('mood_genre_model.pkl')
    mood_vectorizer = joblib.load('mood_vectorizer.pkl')

# === Routes ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    mood_raw = request.form['mood']

    # Predict genre from mood input
    mood_vec = mood_vectorizer.transform([mood_raw])
    predicted_genre = mood_model.predict(mood_vec)[0]

    # Filter movies based on predicted genre
    filtered = df[df['main_genre'] == predicted_genre]

    # Handle empty case
    if filtered.empty:
        return render_template('index.html', genre=predicted_genre, recommendations=[], message=f"No movies found for genre: {predicted_genre}")

    filtered_vecs = tfidf_vectorizer.transform(filtered['combined'])

    # Compare mood input with filtered movies
    mood_tfidf_vec = tfidf_vectorizer.transform([mood_raw])
    sim_scores = cosine_similarity(mood_tfidf_vec, filtered_vecs).flatten()
    top_indices = sim_scores.argsort()[-5:][::-1]

    recommendations = []
    for i in top_indices:
        movie = filtered.iloc[i]
        recommendations.append({
            'title': movie['title'],
            'description': movie['description'],
            'poster_url': movie['poster_url']
        })

    return render_template('index.html', genre=predicted_genre, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
