from flask import Flask, request, jsonify, render_template
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load pre-trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Spotify API setup (replace with your credentials)
SPOTIPY_CLIENT_ID = "34680a2d45e8448785d658c934338a6e"
SPOTIPY_CLIENT_SECRET = "2b42c8324b4943b6aac6fbacd905497e"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# Function to clean user input
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Predict sentiment
def predict_sentiment(user_input):
    cleaned_input = clean_text(user_input)
    input_tfidf = tfidf.transform([cleaned_input])
    sentiment = model.predict(input_tfidf)[0]
    return sentiment

# Define genre mapping based on sentiment
genre_mapping = {
    "Happy": "Pop",
    "Sad": "Electronic",
    "Angry": "Rock",
    "Relaxed": "Jazz",
    "Excited": "Hip-Hop"
}

# Get song based on sentiment
def get_song_by_sentiment(sentiment):
    genre = genre_mapping.get(sentiment, "Pop")
    query = f"{genre} song"
    results = sp.search(q=query, limit=1, type='track')
    if results['tracks']['items']:
        song = results['tracks']['items'][0]
        return song['artists'][0]['name'], song['name'], song['external_urls']['spotify']
    return "Unknown", "No Song Found", "#"

# Flask routes
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        sentiment = predict_sentiment(user_input)
        artist, song, link = get_song_by_sentiment(sentiment)
        return render_template("index.html", sentiment=sentiment, artist=artist, song=song, link=link, user_input=user_input)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
