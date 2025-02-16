from flask import Flask, request, render_template
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
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

def get_song_by_sentiment(sentiment, artist_filter=None):
    # Expanded genre mapping with multiple genres per sentiment
    genre_mapping = {
        "Happy": ["pop", "dance", "disco"],
        "Sad": ["blues", "soul", "r&b"],
        "Angry": ["rock", "metal", "punk"],
        "Relaxed": ["classical", "ambient", "jazz"],
        "Excited": ["edm", "house", "techno"]
    }
    
    # Get genres based on sentiment
    genres = genre_mapping.get(sentiment, ["pop"])
    songs = []
    
    # Build query - prioritize artist filter if specified
    if artist_filter and artist_filter.strip():
        query = f"artist:{artist_filter.strip()}"
    else:
        # Use genre filters if no artist specified
        genre_query = " OR ".join([f"tag:{genre}" for genre in genres])
        query = f"({genre_query}) year:2020-2024"
    print(f"Searching Spotify with query: '{query}'")
    
    try:
        results = sp.search(q=query, limit=10, type='track', market='US')
        if results and results['tracks']['items']:
            print(f"Found {len(results['tracks']['items'])} results for query: '{query}'")
            for item in results['tracks']['items']:
                artist = item['artists'][0]['name']
                song_name = item['name']
                link = item['external_urls']['spotify']
                print(f"Found song: {song_name} by {artist}")
                songs.append((artist, song_name, link))
    except Exception as e:
        print(f"Error fetching songs: {e}")
    
    return songs

# Flask routes
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        artist_filter = request.form.get("artist", "").strip()
        print(f"Artist filter: '{artist_filter}'")  # Debug logging
        sentiment = predict_sentiment(user_input)
        songs = get_song_by_sentiment(sentiment, artist_filter)
        print(f"Found {len(songs)} songs")  # Debug logging
        return render_template("index.html", sentiment=sentiment, songs=songs, user_input=user_input)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
