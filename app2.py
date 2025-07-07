import streamlit as st
import re
import string
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# === Download NLTK resources ===
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# === Load model dan asset ===
model = load_model('lstm_cnn_sentiment_model_80_20.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# === NLP tools ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

important_stopwords = {
    "not", "no", "nor", "never", "none", "nobody", "nothing", "neither",
    "n't", "don't", "didn't", "doesn't", "won't", "wouldn't", "can't",
    "cannot", "couldn't", "shouldn't", "mustn't", "isn't", "aren't",
    "wasn't", "weren't", "haven't", "hasn't", "hadn't", "without",
    "barely", "hardly", "scarcely", "very", "extremely", "really", "so",
    "too", "quite", "but", "although", "however", "yet"
}
custom_stop_words = stop_words - important_stopwords

# === Slang dictionary ===
slang_dict = {
    "u": "you", "r": "are", "ur": "your", "btw": "by the way", "idk": "i do not know",
    "lol": "laughing out loud", "omg": "oh my god", "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing", "brb": "be right back", "gtg": "got to go",
    "imo": "in my opinion", "imho": "in my humble opinion", "fyi": "for your information",
    "tbh": "to be honest", "smh": "shaking my head", "np": "no problem", "jk": "just kidding",
    "nvm": "never mind", "bff": "best friend forever", "dm": "direct message",
    "tldr": "too long did not read", "wth": "what the heck", "ikr": "i know right",
    "ya": "yeah", "thx": "thanks", "ty": "thank you", "plz": "please", "bc": "because",
    "cuz": "because", "tho": "though", "k": "okay", "ok": "okay", "hbu": "how about you",
    "wyd": "what are you doing", "wbu": "what about you", "rn": "right now", "bday": "birthday",
    "gr8": "great", "luv": "love", "xoxo": "hugs and kisses", "yall": "you all",
    "sick": "awesome", "dope": "cool", "lit": "amazing", "fam": "friends", "salty": "upset",
    "shade": "disrespect", "tea": "gossip", "yolo": "you only live once", "fomo": "fear of missing out",
    "vibes": "feelings", "ive": "i have"
}

# === Preprocessing functions ===
def cleaningText(text):
    emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002500-\U00002BEF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def casefoldingText(text):
    return text.lower()

def standard_slangwords(text):
    tokens = word_tokenize(text)
    standardized_tokens = [slang_dict.get(t.lower(), t) for t in tokens]
    return " ".join(standardized_tokens)

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(tokens):
    return [word for word in tokens if word.lower() not in custom_stop_words]

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def lemmatizationText(tokens):
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in pos_tags]

def preprocess_pipeline(text, max_len=100):
    cleaned = cleaningText(text)
    casefolded = casefoldingText(cleaned)
    standardized = standard_slangwords(casefolded)
    tokens = tokenizingText(standardized)
    filtered = filteringText(tokens)
    lemmatized = lemmatizationText(filtered)
    joined = ' '.join(lemmatized)
    padded = pad_sequences(tokenizer.texts_to_sequences([joined]), maxlen=max_len, padding='post')
    return padded, {
        "cleaned": cleaned,
        "casefolded": casefolded,
        "standardized": standardized,
        "tokens": tokens,
        "filtered": filtered,
        "lemmatized": lemmatized,
        "joined": joined
    }

# === Streamlit UI ===
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("💬 Sentiment Analysis - LSTM+CNN")

with st.form("input_form"):
    user_input = st.text_area("Masukkan komentar YouTube:", "")
    submitted = st.form_submit_button("Analisis")

if submitted:
    if not user_input.strip():
        st.error("Komentar tidak boleh kosong!")
    else:
        padded_input, steps = preprocess_pipeline(user_input)
        prediction = model.predict(padded_input, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)
        sentiment = label_encoder.inverse_transform(predicted_class)[0]

        # Confidence scores
        labels = ['positive', 'neutral', 'negative']
        scores = {
            label.capitalize(): prediction[0][label_encoder.transform([label])[0]]
            for label in labels
        }

        st.subheader("📌 Hasil Prediksi")
        st.success(f"Sentimen (Model): **{sentiment.upper()}**")
        for label, score in scores.items():
            st.write(f"**{label}**: {score:.4f}")

        st.subheader("📊 Confidence per Label")
        fig, ax = plt.subplots()
        ax.bar(scores.keys(), scores.values(), color=['#2ecc71', '#3498db', '#e74c3c'])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Confidence")
        ax.set_title("Confidence Score")
        st.pyplot(fig)

        with st.expander("🔍 Detail Preprocessing"):
            for k, v in steps.items():
                st.text(f"{k}: {v}")
        
        with st.expander("📚 Token Frequency from Tokenizer"):
            word_counts = tokenizer.word_counts
            sorted_tokens = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            tokens, freqs = zip(*sorted_tokens)
            fig2, ax2 = plt.subplots()
            ax2.barh(tokens[::-1], freqs[::-1], color='#3498db')
            ax2.set_title("Top 20 Frequent Tokens")
            st.pyplot(fig2)
