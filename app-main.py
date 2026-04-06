# College Sentiment & Sales Insights – Streamlit Prototype
# -------------------------------------------------------
# Postmortem Shala Style: End-to-end app to
# 1) Upload audio + CSV
# 2) Transcribe audio (Whisper or Vosk)
# 3) Merge with CRM logs
# 4) Sentiment analysis (pretrained OR train custom if labels exist)
# 5) Analytics + Recommendations in Streamlit
#
# How to run (first time):
# -------------------------------------------------------
# 1) Install system deps: ffmpeg (for Whisper). On Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg
# 2) Python deps:
#    pip install streamlit pandas numpy scikit-learn transformers torch plotly python-dateutil openai-whisper vosk wordcloud
#    (torch may need a platform-specific install; see https://pytorch.org/ if necessary)
# 3) Run app:
#    streamlit run app.py
#
# Notes:
# - Whisper downloads models automatically on first run. Choose tiny/base/small/medium for speed vs accuracy.
# - Vosk is fully offline but needs a model; download a model (e.g., small-en) and provide its folder path in the sidebar.
# - Expected CSV columns (flexible mapping provided in UI):
#   student_name, year, tech_stack, location (Noida/Lucknow), remarks, call_id, date (YYYY-MM-DD or any parseable date)
#   Optional: label (positive/neutral/negative) for training a custom model.
# - If label column is present, we train TF-IDF + LogisticRegression and use it; otherwise we fallback to a Hugging Face pretrained pipeline.
import os
import io
import tempfile
from datetime import datetime
from dateutil import parser

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import google.generativeai as genai


#set the Enviroment Variable :-
os.environ["PATH"] += os.pathsep + r"D:\Training\spi\Python-with-Datascience\my-softpro-project\softpro-Analytics\ffmpeg\bin"


# Optional imports are wrapped – they may fail gracefully if not installed
try:
    import whisper  # OpenAI Whisper
except Exception:
    whisper = None

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import wave
except Exception:
    VoskModel = None
    KaldiRecognizer = None

from transformers import pipeline

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="College Sentiment & Feedback Insights", layout="wide")

def toggle_theme():
    config_dir = ".streamlit"
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.toml")
    is_dark = False
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            if 'base = "dark"' in f.read():
                is_dark = True
    new_theme = "light" if is_dark else "dark"
    with open(config_path, "w") as f:
        f.write(f'[theme]\nbase = "{new_theme}"\n')

col_spacer, col_toggle = st.columns([0.88, 0.12])
with col_toggle:
    is_dark_mode = False
    if os.path.exists(".streamlit/config.toml"):
        with open(".streamlit/config.toml", "r") as f:
            if 'base = "dark"' in f.read():
                is_dark_mode = True
    btn_label = "🌞 Light Mode" if is_dark_mode else "🌙 Dark Mode"
    st.button(btn_label, on_click=toggle_theme, use_container_width=True, type="primary")

st.markdown("""
<style>

/* CENTER IMAGE */
div[data-testid="column"] img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 6px;
}

/* BUTTON EXACTLY BELOW IMAGE */
button[kind="secondary"] {
    display: block;
    width: 160px;
    margin-left: auto;
    margin-right: auto;
    border-radius: 10px;
    font-weight: 600;
    background-color: var(--background-color);
    color: var(--text-color);
    border: 2px solid var(--text-color);
    transition: all 0.3s ease-in-out;
}

/* HOVER BLUE */
button[kind="secondary"]:hover {
    background-color: #2b6cb0 !important;
    color: white !important;
    border-color: #2b6cb0 !important;
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* ANIMATED TOGGLE BUTTON STYLING (PRIMARY BUTTON) */
button[kind="primary"] {
    border-radius: 25px !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    background: linear-gradient(135deg, #6e8efb, #a777e3) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15) !important;
}

button[kind="primary"]:hover {
    transform: scale(1.05) translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(110, 142, 251, 0.5) !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# APP HEADER 
# -----------------------------
import base64

def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_image_base64("gnct logo.jpg")
st.markdown(
    f"""
    <div style="text-align:center; margin-bottom:25px;">
        <img src="data:image/jpeg;base64,{logo_base64}" width="90" style="margin-bottom:12px;">
        <h1 style="margin:0;">GREATER NOIDA COLLEGE SENTIMENT ANALYSIS</h1>
        <p style="margin-top:6px; color:gray; font-size:15px;">
            Audio + CRM Logs → Transcript → Sentiment → Insights → Recommendations
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# -------- CARD IMAGES (ADD HERE) --------
audio_img = load_image_base64("audio_image.jpeg")
text_img = load_image_base64("text_image.jpeg")
crm_img = load_image_base64("crm_image.jpeg")
csv_img = load_image_base64("full_image.jpeg")

st.markdown("## Select Feedback Input Mode")

# ---------- INPUT MODE STATE ----------
if "input_mode" not in st.session_state:
    st.session_state.input_mode = ""


#cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.image("audio_image.jpeg", width=180)
    cls = "mode-btn-active" if st.session_state.input_mode == "Audio Feedback" else "mode-btn"
    if st.button("Audio Feedback", key="audio"):
        st.session_state.input_mode = "Audio Feedback"
    

with c2:
    st.image("text_image.jpeg", width=180)
    cls = "mode-btn-active" if st.session_state.input_mode == "Text Feedback" else "mode-btn"
    if st.button("Text Feedback", key="text"):
        st.session_state.input_mode = "Text Feedback"
    

with c3:
    st.image("crm_image.jpeg", width=180)
    cls = "mode-btn-active" if st.session_state.input_mode == "CRM Text Log" else "mode-btn"
    if st.button("CRM Text Log", key="crm"):
        st.session_state.input_mode = "CRM Text Log"
    

with c4:
    st.image("full_image.jpeg", width=180)
    cls = "mode-btn-active" if st.session_state.input_mode == "Full CSV Analytics" else "mode-btn"
    if st.button("Full CSV Analytics", key="csv"):
        st.session_state.input_mode = "Full CSV Analytics"
    


#input_mode = st.session_state.input_mode

# -----------------------------
# 1) Upload Data (MODE-WISE)
# -----------------------------
audio_files = None
csv_file = None
crm_txt = None
user_text = ""

#if input_mode == "Audio Feedback":
if st.session_state.input_mode == "Audio Feedback":
    st.subheader("Upload Call Recordings")
    audio_files = st.file_uploader(
        "Upload call recordings (any audio file)",
        accept_multiple_files=True
    )

#elif input_mode == "Text Feedback":
elif st.session_state.input_mode == "Text Feedback":
    st.subheader("Text Feedback Analysis")
    user_text = st.text_area(
        "Enter Student Feedback",
        height=180,
        placeholder="Example: I faced issues during admission counselling..."
    )
# -----------------------------
# Text Feedback Sentiment
# -----------------------------

#elif input_mode == "CRM Text Log":
elif st.session_state.input_mode == "CRM Text Log":
    st.subheader("CRM / Admission Log Analysis")
    crm_txt = st.file_uploader(
        "Upload CRM log (.txt file)",
        type=["txt"]
    )



#elif input_mode == "Full CSV Analytics":
elif st.session_state.input_mode == "Full CSV Analytics":
    st.subheader("Upload CSV Logs")
    csv_file = st.file_uploader(
        "Upload CSV logs (student, year, course, remarks, etc.)",
        type=["csv"]
    )
    
# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Settings")
asr_engine = st.sidebar.selectbox("ASR Engine (Audio → Text)", ["Whisper", "Vosk (offline)"])
if asr_engine == "Whisper":
    whisper_size = st.sidebar.selectbox("Whisper model size", ["tiny", "base", "small", "medium"], index=1)
else:
    vosk_model_dir = st.sidebar.text_input("Vosk model directory (unzipped)", value="")

st.sidebar.markdown("---")
use_pretrained = st.sidebar.checkbox("Force Pretrained Sentiment (skip training even if labels exist)", value=False)

st.sidebar.markdown("---")
st.sidebar.write("**Export**")
save_intermediate = st.sidebar.checkbox("Save processed CSV", value=True)

st.sidebar.markdown("---")
st.sidebar.write("**Gemini Integration**")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", value="AIzaSyAgiAnCqXEO-JUAI5l-M4xtYk10zRQwKBI", placeholder="Enter your API key...")
enhance_transcript = st.sidebar.checkbox("Enhance ASR Transcripts via Gemini", value=False)
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    if whisper is None:
        raise RuntimeError("Whisper not installed. pip install openai-whisper and ensure ffmpeg is present.")
    return whisper.load_model(model_size)

@st.cache_resource(show_spinner=False)
def load_vosk(model_dir: str):
    if not model_dir or not os.path.isdir(model_dir):
        raise RuntimeError("Valid Vosk model directory not provided.")
    if VoskModel is None:
        raise RuntimeError("Vosk not installed. pip install vosk")
    return VoskModel(model_dir)

@st.cache_resource(show_spinner=False)
def load_hf_pipeline():
    # Fast, widely used binary sentiment model
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
@st.cache_resource(show_spinner=False)
def load_zero_shot():
    
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

@st.cache_resource(show_spinner=False)
def train_sklearn_sentiment(texts: pd.Series, labels: pd.Series):
    # labels expected as strings: positive/neutral/negative (case-insensitive is handled)
    y = labels.astype(str).str.lower().replace({
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
        "n": "negative",
        "p": "positive"
    })
    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    report = classification_report(y_test, y_pred, output_dict=False)
    return vectorizer, clf, report


def safe_parse_date(x):
    if pd.isna(x):
        return None
    try:
        return parser.parse(str(x), dayfirst=False, yearfirst=True)
    except Exception:
        return None


def transcribe_with_whisper(audio_bytes: bytes, model, filename: str) -> str:
    if not audio_bytes:
        return "[Empty audio]"

    suffix = os.path.splitext(filename)[1]
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        text = result.get("text", "").strip()
        return text if text else "[No speech detected]"
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def transcribe_with_vosk(audio_bytes: bytes, model, filename: str) -> str:
    # Vosk expects WAV PCM 16k mono. We'll try to coerce using wave if already wav; otherwise rely on ffmpeg via whisper isn't possible here.
    # For simplicity: if not WAV, we save and try to open. If not WAV PCM, we warn the user.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
    try:
        if not path.lower().endswith('.wav'):
            return "[Vosk] Please upload WAV PCM audio (16k mono) or use Whisper for auto-conversion."
        wf = wave.open(path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            return "[Vosk] WAV must be mono 16-bit PCM. Convert your file or use Whisper."
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        text_pieces = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = rec.Result()
                text_pieces.append(res)
        final = rec.FinalResult()
        text_pieces.append(final)
        # Combine naive
        return " ".join(text_pieces)
    except Exception as e:
        return f"[Vosk Error] {e}"
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# -----------------------------
# Batch Sentiment Helper (CSV SPEED FIX)
# -----------------------------
def batch_sentiment_analysis(texts, batch_size=16):
    nlp = load_hf_pipeline()
    sentiments = []
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            results = nlp(
                batch,
                truncation=True,
                padding=True,
                max_length=256
            )
            for r in results:
                sentiments.append(r["label"].lower())
                scores.append(float(r["score"]))
        except Exception:
            for _ in batch:
                sentiments.append("neutral")
                scores.append(0.0)

    return sentiments, scores
# CRM Text Log Sentiment
# -----------------------------
if st.session_state.input_mode == "CRM Text Log" and crm_txt is not None:
    text_data = crm_txt.read().decode("utf-8")

    st.subheader("CRM Log Preview")
    st.text(text_data[:1000])

    nlp = load_hf_pipeline()
    result = nlp(text_data[:4096])[0]

    sentiment = result["label"].lower()
    score = result["score"]

    st.success(f"Sentiment: {sentiment.upper()}")
    st.info(f"Confidence Score: {score:.2f}")

    # Chart
    chart_df = pd.DataFrame({
        "Sentiment": [sentiment.capitalize()],
        "Score": [score]
    })

    fig = px.bar(
        chart_df,
        x="Sentiment",
        y="Score",
        range_y=[0, 1],
        title="CRM Sentiment Confidence"
    )

    st.plotly_chart(fig, use_container_width=True, key="crm_sentiment_chart")

# -----------------------------
# TEXT FEEDBACK SENTIMENT (WORKING)
# -----------------------------
if st.session_state.input_mode == "Text Feedback":

    if user_text.strip():
        st.subheader("Sentiment Result")

        text_lower = user_text.lower()

        complaint_keywords = [
            "fee", "fees", "expensive", "cost", "pricing",
            "not affordable", "too high", "overpriced",
            "roi", "refund", "money"
        ]

        if any(word in text_lower for word in complaint_keywords):
            sentiment = "negative"
            score = 0.90
            st.error("Sentiment: NEGATIVE")
            st.info(f"Confidence Score: {score:.2f}")

        else:
            nlp = load_hf_pipeline()
            result = nlp(user_text[:4096])[0]
            sentiment = result["label"].lower()
            score = result["score"]
            st.success(f"Sentiment: {sentiment.upper()}")
            st.info(f"Confidence Score: {score:.2f}")

        chart_df = pd.DataFrame({
            "Sentiment": [sentiment.capitalize()],
            "Score": [score]
        })

        fig = px.bar(
            chart_df,
            x="Sentiment",
            y="Score",
            range_y=[0, 1],
            title="Text Feedback Sentiment Confidence"
        )

        st.plotly_chart(fig, use_container_width=True, key="text_feedback_sentiment_chart")

        st.subheader("Recommendations")

        if gemini_api_key:
            with st.spinner("Generating AI Recommendation for this student..."):
                try:
                    import time
                    gen_model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"A student provided this feedback which was analyzed as {sentiment.upper()}:\n\n\"{user_text}\"\n\nBased on this, what are 2-3 specific, actionable steps the college administration should take to address this specific feedback? Keep it brief and format as bullet points."
                    for attempt in range(3):
                        try:
                            response = gen_model.generate_content(prompt)
                            st.markdown(response.text)
                            break
                        except Exception as e:
                            if "429" in str(e) or "Quota" in str(e):
                                if attempt < 2:
                                    time.sleep(30)
                                else:
                                    raise e
                            else:
                                raise e
                except Exception as e:
                    st.error(f"Gemini Recommendation generation failed: {e}")
        else:
            if sentiment == "negative":
                st.warning("Action Required")
                st.markdown("""
                - Contact the student immediately  
                - Address fee / timing / service issues  
                - Improve counselling clarity
                """)

            elif sentiment == "positive":
                st.success("Positive Feedback")
                st.markdown("""
                - Maintain service quality  
                - Use this feedback as testimonial  
                - Encourage referrals
                """)

            else:
                st.info("Neutral Feedback")
                st.markdown("""
                - Ask follow-up questions  
                - Provide clearer information  
                - Improve explanation
                """)

    else:
        st.info("Please enter some text to analyze sentiment.")
# -----------------------------
# File Uploaders
# -----------------------------
df = None
merged = None

# -----------------------------
# Load DataFrame + Column Mapping
# -----------------------------
if st.session_state.input_mode == "Full CSV Analytics" and csv_file is not None:

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        df = pd.read_csv(csv_file, encoding="latin-1")

    # FORCE REMOVE transcript_text FROM BASE DATAFRAME
    if "transcript_text" in df.columns:
        df.drop(columns=["transcript_text"], inplace=True)
    st.success(f"CSV loaded with shape {df.shape}")

    # -------- RAW CSV DATA (DISPLAY ONLY, transcript removed visually) --------
    st.subheader("Raw CSV Data (As Uploaded)")

    raw_df = df.copy()
    if "transcript_text" in raw_df.columns:
        raw_df.drop(columns=["transcript_text"], inplace=True)

    st.dataframe(raw_df, use_container_width=True)

    # -------- COMBINED TEXT (EXCLUDE transcript_text COMPLETELY) --------
    text_cols = [
        col for col in df.select_dtypes(include="object").columns
        if col != "transcript_text"
    ]

    if text_cols:
        df["combined_text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
    else:
        df["combined_text"] = ""

    # -------- PROCESSED CSV DATA --------
    st.subheader("Processed CSV Data")

    nlp = load_hf_pipeline()
    processed_df = df.copy()

    sentiments = []
    confidence_scores = []

    for txt in processed_df["combined_text"].fillna(""):
        try:
            r = nlp(txt[:4096])[0]
            sentiments.append(r["label"].lower())
            confidence_scores.append(float(r.get("score", 0)))
        except Exception:
            sentiments.append("neutral")
            confidence_scores.append(0.0)

    processed_df["sentiment_score"] = sentiments
    processed_df["confidence_score"] = confidence_scores

    # FORCE REMOVE transcript_text FROM FINAL OUTPUT
    if "transcript_text" in processed_df.columns:
        processed_df.drop(columns=["transcript_text"], inplace=True)

    st.dataframe(processed_df, use_container_width=True)

    merged = processed_df
    
if merged is not None and "transcript_text" in merged.columns:
    merged.drop(columns=["transcript_text"], inplace=True)

# -----------------------------
# Transcribe Audio
# -----------------------------
transcripts = []
if audio_files and st.session_state.input_mode == "Audio Feedback" and st.session_state.input_mode!= "Full CSV Analytics":
    st.subheader("2) Transcribe Audio")

    if asr_engine == "Whisper":
        whisper_model = load_whisper(whisper_size)

    prog = st.progress(0)

    for i, f in enumerate(audio_files):   # INSIDE
        audio_bytes = f.read()
        suffix = os.path.splitext(f.name)[1] or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = whisper_model.transcribe(tmp_path)
            text = result.get("text", "").strip()
            if not text:
                text = "[No speech detected]"
            elif enhance_transcript and gemini_api_key:
                try:
                    import time
                    gen_model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"Improve the sentence quality and fix any ASR transcription mistakes in the following text. Return ONLY the improved text:\n\n{text}"
                    for attempt in range(3):
                        try:
                            response = gen_model.generate_content(prompt)
                            if response.text:
                                text = response.text.strip()
                            break
                        except Exception as e:
                            if "429" in str(e) or "Quota" in str(e):
                                if attempt < 2:
                                    time.sleep(30)
                                else:
                                    raise e
                            else:
                                raise e
                except Exception as e:
                    st.warning(f"Gemini enhancement error: {e}")
        except Exception as e:
            text = f"[Whisper error: {e}]"
        finally:
            os.remove(tmp_path)

        transcripts.append({
            "call_id": os.path.splitext(f.name)[0],
            "transcript_text": text
        })

        prog.progress(int(((i + 1) / len(audio_files)) * 100))

    st.success(f"Transcribed {len(transcripts)} file(s)")
if transcripts:
    df_tr = pd.DataFrame(transcripts)
else:
    df_tr = pd.DataFrame(columns=["call_id", "transcript_text"])  # empty
    # ALWAYS create merged for Audio Feedback (THIS RESTORES CHARTS)
if st.session_state.input_mode == "Audio Feedback" and df is None:
    merged = pd.DataFrame({
        "call_id": df_tr["call_id"],
        "student_name": None,
        "year": None,
        "tech_stack": None,
        "location": None,
        "remarks": "",
        "date": None,
        "label": None,
        "date_parsed": None,
        "transcript_text": df_tr["transcript_text"]
    })

    merged["combined_text"] = merged["transcript_text"].astype(str)

if (
    st.session_state.input_mode == "Audio Feedback"
    and df is None
    and not df_tr.empty
):
    st.subheader("Sentiment & Objection Analysis (Audio Only)")
    # Load models ONCE
    nlp = load_hf_pipeline()
    zero_shot = load_zero_shot()

    # ALWAYS define lists BEFORE loop
    sentiments = []
    scores = []
    objections = []

    objection_labels = [
        "Fees / Pricing",
        "Timing / Schedule",
        "Placement / Career",
        "Location / Travel",
        "Course Content",
        "General Query"
    ]

    for txt in df_tr["transcript_text"].fillna(""):
        # ---------- SENTIMENT ----------
        try:
            r = nlp(txt[:4096])[0]
            sentiments.append(r["label"].lower())
            scores.append(float(r.get("score", 0)))
        except Exception:
            sentiments.append("neutral")
            scores.append(0.0)

        # ---------- OBJECTION (AI BASED) ----------
        if txt.strip() == "" or txt.startswith("["):
            objections.append("No Feedback")
        else:
            try:
                z = zero_shot(txt, candidate_labels=objection_labels)
                objections.append(z["labels"][0])
            except Exception:
                objections.append("General Query")

    # Attach to dataframe
    df_tr["sentiment"] = sentiments
    df_tr["confidence_score"] = scores
    df_tr["objection_type"] = objections

    # Prepare merged
    merged = df_tr.copy()
    merged["combined_text"] = merged["transcript_text"]
    merged["sentiment_score"] = merged["confidence_score"]

    st.dataframe(merged, use_container_width=True)

   

    # -----------------------------
# Charts (FIXED – Audio Only)
# -----------------------------
if( st.session_state.input_mode == "Audio Feedback"
    and merged is not None
    and not merged.empty
    and "objection_type" in merged.columns):
    st.subheader("Sentiment & Objection Charts")

    col1, col2 = st.columns(2)

    # PIE CHART – Sentiment
    with col1:
        fig_sent = px.pie(
            merged,
            names="sentiment",
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_sent, use_container_width=True, key="audio_sentiment_pie")

    # BAR CHART – Objections
    with col2:
        if "objection_type" in merged.columns:
            fig_obj = px.bar(
               merged,
               x="objection_type",
               title="Objection Type Distribution"
            )
            st.plotly_chart(fig_obj, use_container_width=True, key="audio_objection_bar")
        else:
            st.info("Objection analysis is available only for Audio Feedback mode.")

    # Summary 
    overall = merged["sentiment"].mode()[0]
    st.success(f"Overall Sentiment: {overall.upper()}")
    st.info(f"Average Confidence Score: {merged['confidence_score'].mean():.2f}")

    

#-------------------------------
# Sentiment: Train or Pretrained
# -----------------------------

if st.session_state.input_mode == "Full CSV Analytics" and merged is not None and not merged.empty:
    st.subheader("4) Sentiment Analysis")
    can_train = ("label" in merged.columns) and merged["label"].notna().any() and not use_pretrained

    if can_train:
        st.write("Training custom TF-IDF + LogisticRegression on provided labels…")
        with st.spinner("Training model…"):
            try:
                vectorizer, clf, report = train_sklearn_sentiment(merged["combined_text"].fillna("") , merged["label"])
                st.text("Classification report (hold-out test):\n" + report)
                # Predict using trained model
                Xall = vectorizer.transform(merged["combined_text"].fillna(""))
                merged["sentiment"] = clf.predict(Xall)
                merged["sentiment_score"] = np.nan  # not provided by sklearn model
                model_used = "custom_sklearn"
            except Exception as e:
                st.error(f"Training failed: {e}. Falling back to pretrained pipeline.")
                can_train = False

    if not can_train:
        with st.spinner("Running pretrained sentiment model…"):
            nlp = load_hf_pipeline()
            preds = []
            scores = []
            for txt in merged["combined_text"].fillna(""):
                try:
                    r = nlp(txt[:4096])[0]  # limit length for speed
                    label = r["label"].lower()
                    # Map to positive/neutral/negative (HF model is binary pos/neg)
                    if label == "positive":
                        preds.append("positive")
                    elif label == "negative":
                        preds.append("negative")
                    else:
                        preds.append(label)
                    scores.append(float(r.get("score", np.nan)))
                except Exception:
                    preds.append("neutral")
                    scores.append(np.nan)
            merged["sentiment"] = preds
            merged["sentiment_score"] = scores
            model_used = "hf_distilbert"

    st.success(f"Sentiment computed using: {model_used}")

    # -----------------------------
    # Analytics
    # -----------------------------
    st.subheader("5) Analytics")
    colA, colB, colC = st.columns(3)
    with colA:
        fig = px.pie(merged, names="sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True, key="csv_sentiment_pie")
    with colB:
        if "location" in merged.columns:
            fig2 = px.bar(merged.fillna({"location": "Unknown"}), x="location", color="sentiment", title="Sentiment by Location")
            st.plotly_chart(fig2, use_container_width=True, key="csv_location_bar" )
    with colC:
        if "tech_stack" in merged.columns:
            fig3 = px.bar(merged.fillna({"tech_stack": "Unknown"}), x="tech_stack", color="sentiment", title="Sentiment by Tech Stack")
            st.plotly_chart(fig3, use_container_width=True, key="csv_tech_stack_bar")

    # Time trend
    if "date_parsed" in merged.columns and merged["date_parsed"].notna().any():
        temp = merged.copy()
        temp["month"] = temp["date_parsed"].dt.to_period("M").astype(str)
        ts = temp.groupby(["month", "sentiment"]).size().reset_index(name="count")
        fig4 = px.line(ts, x="month", y="count", color="sentiment", markers=True, title="Monthly Sentiment Trend")
        st.plotly_chart(fig4, use_container_width=True, key="csv_monthly_trend")

    # Top Negative Keywords (TF-IDF on negative rows)
    st.markdown("### Top Negative Keywords")
    neg = merged[merged["sentiment"] == "negative"]["combined_text"].dropna()
    if len(neg) >= 3:
        vec = TfidfVectorizer(max_features=50, stop_words="english")
        X = vec.fit_transform(neg)
        # Sum TF-IDF across docs
        sums = np.asarray(X.sum(axis=0)).ravel()
        vocab = np.array(vec.get_feature_names_out())
        kw_df = pd.DataFrame({"keyword": vocab, "score": sums}).sort_values("score", ascending=False).head(20)
        fig5 = px.bar(kw_df, x="keyword", y="score", title="Top Negative Keywords (TF-IDF)")
        st.plotly_chart(fig5, use_container_width=True, key="csv_negative_keywords")
    else:
        st.info("Not enough negative samples to extract keywords.")

    # -----------------------------
    # Recommendations (rule + keyword heuristics & Gemini AI)
    # -----------------------------
    st.subheader("6) Recommendations")

    def get_ai_recommendations(df_: pd.DataFrame):
        try:
            total_records = len(df_)
            sentiment_counts = df_['sentiment'].value_counts().to_dict()
            
            summary_prompt = f"You are an expert consultant for a college. Analyze the following student feedback sentiment data and provide 3-5 highly actionable recommendations to improve the college's services, courses, and administration.\n\n"
            summary_prompt += f"Total Records: {total_records}\n"
            summary_prompt += f"Overall Sentiment Distribution: {sentiment_counts}\n"
            
            if "location" in df_.columns:
                loc_sent = df_.groupby("location")["sentiment"].value_counts().to_dict()
                summary_prompt += f"Sentiment by Location: {loc_sent}\n"
            
            if "tech_stack" in df_.columns:
                tech_sent = df_.groupby("tech_stack")["sentiment"].value_counts().to_dict()
                summary_prompt += f"Sentiment by Tech Stack: {tech_sent}\n"
            
            neg_text = " ".join(df_[df_["sentiment"] == "negative"]["combined_text"].dropna().astype(str).tolist())[:4000]
            if neg_text:
                summary_prompt += f"\nSample of negative feedback from students:\n{neg_text}\n"
                
            summary_prompt += "\nPlease format your response nicely with markdown bullet points."
            
            import time
            gen_model = genai.GenerativeModel('gemini-2.5-flash')
            for attempt in range(3):
                try:
                    response = gen_model.generate_content(summary_prompt)
                    return response.text
                except Exception as e:
                    if "429" in str(e) or "Quota" in str(e):
                        if attempt < 2:
                            time.sleep(30)
                        else:
                            raise e
                    else:
                        raise e
        except Exception as e:
            st.error(f"Gemini Recommendation generation failed: {e}")
            return None

    def gen_recos(df_: pd.DataFrame):
        recos = []
        # Overall negativity
        total = len(df_)
        negc = (df_["sentiment"] == "negative").sum()
        if total > 0 and negc/total > 0.4:
            recos.append("Overall negative sentiment is high (>40%). Consider immediate coaching for counselors and revising scripts.")

        # By location
        if "location" in df_.columns:
            by_loc = df_.groupby("location")["sentiment"].apply(lambda s: (s=="negative").mean()).sort_values(ascending=False)
            for loc, ratio in by_loc.items():
                if pd.notna(loc) and ratio >= 0.35:
                    recos.append(f"{loc}: Negative ratio {ratio:.0%}. Trial: reduce fees, add evening/weekend batches, or senior counselor callbacks.")

        # By stack
        if "tech_stack" in df_.columns:
            by_stack = df_.groupby("tech_stack")["sentiment"].apply(lambda s: (s=="negative").mean()).sort_values(ascending=False)
            for stack, ratio in by_stack.items():
                if pd.notna(stack) and ratio >= 0.35:
                    tips = {
                        "python": "emphasize job outcomes with case studies, add mini capstone demo",
                        "java": "offer installment plans, highlight placement partners",
                        "mern": "show live project repos and alumni testimonials",
                        "ai": "clarify math prerequisites and provide bridge modules"
                    }
                    extra = ""
                    k = str(stack).lower()
                    for key, val in tips.items():
                        if key in k:
                            extra = "; " + val
                            break
                    recos.append(f"{stack}: Negative ratio {ratio:.0%}. Address objections via FAQs{extra}.")

        # Keyword-based
        text_all = " ".join(df_.get("combined_text", pd.Series(dtype=str)).dropna().astype(str).tolist()).lower()
        if any(k in text_all for k in ["fee", "fees", "price", "cost", "expensive"]):
            recos.append("Many fee-related objections → try scholarships, limited-time discounts, or EMI options.")
        if any(k in text_all for k in ["time", "timing", "slot", "schedule", "evening", "weekend"]):
            recos.append("Timing objections → add evening/weekend batches and flexible slots.")
        if any(k in text_all for k in ["location", "distance", "noida", "lucknow", "commute"]):
            recos.append("Location/commute issues → promote online/hybrid option and campus transfer flexibility.")
        if any(k in text_all for k in ["doubt", "support", "mentor", "teacher", "faculty"]):
            recos.append("Learning support concerns → advertise mentorship hours, doubt-solving sessions, and WhatsApp/Slack groups.")
        if any(k in text_all for k in ["job", "placement", "interview", "resume"]):
            recos.append("Career outcomes focus → showcase placement stats, resume/interview prep workshops.")
        return recos

    if gemini_api_key:
        with st.spinner("Generating AI Recommendations using Gemini..."):
            ai_recos = get_ai_recommendations(merged)
            if ai_recos:
                st.markdown(ai_recos)
            else:
                # Fallback to rule-based
                recommendations = gen_recos(merged)
                if recommendations:
                    for r in recommendations:
                        st.markdown(f"- {r}")
                else:
                    st.info("No strong recommendations detected. With more data, insights will improve.")
    else:
        # Use rule-based recommendations
        recommendations = gen_recos(merged)
        if recommendations:
            for r in recommendations:
                st.markdown(f"- {r}")
        else:
            st.info("No strong recommendations detected. With more data, insights will improve.")

    # -----------------------------
    # Downloads
    # -----------------------------
    st.subheader("7) Export")
    if save_intermediate:
        out_csv = merged.copy()
        out_buf = io.StringIO()
        out_csv.to_csv(out_buf, index=False)
        st.download_button("Download processed CSV", data=out_buf.getvalue(), file_name=f"softpro_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

#else:
    #st.info("Upload at least a CSV or audio to proceed.")

# End of app
