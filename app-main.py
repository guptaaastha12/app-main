# Softpro Sentiment & Sales Insights – Streamlit Prototype
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
#st.set_page_config(page_title="College Sentiment & Feedback Insights", layout="wide")
#st.title("College Sentiment & Feedback Insights")
#st.caption("Audio + CRM logs → Transcripts → Sentiment → Insights → Recommendations")
st.set_page_config(page_title="Softpro Sentiment & Sales Insights", layout="wide")

# Header layout
col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image("gnct logo.jpg", width=90)

with col_title:
    st.markdown("## College Sentiment & Feedback Insights")
    st.caption("Audio + CRM logs → Transcripts → Sentiment → Insights → Recommendations")
#new code adding---------1st
st.markdown("## Select Feedback Input Mode")

input_mode = st.radio(
    "Choose how feedback is provided",
    ["Audio Feedback", "Text Feedback", "CRM Text Log", "Full CSV Analytics"],
    horizontal=True
)
# -----------------------------
# 1) Upload Data (MODE-WISE)
# -----------------------------
audio_files = None
csv_file = None
crm_txt = None
user_text = ""

if input_mode == "Audio Feedback":
    st.subheader("Upload Call Recordings")
    audio_files = st.file_uploader(
        "Upload call recordings (any audio file)",
        accept_multiple_files=True
    )

elif input_mode == "Text Feedback":
    st.subheader("Text Feedback Analysis")
    user_text = st.text_area(
        "Enter student/staff feedback",
        height=180,
        placeholder="Example: I faced issues during admission counselling..."
    )
# -----------------------------
# Text Feedback Sentiment (ADD THIS)
# -----------------------------

elif input_mode == "CRM Text Log":
    st.subheader("CRM / Admission Log Analysis")
    crm_txt = st.file_uploader(
        "Upload CRM log (.txt file)",
        type=["txt"]
    )
    # -----------------------------


elif input_mode == "Full CSV Analytics":
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
        # CRM Text Log Sentiment
# -----------------------------
if input_mode == "CRM Text Log" and crm_txt is not None:
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

    st.plotly_chart(fig, use_container_width=True)
    
        # -----------------------------
# TEXT FEEDBACK SENTIMENT (WORKING)
# -----------------------------
if input_mode == "Text Feedback":
    if user_text.strip():
        st.subheader("Sentiment Result")

        text_lower = user_text.lower()

        complaint_keywords = [
            "fee", "fees", "expensive", "cost", "pricing",
            "not affordable", "too high", "overpriced",
            "roi", "refund", "money"
        ]

        # -------- LAYER 1: HARD COMPLAINT --------
        if any(word in text_lower for word in complaint_keywords):
            sentiment = "negative"
            st.error("Sentiment: NEGATIVE")
            st.info("Detected: Pricing / fee related complaint")

        else:
            # -------- LAYER 2: ZERO-SHOT --------
            zero_shot = load_zero_shot()
            zs = zero_shot(
                user_text,
                candidate_labels=["complaint", "praise", "query", "neutral"]
            )

            top_label = zs["labels"][0]

            if top_label == "complaint":
                st.error("Sentiment: NEGATIVE")
                st.info("Detected: Complaint")

            else:
                # -------- LAYER 3: SENTIMENT MODEL --------
                nlp = load_hf_pipeline()
                result = nlp(user_text[:4096])[0]

                sentiment = result["label"].lower()
                score = result["score"]

                st.success(f"Sentiment: {sentiment.upper()}")
                st.info(f"Confidence Score: {score:.2f}")
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
if input_mode == "Full CSV Analytics":
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            df = pd.read_csv(csv_file, encoding="latin-1")

        st.success(f"CSV loaded with shape {df.shape}")

        st.subheader("Raw CSV Data (As Uploaded)")
        st.dataframe(df, use_container_width=True)

        # Combine all text columns automatically
        text_cols = df.select_dtypes(include="object").columns
        if len(text_cols) > 0:
            df["combined_text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
        else:
            df["combined_text"] = ""
    else:
        df = None
# -----------------------------
# Transcribe Audio
# -----------------------------
transcripts = []
if audio_files and input_mode in ["Audio Feedback", "Full CSV Analytics"]:
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
if input_mode == "Audio Feedback" and df is None:
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
    input_mode == "Audio Feedback"
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
if merged is not None and not merged.empty:
    st.subheader("Sentiment & Objection Charts")

    col1, col2 = st.columns(2)

    # PIE CHART – Sentiment
    with col1:
        fig_sent = px.pie(
            merged,
            names="sentiment",
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # BAR CHART – Objections
    with col2:
        fig_obj = px.bar(
            merged,
            x="objection_type",
            title="Objection Type Distribution"
        )
        st.plotly_chart(fig_obj, use_container_width=True)

    # Summary (SAFE)
    overall = merged["sentiment"].mode()[0]
    st.success(f"Overall Sentiment: {overall.upper()}")
    st.info(f"Average Confidence Score: {merged['confidence_score'].mean():.2f}")

    
# -----------------------------
# Merge Transcripts with CSV
# -----------------------------
# Convert Audio-only feedback into analytics-compatible format

# -----------------------------
# Merge Transcripts with CSV
# -----------------------------
if df is not None:
    st.subheader("3) Merge Logs + Transcripts")

    # CASE 1: CSV + AUDIO (call_id exists)
    if "call_id" in df.columns and not df_tr.empty:
        df["call_id"] = df["call_id"].astype(str)
        df_tr["call_id"] = df_tr["call_id"].astype(str)
        merged = pd.merge(df, df_tr, on="call_id", how="outer")

    # CASE 2: CSV ONLY (most common)
    else:
        merged = df.copy()
        merged["transcript_text"] = ""

    st.dataframe(merged.head(50), use_container_width=True)
else:
    merged = None
# -----------------------------
# Sentiment: Train or Pretrained
# -----------------------------
#if merged is not None and len(merged) > 0:
if input_mode == "Full CSV Analytics" and merged is not None and not merged.empty:
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
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        if "location" in merged.columns:
            fig2 = px.bar(merged.fillna({"location": "Unknown"}), x="location", color="sentiment", title="Sentiment by Location")
            st.plotly_chart(fig2, use_container_width=True)
    with colC:
        if "tech_stack" in merged.columns:
            fig3 = px.bar(merged.fillna({"tech_stack": "Unknown"}), x="tech_stack", color="sentiment", title="Sentiment by Tech Stack")
            st.plotly_chart(fig3, use_container_width=True)

    # Time trend
    if "date_parsed" in merged.columns and merged["date_parsed"].notna().any():
        temp = merged.copy()
        temp["month"] = temp["date_parsed"].dt.to_period("M").astype(str)
        ts = temp.groupby(["month", "sentiment"]).size().reset_index(name="count")
        fig4 = px.line(ts, x="month", y="count", color="sentiment", markers=True, title="Monthly Sentiment Trend")
        st.plotly_chart(fig4, use_container_width=True)

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
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Not enough negative samples to extract keywords.")

    # -----------------------------
    # Recommendations (rule + keyword heuristics)
    # -----------------------------
    st.subheader("6) Recommendations")

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

    recommendations = gen_recos(merged)
    if recommendations:
        for r in recommendations:
            st.markdown(f"-{r}")
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
