import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences
from main import load_compatible_model

# ---------------------------------------------------------
# Page Setup & Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="EmoSense | Deep Emotion AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Emotion Color & Metadata Theme
# ---------------------------------------------------------
EMOTIONS_CONFIG = {
    "joy": {
        "emoji": "😄",
        "color": "#F59E0B",
        "gradient": "linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(251, 191, 36, 0.05))",
        "description": "Radiating cheerfulness, celebration, contentment, or optimism."
    },
    "sadness": {
        "emoji": "😢",
        "color": "#3B82F6",
        "gradient": "linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(96, 165, 250, 0.05))",
        "description": "Feelings of melancholy, sorrow, disappointment, or loneliness."
    },
    "love": {
        "emoji": "🥰",
        "color": "#EC4899",
        "gradient": "linear-gradient(135deg, rgba(236, 72, 153, 0.25), rgba(244, 114, 182, 0.05))",
        "description": "Deep affection, emotional warmth, adoration, or sincere care."
    },
    "anger": {
        "emoji": "😡",
        "color": "#EF4444",
        "gradient": "linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(248, 113, 113, 0.05))",
        "description": "Feelings of fury, deep resentment, frustration, or hostility."
    },
    "fear": {
        "emoji": "😨",
        "color": "#8B5CF6",
        "gradient": "linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(167, 139, 250, 0.05))",
        "description": "Apprehension, nervousness, dread, or anticipation of danger."
    },
    "surprise": {
        "emoji": "😲",
        "color": "#06B6D4",
        "gradient": "linear-gradient(135deg, rgba(6, 182, 212, 0.25), rgba(34, 211, 238, 0.05))",
        "description": "Unexpected discovery, shock, sudden realization, or astonishment."
    }
}

# ---------------------------------------------------------
# Custom Modern CSS & Animations
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Animated Gradient Header */
    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(120deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        animation: fadeIn 0.8s ease-in-out;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
    }

    /* Glassmorphic Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .glass-card:hover {
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
    }

    /* Dominant Emotion Card */
    .result-container {
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
    }

    /* Pulsing Emoji Avatar */
    .emoji-avatar {
        font-size: 4rem;
        display: inline-block;
        animation: pulseEmoji 2.5s infinite ease-in-out;
        filter: drop-shadow(0 6px 14px rgba(0, 0, 0, 0.35));
    }

    /* Metric Badges */
    .confidence-badge {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseEmoji {
        0%, 100% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.1) rotate(3deg); }
    }

    /* St.TextArea Fine-tuning */
    div[data-baseweb="textarea"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background-color: rgba(255, 255, 255, 0.02) !important;
        transition: border-color 0.2s ease;
    }
    div[data-baseweb="textarea"]:focus-within {
        border-color: #6366F1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Loading
# ---------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Loading Neural Architecture...")
def load_resources():
    model = load_compatible_model("Artifacts/BiGRU_Model.keras")
    with open("Artifacts/tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

try:
    model, tokenizer = load_resources()
except Exception as e:
    st.error(f"❌ Failed to load model artifacts. Ensure files exist in `Artifacts/`. Error: {e}")
    st.stop()

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎭 **About EmoSense**")
    st.write(
        "Powered by a **Bidirectional Gated Recurrent Unit (BiGRU)** neural network trained to decode contextual emotional cues from natural text."
    )
    
    st.markdown("---")
    st.markdown("### 📊 **Supported Classes**")
    for emotion, meta in EMOTIONS_CONFIG.items():
        st.markdown(
            f"<span style='color:{meta['color']}; font-weight:600;'>{meta['emoji']} {emotion.capitalize()}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ⚙️ **Specs**")
    st.caption("• **Model**: BiGRU Recurrent Classifier")
    st.caption("• **Max Sequence**: 50 tokens (post-padded)")
    st.caption("• **Inference Mode**: Softmax Multi-class")

    if st.session_state.history:
        st.markdown("---")
        if st.button("🧹 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="hero-title">🎭 Deep Emotion Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Type your thoughts, paste conversations, or pick a sample prompt to detect real-time sentiment nuances.</div>',
    unsafe_allow_html=True
)

# Preset Prompt Buttons
st.markdown("##### 💡 Try an instant sample:")
preset_samples = {
    "Joy 🌟": "I just got promoted to Senior Engineer and I couldn't be happier!",
    "Sadness 💧": "It feels like everyone has moved on with their lives while I'm still stuck here.",
    "Love ❤️": "Having you by my side through everything makes every single day special.",
    "Anger 🔥": "They canceled our flight with zero notice and refused to even refund our tickets!",
    "Fear ⚡": "I have to walk alone through the deserted alley at midnight and I hear footsteps.",
    "Surprise 😲": "I opened the mailbox and saw a handwritten letter from someone I hadn't seen in 10 years!"
}

cols = st.columns(len(preset_samples))
selected_preset = None

for idx, (label, sample_text) in enumerate(preset_samples.items()):
    if cols[idx].button(label, use_container_width=True):
        selected_preset = sample_text

# Text Input Area
default_input = selected_preset if selected_preset else ""
user_input = st.text_area(
    "Enter a sentence:",
    value=default_input,
    placeholder="Write how you are feeling right now...",
    height=125,
    help="Enter natural language text to analyze."
)

# Quick stats underneath input
word_count = len(user_input.split()) if user_input.strip() else 0
char_count = len(user_input)

col_info_1, col_info_2 = st.columns([8, 2])
with col_info_1:
    st.caption(f"📝 {word_count} words | {char_count} characters")
with col_info_2:
    predict_clicked = st.button("🚀 Analyze Emotion", use_container_width=True, type="primary")

# ---------------------------------------------------------
# Inference & Visual Output
# ---------------------------------------------------------
if (predict_clicked or selected_preset) and user_input.strip():
    with st.spinner("Analyzing semantics..."):
        # Preprocessing
        sequence = tokenizer.texts_to_sequences([user_input.lower()])
        padded = pad_sequences(
            sequence,
            maxlen=50,
            padding="post",
            truncating="post"
        )

        probabilities = model.predict(padded, verbose=0)[0]
        labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        
        top_idx = int(np.argmax(probabilities))
        top_emotion = labels[top_idx]
        top_confidence = probabilities[top_idx] * 100
        
        meta = EMOTIONS_CONFIG[top_emotion]

        # Save to session history
        st.session_state.history.append({
            "text": user_input[:45] + ("..." if len(user_input) > 45 else ""),
            "emotion": top_emotion.capitalize(),
            "confidence": f"{top_confidence:.1f}%"
        })

    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1, 1.35], gap="large")

    # Column 1: Dominant Emotion Highlight Card
    with res_col1:
        st.markdown(f"""
        <div class="result-container" style="background: {meta['gradient']}; border-color: {meta['color']}44;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span class="emoji-avatar">{meta['emoji']}</span>
                <span style="background: {meta['color']}22; color: {meta['color']}; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; border: 1px solid {meta['color']}44;">
                    PRIMARY MATCH
                </span>
            </div>
            <div style="margin-top: 14px;">
                <h2 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; text-transform: capitalize;">
                    {top_emotion}
                </h2>
                <div class="confidence-badge" style="color: {meta['color']};">
                    {top_confidence:.2f}% <span style="font-size: 1rem; font-weight: 500; color: #94A3B8;">confidence</span>
                </div>
                <p style="margin-top: 10px; color: #CBD5E1; font-size: 0.95rem; line-height: 1.5;">
                    {meta['description']}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Column 2: Interactive Probability Breakdown Chart
    with res_col2:
        st.markdown("##### 📈 Emotion Probability Spectrum")
        
        # Prepare dataframe for plotting
        prob_df = pd.DataFrame({
            "Emotion": [e.capitalize() for e in labels],
            "Probability": probabilities * 100,
            "Color": [EMOTIONS_CONFIG[e]["color"] for e in labels]
        }).sort_values(by="Probability", ascending=True)

        fig = go.Figure(go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Emotion"],
            orientation='h',
            text=[f"{val:.1f}%" for val in prob_df["Probability"]],
            textposition='auto',
            textfont=dict(color='white', family='Plus Jakarta Sans', size=12),
            marker=dict(
                color=prob_df["Color"],
                line=dict(width=1, color='rgba(255, 255, 255, 0.2)')
            )
        ))

        fig.update_layout(
            margin=dict(l=0, r=20, t=10, b=10),
            height=260,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.08)',
                range=[0, 105],
                title="Probability (%)",
                color='#94A3B8'
            ),
            yaxis=dict(
                color='#F1F5F9',
                tickfont=dict(size=13, weight='bold')
            )
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Detailed Score Table Expander
    with st.expander("🔍 View Complete Probability Matrix"):
        detail_cols = st.columns(6)
        for i, emotion in enumerate(labels):
            p = probabilities[i] * 100
            detail_cols[i].metric(
                label=f"{EMOTIONS_CONFIG[emotion]['emoji']} {emotion.capitalize()}",
                value=f"{p:.2f}%"
            )

# ---------------------------------------------------------
# Session History Table
# ---------------------------------------------------------
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.markdown("##### 🕒 Recent Predictions in this Session")
    history_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(history_df, use_container_width=True, hide_index=True)