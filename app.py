# ============================================================
# Global Multimodal Fact-Checker — Premium Cyber-Security UI
# ============================================================

import streamlit as st
import joblib
import os
import io
import re
import time
import tempfile
import numpy as np
import cv2
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageChops
from transformers import pipeline
from openai import OpenAI
from deep_translator import GoogleTranslator

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    from fast_c2pa_python import read_c2pa_from_file
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False

# ============================================================
# PAGE CONFIG & ADVANCED CUSTOM CSS
# ============================================================
st.set_page_config(page_title="Global Fact-Checker AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    /* --- GLOBAL THEME --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0F172A; /* Deep Slate */
        color: #E2E8F0;
    }
    
    /* Hide Streamlit default header/footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* --- GLASSMORPHISM CONTAINERS --- */
    .glass-panel {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    
    /* --- MODERN BUTTONS --- */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        color: white;
    }
    
    /* --- VERDICT CARDS --- */
    .verdict-card {
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .verdict-fake {
        background: rgba(239, 68, 68, 0.1);
        border: 2px solid #EF4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
    }
    .verdict-real {
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid #10B981;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
    }
    .verdict-uncertain {
        background: rgba(245, 158, 11, 0.1);
        border: 2px solid #F59E0B;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.3);
    }
    
    /* --- SIGNAL GRID CARDS --- */
    .signal-card {
        background: #1E293B;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        text-align: center;
    }
    
    /* Customize Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        color: #3B82F6 !important;
        border-bottom: 2px solid #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
col_logo, col_space, col_nav = st.columns([2, 5, 3])
with col_logo:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px;'>
        <h1 style='margin:0; font-size:2rem;'>🛡️</h1>
        <h2 style='margin:0; font-size:1.2rem; color:#F8FAFC;'>Global Fact-Checker</h2>
    </div>
    """, unsafe_allow_html=True)
with col_nav:
    st.markdown("<p style='text-align:right; color:#94A3B8; font-size:0.9rem; margin-top:10px;'>Enterprise AI Security</p>", unsafe_allow_html=True)

st.divider()

# ============================================================
# PROVIDER REGISTRY & BACKEND LOGIC (Same as before)
# ============================================================
PROVIDER_DEFAULTS = {
    "OpenAI": {"url": "https://api.openai.com/v1", "models": ["gpt-4o", "gpt-4o-mini"]},
    "Anthropic": {"url": "https://api.anthropic.com/v1", "models": ["claude-3-5-sonnet", "claude-3-haiku"]},
    "Google Gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/", "models": ["gemini-2.0-flash", "gemini-1.5-pro"]},
    "Groq": {"url": "https://api.groq.com/openai/v1", "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]},
    "OpenRouter": {"url": "https://openrouter.ai/api/v1", "models": ["openai/gpt-4o", "meta-llama/llama-3.1-70b-instruct"]},
}

with st.sidebar:
    st.header("⚙️ System Settings")
    with st.expander("🔑 LLM API Keys", expanded=False):
        st.markdown("Add up to 3 API keys to activate AI Judges.")
        tabs = st.tabs(["Judge 1", "Judge 2", "Judge 3"])
        for i, tab in enumerate(tabs):
            slot_num = i + 1
            state_key = f"saved_judge_{slot_num}"
            input_key = f"api_key_input_{slot_num}"
            with tab:
                if state_key in st.session_state:
                    st.success(f"✅ Active: {st.session_state[state_key]['name']}")
                    if st.button(f"Remove Judge {slot_num}", key=f"rem_{slot_num}"):
                        del st.session_state[state_key]
                        st.rerun()
                else:
                    provider_choice = st.selectbox(f"Provider", list(PROVIDER_DEFAULTS.keys()), key=f"prov_{slot_num}")
                    default_url = PROVIDER_DEFAULTS[provider_choice]["url"]
                    base_url = st.text_input(f"Base URL", value=default_url, key=f"url_{slot_num}")
                    model_choice = st.selectbox(f"Model", PROVIDER_DEFAULTS[provider_choice]["models"] + ["Custom..."], key=f"model_sel_{slot_num}")
                    model_name = st.text_input(f"Custom Model", key=f"custom_{slot_num}") if model_choice == "Custom..." else model_choice
                    api_key = st.text_input(f"API Key", type="password", key=input_key)
                    if st.button(f"Activate Judge {slot_num}", key=f"btn_{slot_num}"):
                        if api_key and base_url and model_name:
                            st.session_state[state_key] = {"name": f"{provider_choice} ({model_name})", "api_key": api_key, "base_url": base_url, "model": model_name}
                            st.rerun()

# (Model loaders, analyze_audio, analyze_image_deepfake, analyze_c2pa, extract_video_frames, search_web_multi_tier, analyze_with_search, get_dynamic_llm_providers, ask_llm functions go here exactly as they were in your previous code)
# To save space here, I am assuming those functions are defined above this UI block.

# ============================================================
# MAIN UI: HERO INPUT SECTION
# ============================================================
def reset_app():
    st.session_state.form_key = st.session_state.get("form_key", 0) + 1
    for i in range(1, 4):
        if f"saved_judge_{i}" in st.session_state: del st.session_state[f"saved_judge_{i}"]
    st.rerun()

if "form_key" not in st.session_state: st.session_state.form_key = 0

# Hero Card Container
st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#F8FAFC;'>Analyze Content for Manipulation</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94A3B8; margin-bottom: 2rem;'>Our multimodal AI scans text, images, video, and audio simultaneously.</p>", unsafe_allow_html=True)

tab_text, tab_media = st.tabs(["📝 Paste Text or URL", "📁 Upload Media"])

with tab_text:
    text_input = st.text_area("Text Input", height=150, key=f"text_{st.session_state.form_key}", label_visibility="collapsed", placeholder="Paste a headline, a forwarded message, or a link here...")

with tab_media:
    uploaded_file = st.file_uploader(
        "Upload Media",
        type=["jpg", "jpeg", "png", "txt", "mp4", "mov", "wav", "mp3", "m4a", "flac"],
        key=f"file_{st.session_state.form_key}",
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1: analyze_clicked = st.button("🚀 Run AI Analysis", type="primary", use_container_width=True)
with col_btn2: 
    if st.button("🔄 Clear", use_container_width=True): reset_app()

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PROCESSING & RESULTS DASHBOARD
# ============================================================
if analyze_clicked:
    input_text = ""
    media_verdict = None
    media_analyzed = False
    c2pa_result = None
    is_image = is_video = is_audio = False

    # --- SKELETON LOADING ---
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("<div class='glass-panel' style='text-align:center; padding: 4rem;'>", unsafe_allow_html=True)
        st.spinner("Scanning pixel manipulation & deepfakes...")
        time.sleep(0.5)
        st.spinner("Analyzing text semantics via BERT...")
        time.sleep(0.5)
        st.spinner("Consulting Web Search & AI Judges...")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- ACTUAL PROCESSING ---
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        c2pa_result = analyze_c2pa(uploaded_file)
        if filename.endswith(".txt"):
            input_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            is_image, media_analyzed = True, True
            img = Image.open(uploaded_file)
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text: input_text += "\n" + ocr_text 
            vision_fake, _ = analyze_image_deepfake(img)
            ela_score = ela_manipulation_score(img)
            media_verdict = True if (vision_fake or ela_score > 40) else False
        elif filename.endswith((".mp4", ".mov")):
            is_video, media_analyzed = True, True
            frames = extract_video_frames(uploaded_file)
            if frames:
                v_fake, _ = analyze_image_deepfake(frames[0])
                ela = ela_manipulation_score(frames[0])
                media_verdict = True if (v_fake or ela > 40) else False
        elif filename.endswith((".wav", ".mp3", ".m4a", ".flac")):
            is_audio, media_analyzed = True, True
            audio_is_fake, _, _ = analyze_audio(uploaded_file)
            if audio_is_fake is not None: media_verdict = audio_is_fake

    if text_input.strip(): input_text = text_input.strip() + "\n" + input_text

    if not input_text.strip() and not media_analyzed:
        st.error("⚠️ Please paste text or upload a file.")
    else:
        clean_text = re.sub(r"http\S+", "", input_text).strip()
        
        if clean_text:
            try: translated_text = GoogleTranslator(source="auto", target="en").translate(clean_text)
            except: translated_text = clean_text

            # Local Text Models
            tfidf_vote, hf_vote = None, None
            if tfidf is not None and base_model is not None:
                try:
                    vec = tfidf.transform([translated_text])
                    tfidf_vote = -1 if base_model.predict(vec)[0] == 1 else 1
                except: pass
            if hf_fallback is not None:
                try:
                    hf_res = hf_fallback(translated_text[:512])[0]
                    hf_vote = -1 if hf_res['label'] == 'LABEL_1' else 1
                except: pass

            text_votes = [v for v in [tfidf_vote, hf_vote] if v is not None]
            final_text_vote = -1 if text_votes.count(-1) > text_votes.count(1) else (1 if text_votes else None)
            text_fake = (final_text_vote == -1)

            # Web Search
            search_score, search_detail, live_context = analyze_with_search(translated_text)
            search_fake = (search_score is not None and search_score < 0)

            # LLM Judges
            judge_votes = []
            text_for_judges = f"Context:\n{live_context}\n\nText:\n{translated_text[:1500]}"
            for provider in get_dynamic_llm_providers():
                vote = ask_llm(provider, text_for_judges)
                if isinstance(vote, int): judge_votes.append(vote)
            judge_fake = (judge_votes.count(-1) > judge_votes.count(1)) if judge_votes else False

            # Consensus Math
            media_weight = 2 if media_analyzed else 0
            text_weight = 2 if clean_text else 0
            search_weight = 1 if search_score is not None else 0
            judge_weight = 3 if judge_votes else 0

            fake_score = ((media_weight if media_verdict else 0) + (text_weight if text_fake else 0) + (search_weight if search_fake else 0) + (judge_weight if judge_votes and judge_fake else 0))
            total_weight = media_weight + text_weight + search_weight + judge_weight
            fake_ratio = (fake_score / total_weight) if total_weight > 0 else 0

            # --- RENDER RESULTS DASHBOARD ---
            loading_placeholder.empty() # Clear skeleton loading
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 1. THE VERDICT CARD & GAUGE
            col_verdict, col_gauge = st.columns([2, 1])
            with col_verdict:
                if total_weight == 0:
                    st.markdown(f"""<div class='verdict-card verdict-uncertain'><h1 style='color:#F59E0B; font-size:2.5rem; margin:0;'>⚖️ INCONCLUSIVE</h1><p style='color:#F59E0B; font-size:1.2rem;'>Not enough signals to verify.</p></div>""", unsafe_allow_html=True)
                elif fake_ratio > 0.6:
                    st.markdown(f"""<div class='verdict-card verdict-fake'><h1 style='color:#EF4444; font-size:2.5rem; margin:0;'>🚨 LIKELY FAKE</h1><p style='color:#FCA5A5; font-size:1.2rem;'>Manipulation signals detected.</p></div>""", unsafe_allow_html=True)
                elif fake_ratio < 0.4:
                    st.markdown(f"""<div class='verdict-card verdict-real'><h1 style='color:#10B981; font-size:2.5rem; margin:0;'>✅ LIKELY AUTHENTIC</h1><p style='color:#6EE7B7; font-size:1.2rem;'>Content appears genuine.</p></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='verdict-card verdict-uncertain'><h1 style='color:#F59E0B; font-size:2.5rem; margin:0;'>⚖️ MIXED SIGNALS</h1><p style='color:#FCD34D; font-size:1.2rem;'>Conflicting results from models.</p></div>""", unsafe_allow_html=True)
            
            with col_gauge:
                # Animated SVG Circular Gauge
                circumference = 2 * 3.14159 * 45
                offset = circumference - (fake_ratio * circumference)
                color = "#EF4444" if fake_ratio > 0.6 else ("#10B981" if fake_ratio < 0.4 else "#F59E0B")
                
                gauge_html = f"""
                <div style="display:flex; justify-content:center; align-items:center; height:100%;">
                    <svg width="160" height="160" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="45" stroke="#1E293B" stroke-width="8" fill="none" />
                        <circle cx="50" cy="50" r="45" stroke="{color}" stroke-width="8" fill="none" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" stroke-linecap="round" transform="rotate(-90 50 50)" style="transition: stroke-dashoffset 1s ease-in-out;" />
                        <text x="50" y="50" font-family="Inter" font-size="20" font-weight="bold" fill="#F8FAFC" text-anchor="middle" dy=".3em">{fake_ratio*100:.0f}%</text>
                        <text x="50" y="70" font-family="Inter" font-size="10" fill="#94A3B8" text-anchor="middle">Fake Ratio</text>
                    </svg>
                </div>
                """
                st.markdown(gauge_html, unsafe_allow_html=True)

            # 2. SIGNAL BREAKDOWN GRID
            st.markdown("<h3 style='color:#F8FAFC; margin-top:2rem;'>📊 Signal Breakdown</h3>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            
            def render_signal_card(col, title, status, weight):
                if status is True: color, label = "#EF4444", "FAKE"
                elif status is False: color, label = "#10B981", "REAL"
                elif status == -1: color, label = "#EF4444", "FAKE"
                elif status == 1: color, label = "#10B981", "REAL"
                else: color, label = "#64748B", "N/A"
                
                with col:
                    st.markdown(f"""
                    <div class='signal-card'>
                        <p style='color:#94A3B8; margin:0; font-size:0.9rem;'>{title}</p>
                        <h2 style='color:{color}; margin:10px 0;'>{label}</h2>
                        <p style='color:#475569; margin:0; font-size:0.8rem;'>Weight: {weight}</p>
                    </div>
                    """, unsafe_allow_html=True)

            render_signal_card(c1, "Visual Media", media_verdict if media_analyzed else None, media_weight)
            render_signal_card(c2, "Text Semantics", final_text_vote, text_weight)
            render_signal_card(c3, "Web Search", search_score, search_weight)
            render_signal_card(c4, "AI Judges", judge_votes, judge_weight)

            # 3. ACCORDION DETAILS
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🌐 View Scraped Web Context"):
                if live_context: st.markdown(live_context)
                else: st.write(search_detail)
                
            if c2pa_result:
                with st.expander("📜 View C2PA Provenance"):
                    st.info(c2pa_result['summary'])
