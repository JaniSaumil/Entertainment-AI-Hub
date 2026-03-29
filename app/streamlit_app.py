import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from fpdf import FPDF
from datetime import datetime

# ═══════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════
BRANDING = "Entertainment AI Hub \u2022 Crafted by Saumil Jani \u2022 Microsoft Elevate 2026"

st.set_page_config(
    page_title="Entertainment AI Hub",
    page_icon="\U0001F3AC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════
# PREMIUM CSS DESIGN SYSTEM
# ═══════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;600;700;800;900&display=swap');

    /* ── Global Typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Animated Aurora Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 25%, #1b1040 50%, #0d1b2a 75%, #0a0a1a 100%);
        background-size: 400% 400%;
        animation: auroraFlow 20s ease infinite;
    }
    @keyframes auroraFlow {
        0%   { background-position: 0% 50%; }
        25%  { background-position: 50% 0%; }
        50%  { background-position: 100% 50%; }
        75%  { background-position: 50% 100%; }
        100% { background-position: 0% 50%; }
    }

    div.block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* ── Animated Gradient Title ── */
    h1 {
        font-family: 'Poppins', sans-serif !important;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #f5576c, #fda085, #667eea);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900 !important;
        text-align: center;
        font-size: 2.8rem !important;
        letter-spacing: -0.5px;
        animation: gradientShift 6s ease infinite;
        margin-bottom: 5px !important;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    h2 {
        font-family: 'Poppins', sans-serif !important;
        color: #e8e8ff !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        margin-bottom: 5px !important;
    }
    h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #c8c8e8 !important;
        font-weight: 600 !important;
    }

    /* ── Tab Styling with Neon Glow ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 15, 40, 0.5);
        border-radius: 12px;
        padding: 6px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        color: #8888aa !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2)) !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(102,126,234,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        border-bottom: 2px solid #667eea !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(102, 126, 234, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }

    /* ── Glassmorphism Metric Cards ── */
    .metric-card {
        background: rgba(15, 15, 35, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 28px 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 100%;
        animation: gradientShift 4s ease infinite;
    }
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        border: 1px solid rgba(102, 126, 234, 0.4);
        box-shadow: 0 16px 48px rgba(102, 126, 234, 0.2), 0 0 30px rgba(102, 126, 234, 0.1);
    }
    .metric-icon  { font-size: 2.2rem; margin-bottom: 8px; display: block; }
    .metric-label {
        font-family: 'Inter', sans-serif; font-size: 0.8rem; color: #8888aa;
        font-weight: 500; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Poppins', sans-serif; font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }

    /* ── Cinema Ticket Cards ── */
    .cinema-card {
        background: rgba(15, 15, 35, 0.7);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 0;
        margin-bottom: 12px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: stretch;
        overflow: hidden;
    }
    .cinema-card:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 12px 36px rgba(102, 126, 234, 0.15);
    }
    .cinema-rank {
        background: linear-gradient(180deg, #667eea, #764ba2);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 800; font-size: 1.4rem;
        display: flex; align-items: center; justify-content: center;
        min-width: 55px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .cinema-content { padding: 16px 18px; flex: 1; }
    .cinema-title {
        font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1rem;
        color: #ffffff !important; margin: 0 0 8px 0; line-height: 1.3;
    }
    .cinema-genres { display: flex; flex-wrap: wrap; gap: 5px; }
    .genre-pill {
        background: rgba(102, 126, 234, 0.15); color: #a8b8ff;
        font-size: 0.7rem; font-weight: 600; padding: 3px 10px;
        border-radius: 20px; border: 1px solid rgba(102, 126, 234, 0.2);
        font-family: 'Inter', sans-serif; letter-spacing: 0.3px;
    }
    .cinema-score {
        display: flex; align-items: center; justify-content: center;
        padding: 0 16px; font-family: 'Poppins', sans-serif;
        font-weight: 700; font-size: 0.95rem; color: #4ade80;
        min-width: 65px; text-align: center;
        border-left: 1px solid rgba(255,255,255,0.05);
    }

    /* ── Result Glass Panels ── */
    .result-panel {
        background: rgba(15, 15, 35, 0.7);
        backdrop-filter: blur(20px); border-radius: 20px;
        padding: 35px; border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center; margin-top: 20px;
    }
    .result-panel.positive {
        border-color: rgba(74, 222, 128, 0.3);
        box-shadow: 0 8px 32px rgba(74, 222, 128, 0.1), 0 0 60px rgba(74, 222, 128, 0.05);
    }
    .result-panel.negative {
        border-color: rgba(248, 113, 113, 0.3);
        box-shadow: 0 8px 32px rgba(248, 113, 113, 0.1), 0 0 60px rgba(248, 113, 113, 0.05);
    }

    /* ── Gradient Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; padding: 12px 30px !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important; font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5), 0 0 30px rgba(102, 126, 234, 0.2) !important;
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #059669 100%) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important;
        font-family: 'Poppins', sans-serif !important; font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(13, 148, 136, 0.5) !important;
    }

    /* ── Link Button (GitHub / LinkedIn) ── */
    .stLinkButton > a {
        background: rgba(15, 15, 35, 0.7) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 12px !important;
        color: #a8b8ff !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        text-decoration: none !important;
    }
    .stLinkButton > a:hover {
        background: rgba(102, 126, 234, 0.15) !important;
        border-color: rgba(102, 126, 234, 0.4) !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25) !important;
    }

    /* ── Input Styling (Premium Dark) ── */
    .stTextArea textarea {
        background: #1a1a3e !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 14px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25), 0 0 20px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }

    /* ── Horizontal Rule ── */
    hr {
        border: none; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 25px 0;
    }

    /* ── Context Badges ── */
    .context-badge {
        display: inline-block; padding: 8px 20px; border-radius: 25px;
        font-family: 'Poppins', sans-serif; font-weight: 700;
        font-size: 0.9rem; letter-spacing: 0.5px; margin-top: 15px;
    }
    .context-badge.fire     { background: rgba(245,87,108,0.15);  color: #f5576c; border: 1px solid rgba(245,87,108,0.3); }
    .context-badge.warm     { background: rgba(253,160,133,0.15); color: #fda085; border: 1px solid rgba(253,160,133,0.3); }
    .context-badge.moderate { background: rgba(102,126,234,0.15); color: #667eea; border: 1px solid rgba(102,126,234,0.3); }
    .context-badge.cool     { background: rgba(100,200,255,0.15); color: #64c8ff; border: 1px solid rgba(100,200,255,0.3); }

    /* ── Confidence Bar ── */
    .conf-bar-track {
        width: 100%; height: 10px; background: rgba(255,255,255,0.05);
        border-radius: 10px; overflow: hidden; margin-top: 12px;
    }
    .conf-bar-fill {
        height: 100%; border-radius: 10px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Fade-in ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.6s ease-out forwards; }

    /* ── Pulse ── */
    @keyframes softPulse {
        0%, 100% { transform: scale(1); }
        50%      { transform: scale(1.05); }
    }
    .pulse { animation: softPulse 2s ease-in-out infinite; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 25, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* ── Subtitle with Typewriter ── */
    .subtitle {
        text-align: center; font-family: 'Inter', sans-serif;
        font-size: 1.05rem; color: #b0b0d0; margin-top: -10px;
        margin-bottom: 30px; font-weight: 400; letter-spacing: 0.5px;
    }
    .subtitle-typewriter {
        text-align: center; font-family: 'Inter', sans-serif;
        font-size: 1.08rem; color: #c8c8e8; margin-top: -10px;
        margin-bottom: 30px; font-weight: 400; letter-spacing: 0.5px;
        overflow: hidden; white-space: nowrap;
        border-right: 2px solid #667eea;
        display: inline-block;
        max-width: 100%;
        animation: typewriter 3.5s steps(80, end) forwards, blinkCursor 0.75s step-end infinite;
    }
    @keyframes typewriter {
        from { width: 0; }
        to   { width: 100%; }
    }
    @keyframes blinkCursor {
        50% { border-color: transparent; }
    }
    .subtitle-wrapper {
        text-align: center;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .section-desc {
        font-family: 'Inter', sans-serif; font-size: 0.95rem;
        color: #94A3B8; margin-bottom: 25px; line-height: 1.6;
    }

    /* ── Hero Section ── */
    .hero-container {
        position: relative;
        text-align: center;
        padding: 40px 20px 25px 20px;
        margin-bottom: 35px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(240,147,251,0.08) 50%, rgba(245,87,108,0.06) 100%);
        border: 1px solid rgba(255,255,255,0.06);
        backdrop-filter: blur(20px);
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c, #fda085);
        background-size: 300% 100%;
        animation: gradientShift 6s ease infinite;
    }
    .hero-tagline {
        font-family: 'Inter', sans-serif;
        font-size: 1rem; color: #7878a0;
        max-width: 650px; margin: 0 auto 25px auto;
        line-height: 1.7; font-weight: 400;
    }
    .hero-stats {
        display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;
    }
    .hero-stat {
        text-align: center;
        padding: 12px 24px;
        border-radius: 14px;
        background: rgba(15,15,35,0.5);
        border: 1px solid rgba(255,255,255,0.06);
        min-width: 140px;
        transition: all 0.3s ease;
    }
    .hero-stat:hover {
        border-color: rgba(102,126,234,0.3);
        transform: translateY(-2px);
    }
    .hero-stat-value {
        font-family: 'Poppins', sans-serif;
        font-weight: 800; font-size: 1.6rem;
        background: linear-gradient(135deg, #667eea, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem; color: #7878a0;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin-top: 4px; font-weight: 500;
    }

    /* ── Model Lab Cards ── */
    .model-card {
        background: rgba(15, 15, 35, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .model-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        border-radius: 18px 18px 0 0;
    }
    .model-card:hover {
        transform: translateY(-4px);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 16px 48px rgba(102, 126, 234, 0.15);
    }
    .model-card-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700; font-size: 1.1rem;
        color: #e8e8ff; margin-bottom: 4px;
    }
    .model-card-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem; color: #7878a0;
        margin-bottom: 16px; line-height: 1.5;
    }
    .model-metric-row {
        display: flex; justify-content: space-between;
        align-items: center; padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .model-metric-row:last-child { border-bottom: none; }
    .model-metric-name {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem; color: #8888aa;
    }
    .model-metric-value {
        font-family: 'Poppins', sans-serif;
        font-weight: 700; font-size: 1rem; color: #4ade80;
    }

    /* ── Pro Footer ── */
    .pro-footer {
        text-align: center;
        padding: 30px 20px 15px 20px;
        margin-top: 40px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .footer-brand {
        font-family: 'Poppins', sans-serif;
        font-weight: 800; font-size: 1.1rem;
        background: linear-gradient(135deg, #667eea, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
    }
    .footer-links {
        display: flex; justify-content: center; gap: 20px;
        margin: 12px 0;
    }
    .footer-links a {
        color: #7878a0; text-decoration: none;
        font-family: 'Inter', sans-serif; font-size: 0.82rem;
        transition: color 0.3s ease; font-weight: 500;
    }
    .footer-links a:hover { color: #a8b8ff; }
    .footer-copy {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem; color: #444466;
        margin-top: 10px; letter-spacing: 0.5px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(10,10,25,0.5); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #8090ff; }

    /* ── Select box (Premium Dark) ── */
    .stSelectbox > div > div {
        background: #1a1a3e !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25) !important;
    }
    .stMultiSelect > div > div {
        background: #1a1a3e !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    .stMultiSelect > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25) !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background: rgba(102, 126, 234, 0.2) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: #a8b8ff !important;
        border-radius: 6px !important;
    }

    /* ── Number Input (Premium Dark) ── */
    .stNumberInput > div > div {
        border-radius: 10px !important;
        overflow: hidden;
        border: 1px solid #334155 !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    .stNumberInput > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25) !important;
    }
    .stNumberInput input {
        background: #1a1a3e !important;
        border: none !important;
        border-radius: 0 !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    /* +/- Step Buttons */
    .stNumberInput button {
        background: #111827 !important;
        color: #E2E8F0 !important;
        border: none !important;
        border-left: 1px solid #334155 !important;
        transition: all 0.25s ease !important;
        border-radius: 0 !important;
    }
    .stNumberInput button:hover {
        background: #667eea !important;
        color: #ffffff !important;
    }
    .stNumberInput button:active {
        background: #5568d0 !important;
    }

    /* ── Slider (Premium) ── */
    .stSlider > div > div > div {
        color: #E2E8F0 !important;
    }

    /* ── All Form Labels (Premium Typography) ── */
    .stSelectbox label, .stTextArea label, .stTextInput label,
    .stMultiSelect label, .stNumberInput label, .stSlider label {
        color: #94A3B8 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.4px !important;
    }

    /* ── Selectbox Text Visibility ── */
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] input {
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
    }
    .stSelectbox [data-baseweb="select"] input::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
    }
    /* Dropdown Menu */
    [data-baseweb="menu"] [role="option"] {
        color: #E2E8F0 !important;
        background: #111827 !important;
        transition: background 0.15s ease !important;
    }
    [data-baseweb="menu"] [role="option"]:hover,
    [data-baseweb="menu"] [role="option"][aria-selected="true"] {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #ffffff !important;
    }
    [data-baseweb="menu"],
    [data-baseweb="popover"] > div {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5) !important;
    }

    /* ── TextArea Text Visibility ── */
    .stTextArea textarea {
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
        caret-color: #667eea !important;
    }
    .stTextArea textarea::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
    }

    /* ── TextInput Visibility ── */
    .stTextInput input {
        background: #1a1a3e !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
        caret-color: #667eea !important;
        font-size: 0.95rem !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25), 0 0 20px rgba(102, 126, 234, 0.1) !important;
    }

    /* ── STOP Streamlit Dimming on Rerun ── */
    [data-testid="stAppViewBlockContainer"] {
        opacity: 1 !important;
    }
    .stApp > div {
        opacity: 1 !important;
    }
    /* Override Streamlit's stale element styling */
    .element-container {
        opacity: 1 !important;
        transition: none !important;
    }
    [data-stale="true"] {
        opacity: 0.7 !important;
        transition: opacity 0.15s ease !important;
    }

    /* ── Professional Loading Spinner Overlay ── */
    .stSpinner {
        background: rgba(10, 10, 26, 0.6) !important;
        backdrop-filter: blur(4px) !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    .stSpinner > div {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
    }
    .stSpinner > div > i,
    .stSpinner > div > div:first-child {
        border-width: 3px !important;
        border-color: rgba(102, 126, 234, 0.2) !important;
        border-top-color: #667eea !important;
        width: 24px !important;
        height: 24px !important;
    }
    .stSpinner > div > span,
    .stSpinner > div > div + div,
    .stSpinner > div {
        color: #c8c8e8 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════
def safe_pdf_text(text):
    """Ensure text is safe for PDF (latin-1 encoding)."""
    if isinstance(text, str):
        return text.encode('latin-1', 'replace').decode('latin-1')
    return str(text)

def style_chart(fig, ax):
    """Apply premium dark styling to matplotlib chart."""
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.tick_params(colors='#8888aa', labelsize=9)
    ax.xaxis.label.set_color('#8888aa')
    ax.yaxis.label.set_color('#8888aa')
    ax.title.set_color('#c8c8e8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333355')
    ax.spines['left'].set_color('#333355')
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.grid(axis='y', color='#222244', linewidth=0.5, alpha=0.5)

def generate_watchlist_pdf(selected_movie, recommendations, similarities):
    """Generate a branded Watchlist PDF."""
    pdf = FPDF()
    pdf.add_page()

    # Header banner
    pdf.set_fill_color(15, 15, 35)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_fill_color(102, 126, 234)
    pdf.rect(0, 40, 210, 2, 'F')

    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(8)
    pdf.cell(0, 14, safe_pdf_text('Your Movie Watchlist'), align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(150, 150, 180)
    pdf.cell(0, 8, safe_pdf_text(f'Because you loved: "{selected_movie}"'), align='C', new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(50)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(120, 120, 150)
    pdf.cell(0, 8, safe_pdf_text(f'Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}'), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Table header
    pdf.set_fill_color(25, 25, 55)
    pdf.set_text_color(200, 200, 230)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(12, 10, '#', border=0, fill=True, align='C')
    pdf.cell(95, 10, 'Movie Title', border=0, fill=True)
    pdf.cell(60, 10, 'Genres', border=0, fill=True)
    pdf.cell(23, 10, 'Match', border=0, fill=True, align='C', new_x="LMARGIN", new_y="NEXT")

    # Table rows
    for i, (_, row) in enumerate(recommendations.iterrows()):
        if i % 2 == 0:
            pdf.set_fill_color(18, 18, 42)
        else:
            pdf.set_fill_color(22, 22, 48)
        pdf.set_text_color(220, 220, 240)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(12, 9, str(i + 1), border=0, fill=True, align='C')
        pdf.cell(95, 9, safe_pdf_text(row['title'][:52]), border=0, fill=True)
        genres_text = row['genres'].replace('|', ', ')[:32]
        pdf.set_text_color(150, 170, 220)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(60, 9, safe_pdf_text(genres_text), border=0, fill=True)
        pdf.set_text_color(100, 230, 150)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(23, 9, f"{similarities[i]*100:.0f}%", border=0, fill=True, align='C', new_x="LMARGIN", new_y="NEXT")

    # Footer branding (disable auto page-break so it stays on page 1)
    pdf.set_auto_page_break(auto=False)
    pdf.set_y(-15)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 100, 130)
    pdf.cell(0, 10, safe_pdf_text(BRANDING), align='C')

    return bytes(pdf.output())


def generate_sentiment_pdf(review_text, sentiment_label, confidence):
    """Generate a branded Sentiment Report PDF."""
    pdf = FPDF()
    pdf.add_page()

    # Header banner
    pdf.set_fill_color(15, 15, 35)
    pdf.rect(0, 0, 210, 40, 'F')
    if sentiment_label == "Positive":
        pdf.set_fill_color(74, 222, 128)
    else:
        pdf.set_fill_color(248, 113, 113)
    pdf.rect(0, 40, 210, 2, 'F')

    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(8)
    pdf.cell(0, 14, 'Sentiment Analysis Report', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(150, 150, 180)
    pdf.cell(0, 8, safe_pdf_text(f'Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}'), align='C', new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(50)

    # Result
    pdf.set_font('Helvetica', 'B', 16)
    if sentiment_label == "Positive":
        pdf.set_text_color(74, 222, 128)
        pdf.cell(0, 12, 'RESULT:  Positive Sentiment Detected', new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_text_color(248, 113, 113)
        pdf.cell(0, 12, 'RESULT:  Negative Sentiment Detected', new_x="LMARGIN", new_y="NEXT")

    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(200, 200, 230)
    pdf.cell(0, 10, f'Confidence: {confidence:.1f}%', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Review text
    pdf.set_fill_color(22, 22, 48)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(180, 180, 210)
    pdf.cell(0, 10, 'Analyzed Review:', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(160, 160, 190)
    pdf.multi_cell(0, 6, safe_pdf_text(review_text[:2000]))

    # Footer (disable auto page-break so it stays on page 1)
    pdf.set_auto_page_break(auto=False)
    pdf.set_y(-15)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 100, 130)
    pdf.cell(0, 10, safe_pdf_text(BRANDING), align='C')

    return bytes(pdf.output())


# ═══════════════════════════════════════════════
# DATA & MODEL LOADING (CACHED)
# ═══════════════════════════════════════════════
@st.cache_data
def load_datasets():
    movies = pd.read_csv("data/processed/movies_clean.csv") if os.path.exists("data/processed/movies_clean.csv") else None
    ratings = pd.read_csv("data/processed/ratings_clean.csv") if os.path.exists("data/processed/ratings_clean.csv") else None
    return movies, ratings

@st.cache_resource
def load_models():
    rec_model, sent_model, pop_model = None, None, None
    if os.path.exists("models/recommender_model.pkl"):
        with open("models/recommender_model.pkl", "rb") as f:
            rec_model = pickle.load(f)
    if os.path.exists("models/sentiment_model.pkl"):
        with open("models/sentiment_model.pkl", "rb") as f:
            sent_model = pickle.load(f)
    if os.path.exists("models/popularity_model_tmdb.pkl"):
        with open("models/popularity_model_tmdb.pkl", "rb") as f:
            pop_model = pickle.load(f)
    return rec_model, sent_model, pop_model

movies_df, ratings_df = load_datasets()
rec_model, sent_model, pop_model = load_models()

# ═══════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════
if 'rec_results' not in st.session_state:
    st.session_state.rec_results = None
    st.session_state.rec_sims = None
    st.session_state.rec_source = None
if 'sent_result' not in st.session_state:
    st.session_state.sent_result = None
if 'show_docs' not in st.session_state:
    st.session_state.show_docs = False

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 25px 10px 15px 10px;'>
        <div style='font-size: 3.5rem; margin-bottom: 5px;'>🎬</div>
        <h2 style='font-family: Poppins, sans-serif; font-weight: 800;
            background: linear-gradient(135deg, #667eea, #f093fb);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; font-size: 1.5rem; margin-bottom: 2px;'>
            Entertainment AI Hub
        </h2>
        <p style='color: #7878a0; font-size: 0.85rem; font-family: Inter, sans-serif;'>
            Intelligent Media Analytics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### \U0001F9E0 About This Dashboard")
    st.markdown("""
    <p style='color:#8888aa; font-size:0.85rem; line-height:1.6;'>
    This AI-powered dashboard combines <b style='color:#a8b8ff;'>Machine Learning</b>,
    <b style='color:#a8b8ff;'>Natural Language Processing</b>, and
    <b style='color:#a8b8ff;'>Data Analytics</b> to deliver intelligent insights
    about the entertainment industry.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### \u26A1 Tech Stack")
    st.markdown("""
    <p style='color:#8888aa; font-size:0.82rem; line-height:1.8;'>
    \U0001F4CA <b style='color:#c8c8e8;'>Frontend:</b> Streamlit<br>
    \U0001F916 <b style='color:#c8c8e8;'>ML Engine:</b> scikit-learn<br>
    \U0001F4AC <b style='color:#c8c8e8;'>NLP:</b> NLTK + TF-IDF<br>
    \U0001F4C1 <b style='color:#c8c8e8;'>Data:</b> TMDB 5000, MovieLens<br>
    \U0001F4C8 <b style='color:#c8c8e8;'>Visualization:</b> Matplotlib, Seaborn
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; padding: 15px 10px;
        background: rgba(102,126,234,0.08); border-radius: 12px;
        border: 1px solid rgba(102,126,234,0.15);'>
        <p style='color:#a8b8ff; font-family:Poppins,sans-serif;
           font-weight:700; font-size:0.9rem; margin-bottom:4px;'>
           Made with \u2764\uFE0F by Saumil Jani
        </p>
        <p style='color:#667eea; font-family:Inter,sans-serif;
           font-size:0.75rem; letter-spacing:1px;'>
           MICROSOFT ELEVATE 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# MAIN TITLE
# ═══════════════════════════════════════════════
st.title("\U0001F3AC Entertainment AI Hub")
st.markdown("<div class='subtitle-wrapper'><span class='subtitle-typewriter'>Intelligent Insights &bull; Smart Recommendations &bull; Predictive Analytics for the Digital Entertainment Era</span></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════
hero_movies = f"{len(movies_df):,}" if movies_df is not None else "9,000+"
hero_ratings = f"{len(ratings_df):,}" if ratings_df is not None else "100,000+"
hero_models = "4"
hero_accuracy = "85%+"

st.markdown(f"""
<div class='hero-container fade-in'>
    <p class='hero-tagline'>
        Welcome to the next generation of entertainment intelligence. Our platform combines
        <strong style='color:#a8b8ff;'>machine learning</strong>,
        <strong style='color:#a8b8ff;'>natural language processing</strong>, and
        <strong style='color:#a8b8ff;'>predictive analytics</strong>
        to decode audience behavior and market trends.
    </p>
    <div class='hero-stats'>
        <div class='hero-stat'>
            <div class='hero-stat-value'>\U0001F3AC {hero_movies}</div>
            <div class='hero-stat-label'>Movies Analyzed</div>
        </div>
        <div class='hero-stat'>
            <div class='hero-stat-value'>\u2B50 {hero_ratings}</div>
            <div class='hero-stat-label'>User Ratings</div>
        </div>
        <div class='hero-stat'>
            <div class='hero-stat-value'>\U0001F9E0 {hero_models}</div>
            <div class='hero-stat-label'>ML Models</div>
        </div>
        <div class='hero-stat'>
            <div class='hero-stat-value'>\U0001F3AF {hero_accuracy}</div>
            <div class='hero-stat-label'>Model Accuracy</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB DEFINITIONS
# ═══════════════════════════════════════════════
tab_eda, tab_rec, tab_sent, tab_pop, tab_model = st.tabs([
    "\U0001F4CA Market Insights",
    "\U0001F916 Smart Recommender",
    "\U0001F4AC Sentiment Tracker",
    "\U0001F680 Popularity Predictor",
    "\u2699\uFE0F Model Lab"
])

# ═══════════════════════════════════════════════
# TAB 1 — MARKET INSIGHTS (EDA)
# ═══════════════════════════════════════════════
with tab_eda:
    st.header("Platform Intelligence Dashboard")
    st.markdown("<p class='section-desc'>Explore user consumption patterns, content distributions, and market trends across thousands of entertainment titles in our data universe.</p>", unsafe_allow_html=True)

    if movies_df is not None and ratings_df is not None:
        # — 4 Metric Cards —
        col1, col2, col3, col4 = st.columns(4)
        avg_rating = round(ratings_df['rating'].mean(), 2)
        col1.markdown(f"<div class='metric-card'><span class='metric-icon'>\U0001F3AC</span><div class='metric-label'>Total Movies</div><div class='metric-value'>{len(movies_df):,}</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><span class='metric-icon'>\u2B50</span><div class='metric-label'>Total Ratings</div><div class='metric-value'>{len(ratings_df):,}</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><span class='metric-icon'>\U0001F464</span><div class='metric-label'>Unique Users</div><div class='metric-value'>{ratings_df['userId'].nunique():,}</div></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-card'><span class='metric-icon'>\U0001F3AF</span><div class='metric-label'>Avg Rating</div><div class='metric-value'>{avg_rating}</div></div>", unsafe_allow_html=True)

        st.write("---")

        # — Charts Row 1 —
        col_fig1, col_fig2 = st.columns(2)

        with col_fig1:
            st.subheader("Ratings Distribution")
            fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='none')
            palette = ['#4a3f8a','#5c4fa0','#6e5fb6','#8070cc','#9280e0','#a490f0','#b6a0ff','#c8b0ff','#dac0ff','#ecd0ff']
            rating_counts = ratings_df['rating'].value_counts().sort_index()
            bars = ax.bar(rating_counts.index.astype(str), rating_counts.values,
                          color=palette[:len(rating_counts)], edgecolor='none', width=0.7)
            style_chart(fig, ax)
            ax.set_xlabel("Rating", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            st.pyplot(fig)


        with col_fig2:
            st.subheader("Movies by Release Year")
            movies_yr = movies_df.dropna(subset=["year"])
            year_counts = movies_yr["year"].value_counts().sort_index().tail(30)
            fig2, ax2 = plt.subplots(figsize=(7, 4.5), facecolor='none')
            ax2.fill_between(year_counts.index, year_counts.values, alpha=0.15, color='#667eea')
            ax2.plot(year_counts.index, year_counts.values, color='#667eea', linewidth=2.5)
            ax2.scatter(year_counts.index, year_counts.values, color='#f093fb', s=15, zorder=5)
            style_chart(fig2, ax2)
            ax2.set_xlabel("Year", fontsize=10)
            ax2.set_ylabel("Number of Movies", fontsize=10)
            st.pyplot(fig2)

        # — Chart Row 2 — Top 10 Most Rated Movies
        st.subheader("Top 10 Most Rated Movies")
        title_col = 'clean_title' if 'clean_title' in movies_df.columns else 'title'
        top_rated = ratings_df.groupby('movieId').size().reset_index(name='rating_count')
        top_rated = top_rated.merge(movies_df[['movieId', title_col]], on='movieId', how='left')
        top_rated = top_rated.nlargest(10, 'rating_count')

        fig3, ax3 = plt.subplots(figsize=(14, 5), facecolor='none')
        bars = ax3.barh(
            top_rated[title_col].values[::-1],
            top_rated['rating_count'].values[::-1],
            color=[plt.cm.cool(i / 10) for i in range(10)],
            edgecolor='none',
            height=0.65
        )
        style_chart(fig3, ax3)
        ax3.set_xlabel("Number of Ratings", fontsize=10)
        ax3.grid(axis='x', color='#222244', linewidth=0.5, alpha=0.5)
        ax3.grid(axis='y', visible=False)
        for bar in bars:
            width = bar.get_width()
            ax3.text(width + 50, bar.get_y() + bar.get_height()/2,
                     f'{int(width):,}', va='center', ha='left',
                     color='#a8b8ff', fontsize=8, fontfamily='sans-serif')
        plt.tight_layout()
        st.pyplot(fig3)

        # — Chart Row 3 — Genre Distribution Donut —
        st.subheader("Genre Distribution")
        if 'genres' in movies_df.columns:
            all_genre_list = movies_df['genres'].dropna().str.split('|').explode()
            genre_counts = all_genre_list.value_counts().head(12)

            fig4, ax4 = plt.subplots(figsize=(7, 7), facecolor='none')
            donut_colors = [
                '#667eea', '#764ba2', '#f093fb', '#f5576c', '#fda085',
                '#4ade80', '#64c8ff', '#a78bfa', '#f472b6', '#fb923c',
                '#34d399', '#818cf8'
            ]
            wedges, texts, autotexts = ax4.pie(
                genre_counts.values,
                labels=genre_counts.index,
                colors=donut_colors[:len(genre_counts)],
                autopct='%1.1f%%',
                startangle=140,
                pctdistance=0.82,
                wedgeprops=dict(width=0.45, edgecolor='#0a0a1a', linewidth=2)
            )
            for text in texts:
                text.set_color('#c8c8e8')
                text.set_fontsize(9)
                text.set_fontfamily('sans-serif')
            for autotext in autotexts:
                autotext.set_color('#e8e8ff')
                autotext.set_fontsize(7.5)
                autotext.set_fontweight('bold')

            # Center circle for donut effect
            centre_circle = plt.Circle((0, 0), 0.55, fc='#0a0a1a')
            ax4.add_artist(centre_circle)
            ax4.text(0, 0.06, f'{len(genre_counts)}', ha='center', va='center',
                     fontsize=28, fontweight='bold', color='#667eea', fontfamily='sans-serif')
            ax4.text(0, -0.1, 'Genres', ha='center', va='center',
                     fontsize=10, color='#7878a0', fontfamily='sans-serif')
            fig4.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig4)
    else:
        st.warning("Processed datasets not found. Please run preprocessing first.")


# ═══════════════════════════════════════════════
# TAB 2 — SMART RECOMMENDER
# ═══════════════════════════════════════════════
with tab_rec:
    st.header("Discover Your Next Favorite")
    st.markdown("<p class='section-desc'>Select a movie you love, and our AI engine will surface hidden gems tailored to your taste using content-based semantic analysis.</p>", unsafe_allow_html=True)

    if rec_model is not None:
        movies_ref = rec_model["movies"]
        genre_matrix = rec_model["genre_matrix"]

        selected_movie = st.selectbox("Search for a movie you love:", movies_ref["title"].values, key="rec_select")

        if st.button("Generate Recommendations \u2728", type="primary", key="rec_btn"):
            with st.spinner("\U0001F50D Analyzing semantic fingerprints — please wait..."):
                idx = movies_ref[movies_ref["title"] == selected_movie].index[0]
                sim_scores = cosine_similarity(genre_matrix[idx], genre_matrix).flatten()
                top_indices = sim_scores.argsort()[::-1][1:9]  # Top 8
                recs = movies_ref.iloc[top_indices]
                sims = sim_scores[top_indices]
                st.session_state.rec_results = recs
                st.session_state.rec_sims = sims
                st.session_state.rec_source = selected_movie
                st.toast(f"\u2728 Found 8 perfect matches for \"{selected_movie}\"!", icon="\U0001F3AC")

        # Display Results (persisted via session state)
        if st.session_state.rec_results is not None:
            recs = st.session_state.rec_results
            sims = st.session_state.rec_sims
            source = st.session_state.rec_source

            st.markdown(f"<p style='color:#4ade80; font-weight:600; font-size:1rem; margin-top:15px;'>\u2705 Showing 8 best matches for <em>\"{source}\"</em></p>", unsafe_allow_html=True)

            # Cinema Ticket Grid (2 columns)
            col_l, col_r = st.columns(2)
            for i, (_, row) in enumerate(recs.iterrows()):
                genres_list = row['genres'].split('|')
                pills_html = " ".join([f"<span class='genre-pill'>{g}</span>" for g in genres_list])
                sim_pct = f"{sims[i]*100:.0f}%"
                rank = i + 1

                card_html = f"""
                <div class='cinema-card fade-in' style='animation-delay: {i*0.08}s;'>
                    <div class='cinema-rank'>{rank:02d}</div>
                    <div class='cinema-content'>
                        <div class='cinema-title'>{row['title']}</div>
                        <div class='cinema-genres'>{pills_html}</div>
                    </div>
                    <div class='cinema-score'>{sim_pct}</div>
                </div>
                """
                if i % 2 == 0:
                    col_l.markdown(card_html, unsafe_allow_html=True)
                else:
                    col_r.markdown(card_html, unsafe_allow_html=True)

            # Download Watchlist PDF
            st.markdown("---")
            pdf_bytes = generate_watchlist_pdf(source, recs, sims)
            st.download_button(
                label="\U0001F4E5 Download Watchlist as PDF",
                data=pdf_bytes,
                file_name=f"Watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="dl_watchlist"
            )
    else:
        st.error("Recommender model not trained yet. Run `train_recommender.py` first.")


# ═══════════════════════════════════════════════
# TAB 3 — SENTIMENT TRACKER
# ═══════════════════════════════════════════════
with tab_sent:
    st.header("Audience Sentiment Intelligence")
    st.markdown("<p class='section-desc'>Paste any movie review and watch our NLP engine dissect the emotional texture in real-time using TF-IDF vectorization and Logistic Regression.</p>", unsafe_allow_html=True)

    user_review = st.text_area(
        "Paste a movie review or audience comment:",
        height=160,
        placeholder="e.g., The pacing was terrible and the characters felt flat. I wouldn't recommend it to anyone.",
        key="sent_input"
    )

    if st.button("Analyze Sentiment \U0001F50D", type="primary", key="sent_btn"):
        if sent_model is not None:
            if user_review.strip():
                with st.spinner("\U0001F9E0 Processing text through NLP layers — please wait..."):
                    pred = sent_model.predict([user_review])[0]
                    prob = sent_model.predict_proba([user_review])[0]
                    if pred == 1:
                        st.session_state.sent_result = {
                            "label": "Positive", "confidence": prob[1] * 100,
                            "emoji": "\u2728", "review": user_review
                        }
                        st.toast("\u2728 Positive vibes detected!", icon="\U0001F49A")
                    else:
                        st.session_state.sent_result = {
                            "label": "Negative", "confidence": prob[0] * 100,
                            "emoji": "\u26A0\uFE0F", "review": user_review
                        }
                        st.toast("\u26A0\uFE0F Negative sentiment detected", icon="\U0001F534")
            else:
                st.warning("Please enter some text to analyze.")
        else:
            st.error("Sentiment model not found. Run `train_sentiment.py` first.")

    # Display Results
    if st.session_state.sent_result is not None:
        r = st.session_state.sent_result
        panel_class = "positive" if r["label"] == "Positive" else "negative"
        glow_color = "#4ade80" if r["label"] == "Positive" else "#f87171"
        bar_gradient = "linear-gradient(90deg, #059669, #4ade80)" if r["label"] == "Positive" else "linear-gradient(90deg, #dc2626, #f87171)"
        label_text = "\u2728 Positive Sentiment Detected!" if r["label"] == "Positive" else "\u26A0\uFE0F Negative Sentiment Detected!"
        desc_text = "The review expresses a favorable opinion." if r["label"] == "Positive" else "The review expresses an unfavorable opinion."

        st.markdown(f"""
        <div class='result-panel {panel_class} fade-in'>
            <div style='font-size: 4rem;' class='pulse'>{r["emoji"]}</div>
            <h2 style='color:{glow_color}; margin:8px 0 5px 0; font-size:1.5rem;'>{label_text}</h2>
            <p style='color:#8888aa; font-size: 0.95rem; margin-bottom: 15px;'>{desc_text}</p>
            <div style='display:flex; align-items:center; justify-content:center; gap:12px;'>
                <span style='color:#c8c8e8; font-family:Inter,sans-serif; font-weight:600; font-size:1rem;'>Confidence</span>
                <span style='color:{glow_color}; font-family:Poppins,sans-serif; font-weight:800; font-size:2rem;'>{r["confidence"]:.1f}%</span>
            </div>
            <div class='conf-bar-track'>
                <div class='conf-bar-fill' style='width:{r["confidence"]}%; background:{bar_gradient};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Try to show top influential words
        try:
            vectorizer = sent_model[0]
            classifier = sent_model[1]
            tfidf_vec = vectorizer.transform([r["review"]])
            feature_names = vectorizer.get_feature_names_out()
            coefs = classifier.coef_[0]
            word_importance = tfidf_vec.toarray()[0] * coefs
            nonzero_mask = tfidf_vec.toarray()[0] > 0
            active_indices = np.where(nonzero_mask)[0]

            if len(active_indices) > 0:
                active_importance = word_importance[active_indices]
                sorted_idx = active_importance.argsort()
                top_pos_idx = sorted_idx[-5:][::-1]
                top_neg_idx = sorted_idx[:5]

                pos_words = [(feature_names[active_indices[i]], active_importance[i]) for i in top_pos_idx if active_importance[i] > 0]
                neg_words = [(feature_names[active_indices[i]], abs(active_importance[i])) for i in top_neg_idx if active_importance[i] < 0]

                if pos_words or neg_words:
                    st.markdown("#### \U0001F50D Key Words Detected")
                    wc1, wc2 = st.columns(2)
                    with wc1:
                        if pos_words:
                            st.markdown("<p style='color:#4ade80; font-weight:700; font-size:0.9rem;'>\u2705 Positive Signals</p>", unsafe_allow_html=True)
                            for word, score in pos_words:
                                bar_w = min(100, score * 300)
                                st.markdown(f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'><span style='color:#c8c8e8; font-size:0.85rem; min-width:80px;'>{word}</span><div style='flex:1; height:6px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden;'><div style='width:{bar_w}%; height:100%; background:linear-gradient(90deg, #059669, #4ade80); border-radius:4px;'></div></div></div>", unsafe_allow_html=True)
                    with wc2:
                        if neg_words:
                            st.markdown("<p style='color:#f87171; font-weight:700; font-size:0.9rem;'>\u274C Negative Signals</p>", unsafe_allow_html=True)
                            for word, score in neg_words:
                                bar_w = min(100, score * 300)
                                st.markdown(f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'><span style='color:#c8c8e8; font-size:0.85rem; min-width:80px;'>{word}</span><div style='flex:1; height:6px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden;'><div style='width:{bar_w}%; height:100%; background:linear-gradient(90deg, #dc2626, #f87171); border-radius:4px;'></div></div></div>", unsafe_allow_html=True)
        except Exception:
            pass  # Silently skip word breakdown if model structure is unexpected

        # Download Sentiment Report PDF
        st.markdown("---")
        pdf_bytes = generate_sentiment_pdf(r["review"], r["label"], r["confidence"])
        st.download_button(
            label="\U0001F4C4 Download Sentiment Report as PDF",
            data=pdf_bytes,
            file_name=f"Sentiment_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            key="dl_sentiment"
        )


# ═══════════════════════════════════════════════
# TAB 4 — POPULARITY PREDICTOR
# ═══════════════════════════════════════════════
with tab_pop:
    st.header("Content Popularity Oracle")
    st.markdown("<p class='section-desc'>Simulate how a hypothetical film would perform in the market using real TMDB intelligence. Adjust budget, runtime, and genres to see the predicted popularity score.</p>", unsafe_allow_html=True)

    if pop_model is not None:
        model = pop_model["model"]
        mlb = pop_model["mlb"]
        features_order = pop_model["features"]
        max_pop = pop_model.get("max_popularity", 500)

        col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
        with col_p1:
            budget = st.number_input("Estimated Budget ($)", min_value=0, value=50_000_000, step=1_000_000, key="pop_budget")
        with col_p2:
            runtime = st.slider("Runtime (minutes)", min_value=30, max_value=240, value=120, key="pop_runtime")
        with col_p3:
            all_genres = list(mlb.classes_)
            selected_genres = st.multiselect("Target Genres", all_genres, default=["Action", "Adventure"], key="pop_genres")

        if st.button("Predict TMDB Popularity \U0001F52E", type="primary", key="pop_btn"):
            if selected_genres:
                with st.spinner("\U0001F680 Predicting popularity score — please wait..."):
                    genre_df = pd.DataFrame(mlb.transform([selected_genres]), columns=mlb.classes_)
                    input_data = pd.concat([pd.DataFrame({"budget": [budget], "runtime": [runtime]}), genre_df], axis=1)
                    input_data = input_data.reindex(columns=features_order, fill_value=0)
                    pred_pop = model.predict(input_data)[0]

                # Context badge logic
                if pred_pop > 100:
                    badge_class, badge_text, badge_emoji = "fire", "Blockbuster Potential", "\U0001F525"
                    score_color = "#f5576c"
                elif pred_pop > 50:
                    badge_class, badge_text, badge_emoji = "warm", "Strong Market Presence", "\U0001F4C8"
                    score_color = "#fda085"
                elif pred_pop > 20:
                    badge_class, badge_text, badge_emoji = "moderate", "Moderate Interest", "\U0001F4AB"
                    score_color = "#667eea"
                else:
                    badge_class, badge_text, badge_emoji = "cool", "Niche Audience", "\U0001F331"
                    score_color = "#64c8ff"

                pct = min(100, (pred_pop / 150.0) * 100)

                st.markdown(f"""
                <div class='result-panel fade-in' style='border-color: {score_color}33; box-shadow: 0 8px 32px {score_color}15, 0 0 60px {score_color}08;'>
                    <p style='color:#8888aa; font-family:Inter,sans-serif; font-size:0.85rem; text-transform:uppercase; letter-spacing:2px; margin-bottom:10px;'>Predicted TMDB Popularity Score</p>
                    <div style='font-family:Poppins,sans-serif; font-weight:900; font-size:5rem; line-height:1;
                        background: linear-gradient(135deg, #667eea, {score_color});
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
                        {pred_pop:.1f}
                    </div>
                    <div class='conf-bar-track' style='margin-top:20px; height:12px;'>
                        <div class='conf-bar-fill' style='width:{pct}%; background:linear-gradient(90deg, #667eea, {score_color}); height:100%;'></div>
                    </div>
                    <div class='context-badge {badge_class}' style='margin-top:18px;'>
                        {badge_emoji} {badge_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Budget formatting
                if budget >= 1_000_000:
                    budget_str = f"${budget/1_000_000:.0f}M"
                else:
                    budget_str = f"${budget:,}"

                st.markdown(f"""
                <div style='display:flex; justify-content:center; gap:30px; margin-top:20px;'>
                    <div style='text-align:center;'>
                        <p style='color:#8888aa; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:2px;'>Budget</p>
                        <p style='color:#e8e8ff; font-family:Poppins,sans-serif; font-weight:700; font-size:1.1rem;'>{budget_str}</p>
                    </div>
                    <div style='text-align:center;'>
                        <p style='color:#8888aa; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:2px;'>Runtime</p>
                        <p style='color:#e8e8ff; font-family:Poppins,sans-serif; font-weight:700; font-size:1.1rem;'>{runtime} min</p>
                    </div>
                    <div style='text-align:center;'>
                        <p style='color:#8888aa; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:2px;'>Genres</p>
                        <p style='color:#e8e8ff; font-family:Poppins,sans-serif; font-weight:700; font-size:1.1rem;'>{', '.join(selected_genres)}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Select at least one genre to predict.")
    else:
        st.error("Popularity model not trained yet. Run `train_popularity.py` first.")


# ═══════════════════════════════════════════════
# TAB 5 — MODEL LAB
# ═══════════════════════════════════════════════
with tab_model:
    st.header("Under the Hood")
    st.markdown("<p class='section-desc'>Peek behind the curtain and explore the performance metrics, architecture, and evaluation results of every ML model powering this platform.</p>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)

    # --- Recommender Model Info ---
    with col_m1:
        rec_status = "\u2705 Trained" if rec_model is not None else "\u274C Not Trained"
        rec_movies_count = len(rec_model['movies']) if rec_model is not None else 0
        rec_features = rec_model['genre_matrix'].shape[1] if rec_model is not None else 0
        st.markdown(f"""
        <div class='model-card fade-in' style='animation-delay: 0s;'>
            <div style='position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg, #667eea, #764ba2); border-radius:18px 18px 0 0;'></div>
            <div style='font-size:2.5rem; margin-bottom:8px;'>\U0001F916</div>
            <div class='model-card-title'>Content Recommender</div>
            <div class='model-card-sub'>Cosine similarity on genre-encoded feature vectors</div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Status</span>
                <span class='model-metric-value'>{rec_status}</span>
            </div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Algorithm</span>
                <span class='model-metric-value' style='color:#a8b8ff;'>Cosine Sim</span>
            </div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Movies Indexed</span>
                <span class='model-metric-value' style='color:#f093fb;'>{rec_movies_count:,}</span>
            </div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Feature Dims</span>
                <span class='model-metric-value' style='color:#fda085;'>{rec_features}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Sentiment Model Info ---
    with col_m2:
        sent_status = "\u2705 Trained" if sent_model is not None else "\u274C Not Trained"
        sent_algo = "Logistic Regression"
        sent_vectorizer = "TF-IDF"
        if sent_model is not None:
            try:
                vocab_size = len(sent_model[0].vocabulary_)
            except Exception:
                vocab_size = "N/A"
        else:
            vocab_size = 0
        st.markdown(f"""
        <div class='model-card fade-in' style='animation-delay: 0.15s;'>
            <div style='position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg, #4ade80, #059669); border-radius:18px 18px 0 0;'></div>
            <div style='font-size:2.5rem; margin-bottom:8px;'>\U0001F4AC</div>
            <div class='model-card-title'>Sentiment Analyzer</div>
            <div class='model-card-sub'>TF-IDF vectorization + Logistic Regression classifier</div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Status</span>
                <span class='model-metric-value'>{sent_status}</span>
            </div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Algorithm</span>
                <span class='model-metric-value' style='color:#a8b8ff;'>{sent_algo}</span>
            </div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Vectorizer</span>
                <span class='model-metric-value' style='color:#f093fb;'>{sent_vectorizer}</span>
            </div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Vocabulary</span>
                <span class='model-metric-value' style='color:#fda085;'>{vocab_size:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Popularity Model Info ---
    with col_m3:
        pop_status = "\u2705 Trained" if pop_model is not None else "\u274C Not Trained"
        pop_features_count = len(pop_model['features']) if pop_model is not None else 0
        pop_r2 = pop_model.get('r2_score', None) if pop_model is not None else None
        pop_mae = pop_model.get('mae', None) if pop_model is not None else None
        r2_display = f"{pop_r2:.3f}" if pop_r2 is not None else "N/A"
        mae_display = f"{pop_mae:.2f}" if pop_mae is not None else "N/A"
        st.markdown(f"""
        <div class='model-card fade-in' style='animation-delay: 0.3s;'>
            <div style='position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg, #f5576c, #fda085); border-radius:18px 18px 0 0;'></div>
            <div style='font-size:2.5rem; margin-bottom:8px;'>\U0001F680</div>
            <div class='model-card-title'>Popularity Predictor</div>
            <div class='model-card-sub'>Random Forest Regressor on TMDB movie features</div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Status</span>
                <span class='model-metric-value'>{pop_status}</span>
            </div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Algorithm</span>
                <span class='model-metric-value' style='color:#a8b8ff;'>Random Forest</span>
            </div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>Features</span>
                <span class='model-metric-value' style='color:#f093fb;'>{pop_features_count}</span>
            </div>
            <div class='model-metric-row'>
                <span class='model-metric-name'>R\u00b2 Score</span>
                <span class='model-metric-value' style='color:#fda085;'>{r2_display}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Sentiment Evaluation Metrics (if model exists) ---
    if sent_model is not None:
        st.markdown("---")
        st.subheader("Sentiment Model Evaluation")
        st.markdown("<p class='section-desc'>Evaluated on the NLTK movie_reviews corpus test split.</p>", unsafe_allow_html=True)

        try:
            import nltk
            from nltk.corpus import movie_reviews
            nltk.download('movie_reviews', quiet=True)

            neg_files = movie_reviews.fileids('neg')
            pos_files = movie_reviews.fileids('pos')
            reviews = [movie_reviews.raw(f) for f in neg_files] + [movie_reviews.raw(f) for f in pos_files]
            labels = [0] * len(neg_files) + [1] * len(pos_files)

            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

            y_pred = sent_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            eval_col1, eval_col2 = st.columns(2)
            with eval_col1:
                st.markdown(f"""
                <div class='model-card'>
                    <div class='model-card-title'>Classification Metrics</div>
                    <div class='model-metric-row'>
                        <span class='model-metric-name'>Accuracy</span>
                        <span class='model-metric-value'>{acc*100:.1f}%</span>
                    </div>
                    <div class='model-metric-row'>
                        <span class='model-metric-name'>True Positives</span>
                        <span class='model-metric-value' style='color:#4ade80;'>{cm[1][1]}</span>
                    </div>
                    <div class='model-metric-row'>
                        <span class='model-metric-name'>True Negatives</span>
                        <span class='model-metric-value' style='color:#4ade80;'>{cm[0][0]}</span>
                    </div>
                    <div class='model-metric-row'>
                        <span class='model-metric-name'>False Positives</span>
                        <span class='model-metric-value' style='color:#f87171;'>{cm[0][1]}</span>
                    </div>
                    <div class='model-metric-row'>
                        <span class='model-metric-name'>False Negatives</span>
                        <span class='model-metric-value' style='color:#f87171;'>{cm[1][0]}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with eval_col2:
                st.markdown("**Confusion Matrix**")
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor='none')
                sns.heatmap(cm, annot=True, fmt='d', cmap='magma',
                            xticklabels=['Negative', 'Positive'],
                            yticklabels=['Negative', 'Positive'],
                            ax=ax_cm, cbar_kws={'shrink': 0.8},
                            linewidths=2, linecolor='#0a0a1a',
                            annot_kws={'fontsize': 16, 'fontweight': 'bold'})
                ax_cm.set_xlabel('Predicted', fontsize=11, color='#8888aa')
                ax_cm.set_ylabel('Actual', fontsize=11, color='#8888aa')
                ax_cm.tick_params(colors='#8888aa', labelsize=10)
                fig_cm.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig_cm)

        except Exception as e:
            st.info(f"Evaluation data not available: {e}")

    # --- Model Architecture Summary ---
    st.markdown("---")
    st.subheader("\U0001F3D7\uFE0F System Architecture")
    st.markdown("""
    <div class='model-card'>
        <div style='display:flex; flex-wrap:wrap; gap:20px; justify-content:center; text-align:center;'>
            <div style='flex:1; min-width:180px; padding:15px;'>
                <div style='font-size:2rem; margin-bottom:8px;'>\U0001F4C1</div>
                <div style='color:#e8e8ff; font-family:Poppins,sans-serif; font-weight:700; font-size:0.95rem;'>Data Layer</div>
                <div style='color:#7878a0; font-size:0.8rem; margin-top:6px;'>TMDB 5000 + MovieLens<br>Cleaned & Preprocessed CSVs</div>
            </div>
            <div style='color:#667eea; font-size:1.5rem; display:flex; align-items:center;'>\u27A1</div>
            <div style='flex:1; min-width:180px; padding:15px;'>
                <div style='font-size:2rem; margin-bottom:8px;'>\U0001F9E0</div>
                <div style='color:#e8e8ff; font-family:Poppins,sans-serif; font-weight:700; font-size:0.95rem;'>ML Engine</div>
                <div style='color:#7878a0; font-size:0.8rem; margin-top:6px;'>scikit-learn Pipelines<br>TF-IDF, Random Forest, LogReg</div>
            </div>
            <div style='color:#667eea; font-size:1.5rem; display:flex; align-items:center;'>\u27A1</div>
            <div style='flex:1; min-width:180px; padding:15px;'>
                <div style='font-size:2rem; margin-bottom:8px;'>\U0001F310</div>
                <div style='color:#e8e8ff; font-family:Poppins,sans-serif; font-weight:700; font-size:0.95rem;'>Presentation Layer</div>
                <div style='color:#7878a0; font-size:0.8rem; margin-top:6px;'>Streamlit Dashboard<br>Interactive UI + PDF Reports</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PROFESSIONAL FOOTER
# ═══════════════════════════════════════════════
st.markdown("---")

# --- Documentation Section (Toggled by button) ---
doc_cols = st.columns([1, 1, 1])
with doc_cols[0]:
    if st.button("\U0001F4CB Documentation", key="footer_docs_btn", use_container_width=True):
        st.session_state.show_docs = not st.session_state.show_docs
with doc_cols[1]:
    st.link_button("\U0001F4BB GitHub", "https://github.com/JaniSaumil", use_container_width=True)
with doc_cols[2]:
    st.link_button("\U0001F310 LinkedIn", "https://www.linkedin.com/in/saumil-jani-281b92299", use_container_width=True)

import textwrap

if st.session_state.show_docs:
    st.markdown("""
<div style='
background: linear-gradient(135deg, rgba(15,15,35,0.95), rgba(26,26,62,0.95));
backdrop-filter: blur(20px);
border: 1px solid rgba(102,126,234,0.2);
border-radius: 20px;
padding: 40px 45px;
margin: 25px 0;
box-shadow: 0 12px 48px rgba(102,126,234,0.1), 0 0 80px rgba(102,126,234,0.05);
position: relative;
overflow: hidden;
'>
<div style='position:absolute; top:0; left:0; right:0; height:3px;
background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c, #fda085);
background-size: 300% 100%; animation: gradientShift 6s ease infinite;'></div>

<!-- Project Overview -->
<div style='margin-bottom: 32px;'>
<div style='display:flex; align-items:center; gap:10px; margin-bottom:14px;'>
<span style='font-size:1.5rem;'>\U0001F3AF</span>
<h3 style='margin:0; font-family:Poppins,sans-serif; font-weight:700; font-size:1.2rem;
background:linear-gradient(135deg,#667eea,#f093fb);
-webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;'>
Project Overview
</h3>
</div>
<p style='color:#c8c8e8; font-size:0.92rem; line-height:1.7; margin:0; font-family:Inter,sans-serif;'>
<strong style='color:#e8e8ff;'>Entertainment AI Hub</strong> is an intelligent media analytics platform
that tackles the problem of <strong style='color:#f093fb;'>content overload</strong> in today's entertainment landscape.
Using AI-powered recommendation engines, NLP-based sentiment analysis, and predictive
popularity modeling, it delivers <strong style='color:#a8b8ff;'>personalized, data-driven insights</strong>
to help users discover content they'll love.
</p>
</div>

<!-- Features -->
<div style='margin-bottom: 32px;'>
<div style='display:flex; align-items:center; gap:10px; margin-bottom:14px;'>
<span style='font-size:1.5rem;'>\u2728</span>
<h3 style='margin:0; font-family:Poppins,sans-serif; font-weight:700; font-size:1.2rem;
background:linear-gradient(135deg,#667eea,#f093fb);
-webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;'>
Key Features
</h3>
</div>
<div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
<div style='display:flex; align-items:center; gap:8px; padding:10px 14px;
background:rgba(102,126,234,0.08); border-radius:10px; border:1px solid rgba(102,126,234,0.12);'>
<span style='font-size:1.1rem;'>\U0001F916</span>
<span style='color:#c8c8e8; font-size:0.88rem; font-family:Inter,sans-serif;'>Personalized Movie Recommendations</span>
</div>
<div style='display:flex; align-items:center; gap:8px; padding:10px 14px;
background:rgba(240,147,251,0.08); border-radius:10px; border:1px solid rgba(240,147,251,0.12);'>
<span style='font-size:1.1rem;'>\U0001F4CA</span>
<span style='color:#c8c8e8; font-size:0.88rem; font-family:Inter,sans-serif;'>Interactive Data Visualization & EDA</span>
</div>
<div style='display:flex; align-items:center; gap:8px; padding:10px 14px;
background:rgba(74,222,128,0.08); border-radius:10px; border:1px solid rgba(74,222,128,0.12);'>
<span style='font-size:1.1rem;'>\U0001F50D</span>
<span style='color:#c8c8e8; font-size:0.88rem; font-family:Inter,sans-serif;'>NLP Sentiment Analysis of Reviews</span>
</div>
<div style='display:flex; align-items:center; gap:8px; padding:10px 14px;
background:rgba(245,87,108,0.08); border-radius:10px; border:1px solid rgba(245,87,108,0.12);'>
<span style='font-size:1.1rem;'>\U0001F680</span>
<span style='color:#c8c8e8; font-size:0.88rem; font-family:Inter,sans-serif;'>TMDB Popularity Score Prediction</span>
</div>
<div style='display:flex; align-items:center; gap:8px; padding:10px 14px;
background:rgba(253,160,133,0.08); border-radius:10px; border:1px solid rgba(253,160,133,0.12);'>
<span style='font-size:1.1rem;'>\U0001F4C4</span>
<span style='color:#c8c8e8; font-size:0.88rem; font-family:Inter,sans-serif;'>PDF Report Generation & Export</span>
</div>
<div style='display:flex; align-items:center; gap:8px; padding:10px 14px;
background:rgba(100,200,255,0.08); border-radius:10px; border:1px solid rgba(100,200,255,0.12);'>
<span style='font-size:1.1rem;'>\U0001F3AF</span>
<span style='color:#c8c8e8; font-size:0.88rem; font-family:Inter,sans-serif;'>Content-Based Genre Filtering</span>
</div>
</div>
</div>

<!-- Tech Stack -->
<div style='margin-bottom: 32px;'>
<div style='display:flex; align-items:center; gap:10px; margin-bottom:14px;'>
<span style='font-size:1.5rem;'>\u2699\uFE0F</span>
<h3 style='margin:0; font-family:Poppins,sans-serif; font-weight:700; font-size:1.2rem;
background:linear-gradient(135deg,#667eea,#f093fb);
-webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;'>
Tech Stack
</h3>
</div>
<div style='display:flex; flex-wrap:wrap; gap:10px;'>
<span style='padding:8px 18px; background:rgba(102,126,234,0.12); border:1px solid rgba(102,126,234,0.2);
border-radius:25px; color:#a8b8ff; font-family:Inter,sans-serif; font-size:0.82rem; font-weight:600;'>
\U0001F4CA Streamlit
</span>
<span style='padding:8px 18px; background:rgba(74,222,128,0.12); border:1px solid rgba(74,222,128,0.2);
border-radius:25px; color:#4ade80; font-family:Inter,sans-serif; font-size:0.82rem; font-weight:600;'>
\U0001F40D Python
</span>
<span style='padding:8px 18px; background:rgba(240,147,251,0.12); border:1px solid rgba(240,147,251,0.2);
border-radius:25px; color:#f093fb; font-family:Inter,sans-serif; font-size:0.82rem; font-weight:600;'>
\U0001F9E0 scikit-learn
</span>
<span style='padding:8px 18px; background:rgba(253,160,133,0.12); border:1px solid rgba(253,160,133,0.2);
border-radius:25px; color:#fda085; font-family:Inter,sans-serif; font-size:0.82rem; font-weight:600;'>
\U0001F4CA Pandas & NumPy
</span>
<span style='padding:8px 18px; background:rgba(245,87,108,0.12); border:1px solid rgba(245,87,108,0.2);
border-radius:25px; color:#f5576c; font-family:Inter,sans-serif; font-size:0.82rem; font-weight:600;'>
\U0001F3A8 Matplotlib & Seaborn
</span>
<span style='padding:8px 18px; background:rgba(100,200,255,0.12); border:1px solid rgba(100,200,255,0.2);
border-radius:25px; color:#64c8ff; font-family:Inter,sans-serif; font-size:0.82rem; font-weight:600;'>
\U0001F4DD NLTK (NLP)
</span>
</div>
</div>

<!-- How to Use -->
<div style='margin-bottom: 32px;'>
<div style='display:flex; align-items:center; gap:10px; margin-bottom:14px;'>
<span style='font-size:1.5rem;'>\U0001F4D6</span>
<h3 style='margin:0; font-family:Poppins,sans-serif; font-weight:700; font-size:1.2rem;
background:linear-gradient(135deg,#667eea,#f093fb);
-webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;'>
How to Use
</h3>
</div>
<div style='display:flex; gap:15px; flex-wrap:wrap;'>
<div style='flex:1; min-width:140px; text-align:center; padding:16px 12px;
background:rgba(15,15,35,0.6); border-radius:14px; border:1px solid rgba(255,255,255,0.06);'>
<div style='font-size:1.8rem; margin-bottom:6px;'>1\uFE0F\u20E3</div>
<div style='color:#c8c8e8; font-size:0.82rem; font-weight:600; font-family:Inter,sans-serif;'>Select a tab</div>
<div style='color:#7878a0; font-size:0.75rem; margin-top:4px;'>Choose your analysis</div>
</div>
<div style='flex:1; min-width:140px; text-align:center; padding:16px 12px;
background:rgba(15,15,35,0.6); border-radius:14px; border:1px solid rgba(255,255,255,0.06);'>
<div style='font-size:1.8rem; margin-bottom:6px;'>2\uFE0F\u20E3</div>
<div style='color:#c8c8e8; font-size:0.82rem; font-weight:600; font-family:Inter,sans-serif;'>Enter input</div>
<div style='color:#7878a0; font-size:0.75rem; margin-top:4px;'>Movie, review, or budget</div>
</div>
<div style='flex:1; min-width:140px; text-align:center; padding:16px 12px;
background:rgba(15,15,35,0.6); border-radius:14px; border:1px solid rgba(255,255,255,0.06);'>
<div style='font-size:1.8rem; margin-bottom:6px;'>3\uFE0F\u20E3</div>
<div style='color:#c8c8e8; font-size:0.82rem; font-weight:600; font-family:Inter,sans-serif;'>Click analyze</div>
<div style='color:#7878a0; font-size:0.75rem; margin-top:4px;'>Run the AI model</div>
</div>
<div style='flex:1; min-width:140px; text-align:center; padding:16px 12px;
background:rgba(15,15,35,0.6); border-radius:14px; border:1px solid rgba(255,255,255,0.06);'>
<div style='font-size:1.8rem; margin-bottom:6px;'>4\uFE0F\u20E3</div>
<div style='color:#c8c8e8; font-size:0.82rem; font-weight:600; font-family:Inter,sans-serif;'>View results</div>
<div style='color:#7878a0; font-size:0.75rem; margin-top:4px;'>Explore & export PDF</div>
</div>
</div>
</div>

<!-- Dataset Info -->
<div style='margin-bottom: 32px;'>
<div style='display:flex; align-items:center; gap:10px; margin-bottom:14px;'>
<span style='font-size:1.5rem;'>\U0001F4C1</span>
<h3 style='margin:0; font-family:Poppins,sans-serif; font-weight:700; font-size:1.2rem;
background:linear-gradient(135deg,#667eea,#f093fb);
-webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;'>
Dataset Information
</h3>
</div>
<div style='display:grid; grid-template-columns:1fr 1fr; gap:12px;'>
<div style='padding:16px 18px; background:rgba(15,15,35,0.6); border-radius:12px;
border:1px solid rgba(255,255,255,0.06);'>
<div style='color:#8888aa; font-size:0.75rem; text-transform:uppercase;
letter-spacing:1px; margin-bottom:6px; font-family:Inter,sans-serif;'>Source</div>
<div style='color:#e8e8ff; font-size:0.92rem; font-weight:600; font-family:Inter,sans-serif;'>
TMDB 5000 + MovieLens (Kaggle)
</div>
</div>
<div style='padding:16px 18px; background:rgba(15,15,35,0.6); border-radius:12px;
border:1px solid rgba(255,255,255,0.06);'>
<div style='color:#8888aa; font-size:0.75rem; text-transform:uppercase;
letter-spacing:1px; margin-bottom:6px; font-family:Inter,sans-serif;'>Scale</div>
<div style='color:#e8e8ff; font-size:0.92rem; font-weight:600; font-family:Inter,sans-serif;'>
62K+ movies, 25M+ ratings
</div>
</div>
<div style='padding:16px 18px; background:rgba(15,15,35,0.6); border-radius:12px;
border:1px solid rgba(255,255,255,0.06); grid-column: span 2;'>
<div style='color:#8888aa; font-size:0.75rem; text-transform:uppercase;
letter-spacing:1px; margin-bottom:6px; font-family:Inter,sans-serif;'>Key Columns</div>
<div style='color:#c8c8e8; font-size:0.88rem; font-family:Inter,sans-serif; line-height:1.6;'>
movieId, title, genres, userId, rating, budget, runtime, popularity, vote_average
</div>
</div>
</div>
</div>

<!-- Model Explanation -->
<div style='margin-bottom: 32px;'>
<div style='display:flex; align-items:center; gap:10px; margin-bottom:14px;'>
<span style='font-size:1.5rem;'>\U0001F9E0</span>
<h3 style='margin:0; font-family:Poppins,sans-serif; font-weight:700; font-size:1.2rem;
background:linear-gradient(135deg,#667eea,#f093fb);
-webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;'>
Model Architecture
</h3>
</div>
<div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;'>
<div style='padding:18px; background:rgba(102,126,234,0.06); border-radius:14px;
border:1px solid rgba(102,126,234,0.15);'>
<div style='color:#a8b8ff; font-family:Poppins,sans-serif; font-weight:700;
font-size:0.95rem; margin-bottom:8px;'>Content-Based Filtering</div>
<div style='color:#94A3B8; font-size:0.82rem; line-height:1.6; font-family:Inter,sans-serif;'>
Encodes movie genres into binary vectors and computes cosine similarity
to find movies with the most similar content DNA.
</div>
</div>
<div style='padding:18px; background:rgba(74,222,128,0.06); border-radius:14px;
border:1px solid rgba(74,222,128,0.15);'>
<div style='color:#4ade80; font-family:Poppins,sans-serif; font-weight:700;
font-size:0.95rem; margin-bottom:8px;'>Sentiment Analysis</div>
<div style='color:#94A3B8; font-size:0.82rem; line-height:1.6; font-family:Inter,sans-serif;'>
Converts review text into TF-IDF feature vectors, then classifies
sentiment as positive or negative using Logistic Regression.
</div>
</div>
<div style='padding:18px; background:rgba(245,87,108,0.06); border-radius:14px;
border:1px solid rgba(245,87,108,0.15);'>
<div style='color:#f5576c; font-family:Poppins,sans-serif; font-weight:700;
font-size:0.95rem; margin-bottom:8px;'>Popularity Prediction</div>
<div style='color:#94A3B8; font-size:0.82rem; line-height:1.6; font-family:Inter,sans-serif;'>
A Random Forest Regressor trained on budget, runtime, and genre features
from TMDB data to predict real-world popularity scores.
</div>
</div>
</div>
</div>

<!-- Future Improvements -->
<div>
<div style='display:flex; align-items:center; gap:10px; margin-bottom:14px;'>
<span style='font-size:1.5rem;'>\U0001F52E</span>
<h3 style='margin:0; font-family:Poppins,sans-serif; font-weight:700; font-size:1.2rem;
background:linear-gradient(135deg,#667eea,#f093fb);
-webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;'>
Future Roadmap
</h3>
</div>
<div style='display:flex; flex-wrap:wrap; gap:10px;'>
<span style='padding:8px 16px; background:rgba(102,126,234,0.08); border:1px solid rgba(102,126,234,0.15);
border-radius:10px; color:#a8b8ff; font-size:0.82rem; font-family:Inter,sans-serif; font-weight:500;'>
\U0001F310 Real-time streaming data integration
</span>
<span style='padding:8px 16px; background:rgba(240,147,251,0.08); border:1px solid rgba(240,147,251,0.15);
border-radius:10px; color:#f093fb; font-size:0.82rem; font-family:Inter,sans-serif; font-weight:500;'>
\U0001F9E0 Deep learning recommendation models
</span>
<span style='padding:8px 16px; background:rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.15);
border-radius:10px; color:#4ade80; font-size:0.82rem; font-family:Inter,sans-serif; font-weight:500;'>
\U0001F4F1 Mobile-responsive deployment
</span>
<span style='padding:8px 16px; background:rgba(253,160,133,0.08); border:1px solid rgba(253,160,133,0.15);
border-radius:10px; color:#fda085; font-size:0.82rem; font-family:Inter,sans-serif; font-weight:500;'>
\U0001F465 Collaborative filtering with user profiles
</span>
<span style='padding:8px 16px; background:rgba(100,200,255,0.08); border:1px solid rgba(100,200,255,0.15);
border-radius:10px; color:#64c8ff; font-size:0.82rem; font-family:Inter,sans-serif; font-weight:500;'>
\U0001F4E1 API deployment for third-party integration
</span>
</div>
</div>

</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='pro-footer'>
    <div class='footer-brand'>\U0001F3AC Entertainment AI Hub</div>
    <div class='footer-copy'>
        {BRANDING} &bull; v2.0 &bull; Built with Streamlit &bull; Powered by scikit-learn<br>
        &copy; 2026 Saumil Jani. All rights reserved.
    </div>
</div>
""", unsafe_allow_html=True)
