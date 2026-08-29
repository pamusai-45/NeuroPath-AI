"""
NeuroPath AI — Clinical Decision-Support Research Prototype
Streamlit frontend for the NeuroPath V2 tabular ML model.

Run with:
    streamlit run app/app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "neuropath_v2_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "neuropath_v2_metadata.json"

DISCLAIMER_TEXT = (
    "NeuroPath AI is a research prototype and clinical decision-support tool. "
    "The score should not be interpreted as a diagnosis or as a clinically "
    "validated probability of Alzheimer's disease."
)

FALLBACK_THRESHOLDS = {"low": 0.20, "high": 0.40}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroPath AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>

        /* ============================================================
           FORCE LIGHT THEME REGARDLESS OF BROWSER / STREAMLIT SETTING
           (belt-and-suspenders alongside .streamlit/config.toml)
           ============================================================ */

        :root, html[data-theme="dark"], body {
            --primary-color: #7c3aed !important;
            --background-color: #f5f6fb !important;
            --secondary-background-color: #ffffff !important;
            --text-color: #1c2140 !important;
        }

        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        [data-testid="stMain"] {
            background-color: #f5f6fb !important;
            color: #1c2140 !important;
        }

        /* Hide the default Streamlit chrome (deploy bar / menu / footer)
           so the layout matches the target design cleanly. */
        header[data-testid="stHeader"] {
            background-color: #f5f6fb !important;
            box-shadow: none !important;
        }
        #MainMenu, footer {
            visibility: hidden;
        }

        /* ---------- Blanket label fix (main content area only) ----------
           Cast the widest possible net so labels are never invisible,
           regardless of Streamlit version or which testid wraps them. */
        [data-testid="stMain"] label,
        [data-testid="stMain"] label p,
        [data-testid="stMain"] label span,
        [data-testid="stMain"] div[data-testid="stWidgetLabel"],
        [data-testid="stMain"] div[data-testid="stWidgetLabel"] p,
        [data-testid="stMain"] div[data-testid="stWidgetLabel"] span,
        [data-testid="stMain"] .stMarkdown p,
        [data-testid="stMain"] .stMarkdown li,
        [data-testid="stMain"] .stMarkdown span,
        [data-testid="stMain"] p,
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4 {
            color: #1c2140 !important;
        }

        /* ---------- Main content padding ---------- */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1300px;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1130 0%, #131b45 100%) !important;
        }
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        /* Sidebar radio nav: hide native circle, style label as a pill */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.4rem;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background-color: rgba(255, 255, 255, 0.04);
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
            display: flex;
            align-items: center;
            width: 100%;
            cursor: pointer;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: linear-gradient(90deg, #7c3aed 0%, #4f46e5 100%) !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg, #7c3aed 0%, #4f46e5 100%) !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
        }

        /* ---------- Headings ---------- */
        h1, h2, h3 {
            color: #14183a !important;
        }

        /* ---------- Widget labels (main area) must be dark ---------- */
        div[data-testid="stAppViewContainer"] div[data-testid="stWidgetLabel"] p,
        div[data-testid="stAppViewContainer"] div[data-testid="stWidgetLabel"] label {
            color: #1c2140 !important;
            font-weight: 600 !important;
        }

        /* ---------- Number / text inputs ---------- */
        [data-testid="stMain"] div[data-testid="stNumberInput"] input,
        [data-testid="stMain"] div[data-testid="stTextInput"] input,
        [data-testid="stMain"] input {
            background-color: #ffffff !important;
            color: #1c2140 !important;
            border: 1px solid #d8dae8 !important;
            -webkit-text-fill-color: #1c2140 !important;
            caret-color: #1c2140 !important;
        }
        [data-testid="stMain"] div[data-testid="stNumberInput"] button {
            background-color: #ffffff !important;
            border: 1px solid #d8dae8 !important;
        }
        [data-testid="stMain"] div[data-testid="stNumberInput"] button svg {
            fill: #1c2140 !important;
        }

        /* ---------- Selectbox ---------- */
        [data-testid="stMain"] div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            background-color: #ffffff !important;
        }
        [data-testid="stMain"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border: 1px solid #d8dae8 !important;
        }
        [data-testid="stMain"] div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
        [data-testid="stMain"] div[data-testid="stSelectbox"] svg {
            color: #1c2140 !important;
            fill: #1c2140 !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: #ffffff !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li,
        ul[data-testid="stSelectboxVirtualDropdown"] li * {
            color: #1c2140 !important;
            background-color: #ffffff !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
            background-color: #f4f0fd !important;
        }

        /* ---------- Slider (force purple, override default red accent) ---------- */
        [data-testid="stMain"] div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background-color: #7c3aed !important;
            border-color: #7c3aed !important;
        }
        [data-testid="stMain"] div[data-testid="stSlider"] [data-baseweb="slider"] div {
            background-color: #7c3aed !important;
        }
        [data-testid="stMain"] div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
            background: #e2d7fb !important;
        }
        [data-testid="stMain"] div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
        [data-testid="stMain"] div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
            color: #6b6f85 !important;
        }
        [data-testid="stMain"] div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
            color: #6d28d9 !important;
            font-weight: 700 !important;
        }

        /* ---------- Progress bar ---------- */
        div[data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #7c3aed 0%, #2563eb 100%) !important;
        }

        /* ---------- Bordered container used for the Patient Assessment card ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px !important;
            box-shadow: 0 4px 18px rgba(30, 34, 90, 0.06);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            border-radius: 16px !important;
            background-color: #ffffff !important;
        }

        /* ---------- Static HTML cards (single-call blocks only) ---------- */
        .np-card {
            background-color: #ffffff;
            border: 1px solid #e7e8f2;
            border-radius: 16px;
            padding: 1.75rem 2rem;
            box-shadow: 0 4px 18px rgba(30, 34, 90, 0.06);
            margin-bottom: 1.5rem;
        }

        .np-card-heading {
            color: #6d28d9;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .np-info-banner {
            background-color: #e8f0fe;
            border: 1px solid #c9dcfb;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            color: #1c3a6e;
            margin-bottom: 1.25rem;
            font-size: 0.95rem;
        }

        .np-clinical-card {
            background: #f4f0fd;
            border: 1px solid #e2d7fb;
            border-radius: 14px;
            padding: 1.1rem 1.4rem;
            color: #3b2c66;
            margin-top: 0.5rem;
        }

        .np-purple-support-card {
            background: #f4f0fd;
            border: 1px solid #e2d7fb;
            border-radius: 14px;
            padding: 1rem 1.25rem;
            color: #3b2c66;
            font-size: 0.9rem;
        }

        .np-sidebar-warning {
            background: #241c0d;
            border: 1px solid #7a5b12;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 1rem;
        }
        .np-sidebar-warning * {
            color: #f5c451 !important;
        }
        .np-sidebar-warning .np-warning-body {
            color: #f0e6c8 !important;
            font-weight: 400 !important;
        }

        /* ---------- Analyze button ---------- */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #7c3aed 0%, #4f46e5 50%, #2563eb 100%);
            color: #ffffff;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 0.85rem 0;
            border: none;
            border-radius: 12px;
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
            transition: transform 0.08s ease-in-out, box-shadow 0.08s ease-in-out;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(79, 70, 229, 0.45);
        }
        div.stButton > button:active {
            transform: translateY(0px);
        }

        /* ---------- Metric cards ---------- */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e7e8f2;
            border-radius: 14px;
            padding: 1rem 1.25rem;
            box-shadow: 0 4px 14px rgba(30, 34, 90, 0.05);
        }
        div[data-testid="stMetric"] label {
            color: #5b5f7a !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #14183a !important;
        }

        /* ---------- Priority banners ---------- */
        .np-priority-low {
            background-color: #e7f7ec;
            border: 1px solid #a7e3ba;
            color: #14532d;
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
        }
        .np-priority-medium {
            background-color: #fff8e6;
            border: 1px solid #f4dd9a;
            color: #7a5b06;
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
        }
        .np-priority-high {
            background-color: #fdecec;
            border: 1px solid #f4a9a9;
            color: #7a1414;
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
        }

        .np-disclaimer {
            font-size: 0.82rem;
            color: #6b6f85;
            border-top: 1px solid #e7e8f2;
            padding-top: 0.9rem;
            margin-top: 1.5rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA / MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained sklearn pipeline. Returns None if missing."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@st.cache_data
def load_metadata():
    """Load model metadata JSON. Returns {} if missing or invalid."""
    if not METADATA_PATH.exists():
        return {}
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def get_thresholds(metadata: dict) -> dict:
    """
    Resolve LOW/HIGH thresholds from metadata, supporting either
    {"low":..,"high":..} or {"t_low":..,"t_high":..}. Falls back to
    default thresholds if metadata is missing or malformed.
    """
    thresholds = metadata.get("thresholds", {}) if isinstance(metadata, dict) else {}

    low = thresholds.get("low", thresholds.get("t_low"))
    high = thresholds.get("high", thresholds.get("t_high"))

    if low is None:
        low = FALLBACK_THRESHOLDS["low"]
    if high is None:
        high = FALLBACK_THRESHOLDS["high"]

    return {"low": float(low), "high": float(high)}


def get_priority(probability: float, thresholds: dict) -> dict:
    """Map a probability to a priority level, color, and recommended action."""
    if probability < thresholds["low"]:
        return {
            "level": "LOW",
            "color": "low",
            "action": "Routine follow-up interval; no immediate escalation indicated.",
        }
    elif probability < thresholds["high"]:
        return {
            "level": "MEDIUM",
            "color": "medium",
            "action": "Consider prioritizing for further clinical assessment.",
        }
    else:
        return {
            "level": "HIGH",
            "color": "high",
            "action": "Recommend prioritizing for prompt clinical assessment.",
        }


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.1rem;">
                <div style="width:2.6rem; height:2.6rem; border-radius:50%;
                            background: linear-gradient(135deg, #ec4899 0%, #7c3aed 100%);
                            display:flex; align-items:center; justify-content:center;
                            font-size:1.3rem; flex-shrink:0;
                            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.45);">
                    🧠
                </div>
                <span style="font-size:1.45rem; font-weight:800;">NeuroPath AI</span>
            </div>
            <div style="color:#c7cbe8; font-size:0.9rem; margin-bottom:1.5rem;">
                Clinical Decision-Support Prototype
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='color:#a5abd6; font-size:0.78rem; letter-spacing:0.08em; "
            "font-weight:700; margin-bottom:0.4rem;'>NAVIGATION</div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            options=["Patient Assessment", "How It Works", "Model Information"],
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

        st.markdown(
            "<div style='color:#a5abd6; font-size:0.78rem; letter-spacing:0.08em; "
            "font-weight:700; margin-bottom:0.6rem;'>CURRENT SYSTEM</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="font-size:0.92rem; line-height:1.9;">
                ⚙️ <b>Model:</b> NeuroPath V2<br>
                🎯 <b>Purpose:</b> Patient prioritization<br>
                🛡️ <b>Output:</b> Risk score + priority
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="np-sidebar-warning">
                ⚠️ <b>Research prototype only.</b><br>
                <span class="np-warning-body">Not intended to diagnose Alzheimer's disease.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # render_sidebar_brain_illustration()

    return page


def render_sidebar_brain_illustration() -> None:
    """Decorative glowing neon-brain SVG for the sidebar."""

    svg = """
    <div style="
        width:100%;
        display:flex;
        justify-content:center;
        align-items:center;
        margin-top:20px;
        background:transparent;
    ">

        <svg
            width="190"
            height="150"
            viewBox="0 0 190 150"
            xmlns="http://www.w3.org/2000/svg"
        >

            <defs>

                <radialGradient id="npGlow" cx="50%" cy="40%" r="65%">
                    <stop
                        offset="0%"
                        stop-color="#7c3aed"
                        stop-opacity="0.35"
                    />

                    <stop
                        offset="100%"
                        stop-color="#7c3aed"
                        stop-opacity="0"
                    />
                </radialGradient>


                <linearGradient
                    id="npBrainStroke"
                    x1="0%"
                    y1="0%"
                    x2="100%"
                    y2="100%"
                >
                    <stop offset="0%" stop-color="#22d3ee"/>
                    <stop offset="50%" stop-color="#a78bfa"/>
                    <stop offset="100%" stop-color="#ec4899"/>
                </linearGradient>

            </defs>


            <!-- Glow -->
            <circle
                cx="95"
                cy="65"
                r="70"
                fill="url(#npGlow)"
            />


            <!-- Brain -->
            <g
                fill="none"
                stroke="url(#npBrainStroke)"
                stroke-width="1.4"
                stroke-linecap="round"
            >

                <path d="
                    M60 35
                    C45 35 35 48 35 62
                    C28 66 24 76 28 86
                    C25 96 32 106 43 108
                    C46 118 58 124 68 120
                    C74 126 86 126 92 119
                    C100 124 112 120 115 111
                    C126 111 134 101 132 90
                    C140 84 140 70 130 63
                    C132 50 122 38 108 38
                    C104 30 92 27 84 32
                    C77 27 66 28 60 35 Z
                "/>

                <path d="
                    M95 40
                    C95 60 90 70 95 90
                    C98 105 95 112 92 119
                "/>

                <path d="
                    M60 35
                    C65 45 65 55 58 62
                    C52 68 52 78 60 84
                "/>

                <path d="
                    M108 38
                    C104 48 106 58 114 64
                    C120 70 120 80 112 87
                "/>

                <path d="
                    M35 62
                    C45 65 52 72 50 82
                "/>

                <path d="
                    M132 63
                    C122 66 116 74 118 84
                "/>

                <path d="
                    M43 108
                    C52 104 58 96 56 88
                "/>

                <path d="
                    M115 111
                    C108 104 104 96 108 88
                "/>


                <!-- Neural points -->
                <circle
                    cx="95"
                    cy="55"
                    r="2"
                    fill="#22d3ee"
                    stroke="none"
                />

                <circle
                    cx="70"
                    cy="70"
                    r="1.6"
                    fill="#a78bfa"
                    stroke="none"
                />

                <circle
                    cx="118"
                    cy="75"
                    r="1.6"
                    fill="#ec4899"
                    stroke="none"
                />

                <circle
                    cx="85"
                    cy="100"
                    r="1.6"
                    fill="#22d3ee"
                    stroke="none"
                />

            </g>


            <!-- Spinal cord -->
            <path
                d="M92 119 C90 128 90 136 88 144"
                fill="none"
                stroke="url(#npBrainStroke)"
                stroke-width="1.4"
                stroke-linecap="round"
            />

        </svg>

    </div>
    """

    components.html(
        svg,
        height=180,
        scrolling=False
    )

# ============================================================
# PATIENT ASSESSMENT PAGE
# ============================================================

def render_header() -> None:
    col_title, col_card = st.columns([3, 1.3])

    with col_title:
        st.markdown(
            "<div style='font-size:2.6rem; font-weight:800; color:#14183a; "
            "line-height:1.1;'>NeuroPath AI</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:1.1rem; color:#4a4e68; margin-top:0.2rem;'>"
            "AI-Driven Prioritization System for Early Alzheimer's Diagnostic Pathways"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_card:
        st.markdown(
            """
            <div class="np-purple-support-card">
                🛡️ Supporting clinicians in identifying patients who may benefit
                from earlier assessment.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="np-info-banner">
            ℹ️ Enter patient information below. The system estimates a research
            risk score and assigns a priority level for further clinical assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_patient_form():
    # NOTE: st.container(border=True) is used here (rather than a raw HTML
    # <div>) because Streamlit widgets called across multiple st.markdown()
    # calls do NOT nest inside an HTML div opened in an earlier call - each
    # call renders as a sibling in the DOM. A native bordered container is
    # the correct way to visually group real widgets into one card.
    with st.container(border=True):
        st.markdown(
            '<div class="np-card-heading">👥 Patient Assessment</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=110,
                value=72,
                step=1,
                key="input_age",
            )
            education = st.number_input(
                "Years of Education",
                min_value=0,
                max_value=30,
                value=16,
                step=1,
                key="input_education",
            )
            nwbv = st.number_input(
                "Normalized Whole Brain Volume (nWBV)",
                min_value=0.40,
                max_value=1.00,
                value=0.720,
                step=0.001,
                format="%.3f",
                key="input_nwbv",
            )

        with col2:
            sex = st.selectbox(
                "Sex",
                options=["M", "F"],
                index=0,
                key="input_sex",
            )
            ses = st.selectbox(
                "Socioeconomic Status (SES)",
                options=[1, 2, 3, 4, 5],
                index=1,
                key="input_ses",
            )

        with col3:
            mmse = st.slider(
                "MMSE Score (0–30)",
                min_value=0,
                max_value=30,
                value=26,
                step=1,
                key="input_mmse",
            )
            etiv = st.number_input(
                "Estimated Total Intracranial Volume (eTIV)",
                min_value=500,
                max_value=2500,
                value=1500,
                step=1,
                key="input_etiv",
            )

        analyze_clicked = st.button("🧠 Analyze Patient", key="analyze_button")

    st.markdown(
        """
        <div class="np-clinical-card">
            💡 <b>Clinical Tip</b><br>
            All inputs correspond to features used by the trained model.
            Ensure values are entered as accurately as possible.
        </div>
        """,
        unsafe_allow_html=True,
    )

    inputs = {
        "Age": age,
        "M/F": sex,
        "EDUC": education,
        "SES": ses,
        "MMSE": mmse,
        "eTIV": etiv,
        "nWBV": nwbv,
    }

    return inputs, analyze_clicked


def render_results(inputs: dict, model, metadata: dict) -> None:
    patient_df = pd.DataFrame({
        "Age": [inputs["Age"]],
        "M/F": [inputs["M/F"]],
        "EDUC": [inputs["EDUC"]],
        "SES": [inputs["SES"]],
        "MMSE": [inputs["MMSE"]],
        "eTIV": [inputs["eTIV"]],
        "nWBV": [inputs["nWBV"]],
    })

    try:
        probability = model.predict_proba(patient_df)[0][1]
    except Exception as e:
        st.error(f"Prediction failed. Please check patient inputs and try again.\n\nDetails: {e}")
        return

    risk_score = probability * 100
    thresholds = get_thresholds(metadata)
    priority = get_priority(probability, thresholds)

    st.markdown("### Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Score", f"{risk_score:.1f} / 100")
    m2.metric("Model Probability", f"{probability * 100:.1f}%")
    m3.metric("Priority", priority["level"])

    priority_css_class = {
        "LOW": "np-priority-low",
        "MEDIUM": "np-priority-medium",
        "HIGH": "np-priority-high",
    }[priority["level"]]

    st.markdown(
        f"""
        <div class="{priority_css_class}">
            <div style="font-size:1.2rem; font-weight:800; margin-bottom:0.4rem;">
                Priority: {priority["level"]}
            </div>
            <div>{priority["action"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(min(max(probability, 0.0), 1.0))

    st.markdown(
        "<div style='color:#4a4e68; font-size:0.92rem; margin-top:0.5rem;'>"
        "Higher scores indicate greater priority for additional clinical evaluation "
        "according to the current prototype model."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Patient Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Age", f"{inputs['Age']}")
    s2.metric("MMSE", f"{inputs['MMSE']}")
    s3.metric("nWBV", f"{inputs['nWBV']:.3f}")
    s4.metric("Education", f"{inputs['EDUC']} yrs")

    st.markdown("#### Explanation")
    st.info(
        "The current deployment provides the model output, but patient-level "
        "feature attribution is not yet enabled."
    )

    st.markdown(f"<div class='np-disclaimer'>{DISCLAIMER_TEXT}</div>", unsafe_allow_html=True)


def render_patient_assessment(model, metadata: dict) -> None:
    render_header()

    if model is None:
        st.error(
            f"Model file not found at expected path:\n\n`{MODEL_PATH}`\n\n"
            "Please ensure the trained model file is present before running a prediction."
        )

    inputs, analyze_clicked = render_patient_form()

    if analyze_clicked:
        if model is None:
            st.error("Cannot run prediction: model file is missing.")
        else:
            render_results(inputs, model, metadata)


# ============================================================
# HOW IT WORKS PAGE
# ============================================================

def render_how_it_works() -> None:
    st.markdown(
        "<div style='font-size:2rem; font-weight:800; color:#14183a;'>How It Works</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#4a4e68; margin-bottom:1.5rem;'>"
        "An overview of the current NeuroPath V2 pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    steps = [
        ("1", "Patient Information", "Clinician enters demographic, cognitive, and imaging-derived measures."),
        ("2", "Preprocessing", "Inputs are structured into the exact feature format expected by the trained pipeline."),
        ("3", "ML Risk Estimation", "The NeuroPath V2 model outputs a probability score from the tabular inputs."),
        ("4", "Prioritization", "The probability is mapped to a LOW / MEDIUM / HIGH priority level using configured thresholds."),
        ("5", "Clinical Decision Support", "The priority and recommended action are presented to support — not replace — clinician judgment."),
    ]

    for i, (num, title, desc) in enumerate(steps):
        st.markdown(
            f"""
            <div class="np-card" style="padding:1.2rem 1.75rem; margin-bottom:0.75rem;">
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div style="background:linear-gradient(90deg,#7c3aed,#4f46e5);
                                color:white; font-weight:800; width:2.2rem; height:2.2rem;
                                border-radius:50%; display:flex; align-items:center;
                                justify-content:center; flex-shrink:0;">
                        {num}
                    </div>
                    <div>
                        <div style="font-weight:700; color:#14183a; font-size:1.05rem;">{title}</div>
                        <div style="color:#4a4e68; font-size:0.92rem;">{desc}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if i < len(steps) - 1:
            st.markdown(
                "<div style='text-align:center; color:#a5abd6; font-size:1.3rem; "
                "margin:-0.3rem 0 0.3rem 0;'>↓</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="np-card">
            <div class="np-card-heading">Current V2 Features</div>
            <ul style="color:#1c2140; line-height:1.8;">
                <li>Age</li>
                <li>Sex</li>
                <li>Years of Education</li>
                <li>Socioeconomic Status (SES)</li>
                <li>MMSE Score</li>
                <li>Estimated Total Intracranial Volume (eTIV)</li>
                <li>Normalized Whole Brain Volume (nWBV)</li>
            </ul>
            <p style="color:#4a4e68;">The current V2 model is a <b>tabular machine
            learning model</b> trained only on the structured features listed above.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="np-clinical-card">
            🗺️ <b>Future Roadmap (not yet implemented)</b><br>
            Potential future directions being explored include incorporating blood
            biomarkers, additional MRI-derived information, and multimodal modeling
            approaches. These are <b>not</b> part of the current V2 deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='np-disclaimer'>{DISCLAIMER_TEXT}</div>", unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

def render_model_information(metadata: dict) -> None:
    st.markdown(
        "<div style='font-size:2rem; font-weight:800; color:#14183a;'>Model Information</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="np-card">
                <div class="np-card-heading">Overview</div>
                <p style="color:#1c2140;"><b>Model:</b> NeuroPath AI V2</p>
                <p style="color:#1c2140;"><b>Purpose:</b> CDR-defined patient
                prioritization research prototype</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="np-card">
                <div class="np-card-heading">Input Features</div>
                <ul style="color:#1c2140; line-height:1.8;">
                    <li>Age</li>
                    <li>Sex (M/F)</li>
                    <li>Years of Education (EDUC)</li>
                    <li>Socioeconomic Status (SES)</li>
                    <li>MMSE Score</li>
                    <li>Estimated Total Intracranial Volume (eTIV)</li>
                    <li>Normalized Whole Brain Volume (nWBV)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        thresholds = get_thresholds(metadata)
        st.markdown(
            f"""
            <div class="np-card">
                <div class="np-card-heading">Priority Thresholds</div>
                <p style="color:#1c2140;">🟢 <b>LOW:</b> probability &lt; {thresholds['low']:.2f}</p>
                <p style="color:#1c2140;">🟡 <b>MEDIUM:</b> {thresholds['low']:.2f} ≤ probability &lt; {thresholds['high']:.2f}</p>
                <p style="color:#1c2140;">🔴 <b>HIGH:</b> probability ≥ {thresholds['high']:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
        if metrics:
            metrics_html = "".join(
                f"<p style='color:#1c2140;'><b>{key}:</b> {value}</p>"
                for key, value in metrics.items()
            )
            st.markdown(
                f"""
                <div class="np-card">
                    <div class="np-card-heading">Model Metrics</div>
                    {metrics_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="np-card">
            <div class="np-card-heading">Limitations</div>
            <ul style="color:#1c2140; line-height:1.8;">
                <li>Trained on a limited tabular dataset; performance on other
                populations is not established.</li>
                <li>Does not diagnose Alzheimer's disease and does not predict
                future conversion to Alzheimer's disease.</li>
                <li>Feature-level explanation is not yet enabled in this deployment.</li>
                <li>Intended strictly as a research-stage clinical decision-support
                aid, not a standalone diagnostic tool.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='np-disclaimer'>{DISCLAIMER_TEXT}</div>", unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    inject_css()

    model = load_model()
    metadata = load_metadata()

    page = render_sidebar()

    if page == "Patient Assessment":
        render_patient_assessment(model, metadata)
    elif page == "How It Works":
        render_how_it_works()
    elif page == "Model Information":
        render_model_information(metadata)


if __name__ == "__main__":
    main()
