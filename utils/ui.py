"""
Shared UI styling and helpers for Streamlit pages.
"""

import streamlit as st

# Minimal, modern palette
TEXT = "#111827"     # near-black text
ACCENT = "#2563EB"   # blue-600 for primary actions/links
SUCCESS = "#10B981"  # emerald-500
WARNING = "#F59E0B"  # amber-500
DANGER = "#EF4444"   # red-500
MUTED = "#6B7280"    # gray-500
BG = "#FAFAFA"       # light background
CARD_BG = "#FFFFFF"  # white cards


def apply_theme():
    """Inject a clean, modern theme with black text and light backgrounds.
    Call near top of each page.
    """
    st.markdown(
        f"""
        <style>
            /* Import Inter font */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            html, body, [class*="css"]  {{
                font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
                color: {TEXT};
                background: {BG};
            }}
            .stApp {{ background: {BG}; }}
            .main-header {{
                font-size: 2.2rem; font-weight: 700; color: {TEXT}; margin: 0.5rem 0 1rem 0;
            }}
            .sub-header {{
                font-size: 1.2rem; font-weight: 600; color: {TEXT}; margin: 1rem 0 0.5rem 0;
            }}
            .metric-card {{
                background: {CARD_BG}; border: 1px solid #E5E7EB;
                padding: 1rem; border-radius: 10px; color: {TEXT}; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            }}
            .info-box {{
                background-color: #EEF2FF; border-left: 4px solid {ACCENT}; padding: 0.8rem 1rem; border-radius: 10px; color: {TEXT};
            }}
            .warning-box {{
                background-color: #FFF7ED; border-left: 4px solid {WARNING}; padding: 0.8rem 1rem; border-radius: 10px; color: {TEXT};
            }}
            .success-box {{
                background-color: #ECFDF5; border-left: 4px solid {SUCCESS}; padding: 0.8rem 1rem; border-radius: 10px; color: {TEXT};
            }}
            .danger-box {{
                background-color: #FEF2F2; border-left: 4px solid {DANGER}; padding: 0.8rem 1rem; border-radius: 10px; color: {TEXT};
            }}
            .card {{
                background: {CARD_BG}; border: 1px solid #E5E7EB; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            }}
            .muted {{ color: {MUTED}; }}

            /* Sidebar: light */
            div[data-testid="stSidebar"] {{
                background: {CARD_BG};
                border-right: 1px solid #E5E7EB;
            }}
            div[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

            /* Buttons */
            .stButton>button {{
                background: {ACCENT};
                color: white; border: none; border-radius: 8px; padding: 0.5rem 1.0rem; font-weight: 600;
                transition: transform 0.12s ease, box-shadow 0.12s ease; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            }}
            .stButton>button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
                filter: brightness(0.95);
            }}
            .stButton>button:active {{ transform: translateY(0); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, emoji: str = ""):
    """Render a consistent page header."""
    icon = f"{emoji} " if emoji else ""
    st.markdown(f'<h1 class="main-header">{icon}{title}</h1>', unsafe_allow_html=True)
