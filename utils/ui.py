"""
Shared UI styling and helpers for Streamlit pages.
"""

import streamlit as st

PRIMARY = "#1E3A8A"  # Indigo-900
ACCENT = "#2563EB"   # Blue-600
SUCCESS = "#10B981"  # Emerald-500
WARNING = "#F59E0B"  # Amber-500
DANGER = "#EF4444"   # Red-500
MUTED = "#6B7280"    # Gray-500
BG_SOFT = "#F8FAFC"  # Slate-50


def apply_theme():
    """Inject modern CSS theme (fonts, colors, components). Call near top of each page."""
    st.markdown(
        f"""
        <style>
            /* Import Inter font */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            html, body, [class*="css"]  {{
                font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
            }}
            .main-header {{
                font-size: 2.2rem; font-weight: 700; color: {PRIMARY}; margin: 0.5rem 0 1rem 0;
            }}
            .sub-header {{
                font-size: 1.2rem; font-weight: 600; color: {ACCENT}; margin: 1rem 0 0.5rem 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem; border-radius: 12px; color: white; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            }}
            .info-box {{
                background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 0.8rem 1rem; border-radius: 10px;
            }}
            .warning-box {{
                background-color: #FEF3C7; border-left: 4px solid {WARNING}; padding: 0.8rem 1rem; border-radius: 10px;
            }}
            .success-box {{
                background-color: #D1FAE5; border-left: 4px solid {SUCCESS}; padding: 0.8rem 1rem; border-radius: 10px;
            }}
            .danger-box {{
                background-color: #FEE2E2; border-left: 4px solid {DANGER}; padding: 0.8rem 1rem; border-radius: 10px;
            }}
            .card {{
                background: white; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            }}
            .muted {{ color: {MUTED}; }}

            /* Sidebar gradient */
            div[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {PRIMARY} 0%, #3B82F6 100%);
            }}
            div[data-testid="stSidebar"] * {{ color: white !important; }}

            /* Buttons */
            .stButton>button {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; border: none; border-radius: 10px; padding: 0.5rem 1.2rem; font-weight: 600;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }}
            .stButton>button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.12);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, emoji: str = ""):
    """Render a consistent page header."""
    icon = f"{emoji} " if emoji else ""
    st.markdown(f'<h1 class="main-header">{icon}{title}</h1>', unsafe_allow_html=True)
