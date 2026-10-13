import streamlit as st

def get_custom_css():
    return """
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Dark Tech Theme Backgrounds */
        .stApp {
            background-color: #0B0F19 !important;
            color: #F8FAFC !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #1E293B !important;
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem !important;
        }

        /* Hide duplicate native Streamlit multi-page sidebar navigation */
        [data-testid="stSidebarNav"], 
        [data-testid="stSidebarNavItems"], 
        ul[data-testid="stSidebarNavItems"], 
        div[data-testid="stSidebarNavSeparator"],
        nav[data-testid="stSidebarNav"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            opacity: 0 !important;
        }

        /* Main Container Width and Spacing */
        .main .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1350px;
        }

        /* Card Containers */
        .dark-card {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            padding: 20px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
            transition: border-color 0.2s ease !important;
        }
        .dark-card:hover {
            border-color: #38BDF8 !important;
        }

        .dark-card-header {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #38BDF8 !important;
            margin-bottom: 12px !important;
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
        }

        /* Metric Cards Styling */
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%) !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #38BDF8 !important;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.2) !important;
            transform: translateY(-2px) !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 22px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 12px !important;
        }

        /* Buttons Styling */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0369A1 0%, #1D4ED8 100%) !important;
            box-shadow: 0 6px 16px rgba(2, 132, 199, 0.5) !important;
            transform: translateY(-1px) !important;
        }
        div.stButton > button[kind="secondary"] {
            background-color: #1E293B !important;
            color: #CBD5E1 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
        }

        /* Streamlit Input Dark Theme Overrides */
        .stSelectbox label, .stSlider label, .stRadio label {
            color: #CBD5E1 !important;
            font-weight: 500 !important;
        }

        /* Option Menu Sidebar Overrides */
        .nav-link {
            font-size: 14px !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            margin: 3px 0 !important;
        }

        /* Footer styling */
        .footer-container {
            margin-top: 40px;
            padding: 16px;
            background-color: #0F172A;
            border-top: 1px solid #334155;
            border-radius: 8px;
            text-align: center;
            font-size: 12px;
            color: #64748B;
        }
        .footer-note {
            color: #38BDF8;
            font-weight: 600;
        }
    </style>
    """

def render_footer():
    st.markdown(
        """
        <div class="footer-container">
            <span class="footer-note">🛰️ Digital Twin 6G Semantic Communication Platform</span> | 
            Telemetry Values: <strong>Calculated & Simulated in Real-Time</strong> | 
            ResNet-18 + Deep JSCC + LSTM Predictor + Adaptive Controller
        </div>
        """,
        unsafe_allow_html=True
    )
