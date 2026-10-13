import streamlit as st

def render_dashboard_header(current_stage=None):
    """
    Renders sleek 6G Network Monitoring Control Center Header Shell with
    an ultra-compact visual Experiment Progress Bar across the actual experiment workflow:
    Image & Semantics → Network → Communication → Digital Twin → Prediction → Adaptive Control → Deep JSCC → Results
    """
    st.markdown(
        """
        <style>
            /* Universal App Dark Styling & Spacing */
            [data-testid="stSidebarNav"], ul[data-testid="stSidebarNavItems"] {
                display: none !important;
            }

            .main .block-container {
                padding-top: 1.2rem !important;
                padding-bottom: 2rem !important;
                max-width: 1350px;
            }
            
            /* Header Banner */
            .app-header-container {
                background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 12px 20px;
                margin-bottom: 16px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            .app-header-title {
                font-size: 20px;
                font-weight: 700;
                color: #F8FAFC;
                margin: 0;
                letter-spacing: -0.4px;
            }
            .app-header-subtitle {
                font-size: 12px;
                color: #94A3B8;
                margin-top: 2px;
            }
            .status-badge-live {
                display: inline-flex;
                align-items: center;
                background: rgba(16, 185, 129, 0.12);
                border: 1px solid #10B981;
                color: #10B981;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            .status-dot {
                height: 7px;
                width: 7px;
                background-color: #10B981;
                border-radius: 50%;
                display: inline-block;
                margin-right: 6px;
                box-shadow: 0 0 6px #10B981;
            }
        </style>
        
        <div class="app-header-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="app-header-title">6G Semantic Communication & Digital Twin Control Center</div>
                    <div class="app-header-subtitle">Deep-learning based semantic image transmission and network digital twin</div>
                </div>
                <div>
                    <span class="status-badge-live"><span class="status-dot"></span>SYSTEM LIVE</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_experiment_flow(current_stage=None):
    """Legacy helper kept for backward compatibility (no-op)."""
    pass

