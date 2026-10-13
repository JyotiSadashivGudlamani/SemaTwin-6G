import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        torch_lib_path = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

def test_navigation():
    print("=== TESTING BUTTON NAVIGATION FLOW ACROSS ALL PAGES ===")

    # Mock Streamlit session_state
    class SessionState(dict):
        def __getattr__(self, key):
            return self.get(key, None)
        def __setattr__(self, key, value):
            self[key] = value

    class CustomRerunException(Exception):
        pass

    import streamlit as st
    st.session_state = SessionState()
    st.rerun = lambda: (_ for _ in ()).throw(CustomRerunException())

    from dashboard.app import navigate_to, PAGE_LIST

    nav_chain = [
        ("Overview", "Image & Semantics"),
        ("Image & Semantics", "Network"),
        ("Network", "Communication"),
        ("Communication", "Digital Twin"),
        ("Digital Twin", "Prediction"),
        ("Prediction", "Adaptive Control"),
        ("Adaptive Control", "Results"),
        ("Results", "Overview")
    ]

    for start_page, target_page in nav_chain:
        st.session_state["current_page"] = start_page
        st.session_state["app_sidebar_option_menu"] = start_page

        try:
            navigate_to(target_page)
        except CustomRerunException:
            pass

        # Simulate top-of-script run on rerun
        if "pending_nav_section" in st.session_state:
            raw_pending = st.session_state.pop("pending_nav_section")
            clean_pending = raw_pending.strip()
            if clean_pending in PAGE_LIST:
                st.session_state["current_page"] = clean_pending
                st.session_state["app_sidebar_option_menu"] = clean_pending

        assert st.session_state["current_page"] == target_page, f"Expected {target_page}, got {st.session_state['current_page']}"
        assert st.session_state["app_sidebar_option_menu"] == target_page, f"Expected sidebar menu to be {target_page}, got {st.session_state['app_sidebar_option_menu']}"
        print(f"[OK] Transition: {start_page} -> {target_page} PASSED!")

    print("\n[OK] ALL PAGE CONTINUATION BUTTON TRANSITIONS WORK PERFECTLY!")

if __name__ == "__main__":
    test_navigation()
