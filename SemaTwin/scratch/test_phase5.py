import subprocess
import time
from playwright.sync_api import sync_playwright

proc = subprocess.Popen([
    r"c:\Users\shree\Downloads\Bappa\venv\Scripts\python.exe",
    "-m", "streamlit", "run", "dashboard/pages/4_Communication.py",
    "--server.port", "8514",
    "--server.headless", "true"
])

time.sleep(5)

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('http://localhost:8514')
        btn = page.wait_for_selector('button:has-text("START TRANSMISSION")', timeout=30000)
        if btn:
            btn.click()
            page.wait_for_timeout(6000)
            
        page.evaluate("window.scrollTo(0, 800)")
        page.wait_for_timeout(1000)
        page.screenshot(path='C:/Users/shree/.gemini/antigravity/brain/db09aa0c-0a89-4cb3-a053-eb7f6d21b157/phase5_communication_results_screenshot.png')
        browser.close()
    print("Scrolled 800 Phase 5 Communication screenshot captured successfully")
except Exception as err:
    print(f"Error during screenshot: {err}")
finally:
    proc.terminate()
