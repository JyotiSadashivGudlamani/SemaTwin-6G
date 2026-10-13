import subprocess
import time
from playwright.sync_api import sync_playwright

proc = subprocess.Popen([
    r"c:\Users\shree\Downloads\Bappa\venv\Scripts\python.exe",
    "-m", "streamlit", "run", "dashboard/pages/3_Network_Topology.py",
    "--server.port", "8509",
    "--server.headless", "true"
])

time.sleep(5)

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('http://localhost:8509')
        page.wait_for_timeout(6000)
        page.screenshot(path='C:/Users/shree/.gemini/antigravity/brain/db09aa0c-0a89-4cb3-a053-eb7f6d21b157/phase4_no_status_overview_screenshot.png')
        browser.close()
    print("Clean Network page screenshot captured successfully")
except Exception as err:
    print(f"Error during screenshot: {err}")
finally:
    proc.terminate()
