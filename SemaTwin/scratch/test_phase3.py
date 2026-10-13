import subprocess
import time
from playwright.sync_api import sync_playwright

proc = subprocess.Popen([
    r"c:\Users\shree\Downloads\Bappa\venv\Scripts\python.exe",
    "-m", "streamlit", "run", "dashboard/pages/2_Image_and_Semantics.py",
    "--server.port", "8505",
    "--server.headless", "true"
])

time.sleep(5)

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('http://localhost:8505')
        page.wait_for_selector('img', timeout=30000)
        page.wait_for_timeout(4000)
        page.screenshot(path='C:/Users/shree/.gemini/antigravity/brain/db09aa0c-0a89-4cb3-a053-eb7f6d21b157/phase3_image_semantics_screenshot.png')
        browser.close()
    print("Page 2 screenshot captured successfully")
except Exception as err:
    print(f"Error during screenshot: {err}")
finally:
    proc.terminate()
