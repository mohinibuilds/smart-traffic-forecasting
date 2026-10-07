"""
Take screenshots of the Streamlit app using Playwright.
Run: python take_screenshots.py
"""
import asyncio
import os
from playwright.async_api import async_playwright

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "assets", "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

async def take_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 900})

        print("Opening Streamlit app...")
        await page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(4000)

        # Screenshot 1: Full dashboard top (header + KPIs)
        await page.screenshot(
            path=os.path.join(SCREENSHOTS_DIR, "01_dashboard_overview.png"),
            full_page=False
        )
        print("Screenshot 1: Dashboard overview saved")

        # Screenshot 2: Scroll down — forecast section
        await page.evaluate("window.scrollTo(0, 600)")
        await page.wait_for_timeout(1500)
        await page.screenshot(
            path=os.path.join(SCREENSHOTS_DIR, "02_forecast_section.png"),
            full_page=False
        )
        print("Screenshot 2: Forecast section saved")

        # Screenshot 3: Scroll down more — charts
        await page.evaluate("window.scrollTo(0, 1400)")
        await page.wait_for_timeout(1500)
        await page.screenshot(
            path=os.path.join(SCREENSHOTS_DIR, "03_charts_section.png"),
            full_page=False
        )
        print("Screenshot 3: Charts section saved")

        # Screenshot 4: Full page screenshot
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(1000)
        await page.screenshot(
            path=os.path.join(SCREENSHOTS_DIR, "04_full_page.png"),
            full_page=True
        )
        print("Screenshot 4: Full page saved")

        await browser.close()
        print(f"\nAll screenshots saved to: {SCREENSHOTS_DIR}")

asyncio.run(take_screenshots())
