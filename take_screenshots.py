from playwright.sync_api import sync_playwright
import os, time

os.makedirs('screenshots', exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1280, 'height': 800})

    # 1. Login page
    page.goto('http://localhost:8503', wait_until='networkidle', timeout=15000)
    time.sleep(2)
    page.screenshot(path='screenshots/01_login_page.png', full_page=False)
    print('Screenshot 1: Login page')

    # 2. Create Account tab
    page.locator('text=Create Account').first.click()
    time.sleep(1)
    page.screenshot(path='screenshots/02_create_account.png', full_page=False)
    print('Screenshot 2: Create Account')

    # 3. Sign in
    page.locator('text=Sign In').first.click()
    time.sleep(0.5)
    page.get_by_placeholder('Enter your username').fill('admin')
    page.get_by_placeholder('Enter your password').fill('traffic123')
    page.screenshot(path='screenshots/03_signin_filled.png', full_page=False)
    print('Screenshot 3: Sign In filled')

    page.get_by_role('button', name='Sign In').click()
    page.wait_for_load_state('networkidle', timeout=20000)
    time.sleep(4)

    # 4. Dashboard top
    page.screenshot(path='screenshots/04_dashboard_top.png', full_page=False)
    print('Screenshot 4: Dashboard top')

    # 5. Forecast chart
    page.evaluate('window.scrollTo(0, 600)')
    time.sleep(1)
    page.screenshot(path='screenshots/05_forecast_chart.png', full_page=False)
    print('Screenshot 5: Forecast chart')

    # 6. Historical trend
    page.evaluate('window.scrollTo(0, 1200)')
    time.sleep(1)
    page.screenshot(path='screenshots/06_historical_trend.png', full_page=False)
    print('Screenshot 6: Historical trend')

    # 7. Heatmap section
    page.evaluate('window.scrollTo(0, 2000)')
    time.sleep(1)
    page.screenshot(path='screenshots/07_heatmap.png', full_page=False)
    print('Screenshot 7: Heatmap')

    browser.close()
    print('All screenshots saved to screenshots/')
