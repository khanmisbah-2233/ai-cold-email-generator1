@echo off
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

if exist venv\Scripts\streamlit.exe (
    venv\Scripts\streamlit.exe run app.py --server.headless false --browser.gatherUsageStats false
) else (
    streamlit run app.py --server.headless false --browser.gatherUsageStats false
)
