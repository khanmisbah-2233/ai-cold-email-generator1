$ErrorActionPreference = "Stop"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

$streamlit = ".\venv\Scripts\streamlit.exe"
if (Test-Path -LiteralPath $streamlit) {
    & $streamlit run app.py --server.headless false --browser.gatherUsageStats false
} else {
    streamlit run app.py --server.headless false --browser.gatherUsageStats false
}
