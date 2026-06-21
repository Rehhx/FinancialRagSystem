@echo off
REM ==========================================================================
REM  S&P 500 RAG Vault — real-time 8-K poller (intraday fast path)
REM  Polls SEC EDGAR's getcurrent feed every 2 minutes; when a universe company
REM  files an 8-K, it refreshes that ticker's filings and re-embeds its Material
REM  Events chunk — so the vault reacts within minutes of the filing hitting the
REM  wire. Free (no API key, no quota). Logs to data\edgar_live.log.
REM
REM  Run as a Startup task / always-on process. To poll on a schedule instead,
REM  point Task Scheduler at:  sp500_vault.edgar_live tick  (every 2-5 min).
REM ==========================================================================
cd /d "C:\Users\pcagm\PycharmProjects\PythonProject12"
echo. >> "data\edgar_live.log"
echo ======== %DATE% %TIME%  edgar_live watch started ======== >> "data\edgar_live.log"
".venv\Scripts\python.exe" -m sp500_vault.edgar_live watch --interval 120 >> "data\edgar_live.log" 2>&1
