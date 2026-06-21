@echo off
REM ==========================================================================
REM  S&P 500 RAG Vault — daily refresh
REM  Run by Windows Task Scheduler. Runs every due layer (sentiment/signals/
REM  backtest daily; quant quarterly; relationships annually), then re-renders
REM  the vault and incrementally re-embeds only the chunks that changed.
REM  Logs to data\scheduler.log.
REM ==========================================================================
cd /d "C:\Users\pcagm\PycharmProjects\PythonProject12"
echo. >> "data\scheduler.log"
echo ======== %DATE% %TIME% ======== >> "data\scheduler.log"
".venv\Scripts\python.exe" -m sp500_vault.scheduler tick >> "data\scheduler.log" 2>&1
echo exit code %ERRORLEVEL% >> "data\scheduler.log"
