"""Entry point — delegates to the pipeline CLI.

Equivalent to ``python -m sp500_vault.pipeline``. Examples:
    python main.py all
    python main.py all --tickers NVDA,AMD,AAPL
    python main.py query "Which companies are most exposed to NVIDIA guidance cuts?"
"""
from sp500_vault.pipeline import main

if __name__ == "__main__":
    main()
