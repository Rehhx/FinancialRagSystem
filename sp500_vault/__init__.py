"""S&P 500 Financial RAG Vault.

An incremental pipeline that turns market data, SEC filings, and news into an
Obsidian vault of wikilinked company notes, indexed for natural-language RAG.

Layers (each independently runnable via ``pipeline.py``):
    1. quant         - fundamentals -> per-ticker metrics + sector percentiles
    2. relationships - EDGAR + Claude extraction -> supplier/customer/competitor edges
    3. sentiment     - news + Claude scoring -> per-node sentiment
    4. vault         - render everything into markdown notes with [[wikilinks]]
    5. index         - chunk + embed notes into a vector store
    6. query         - retrieve + answer with Claude
"""

__version__ = "0.1.0"
