"""The pilot ticker universe, organized by sector cluster.

Per the project plan we scale "sector by sector". The universe is therefore a
dict of named clusters; the pipeline can run a single cluster (``--sector``) or
the whole set. We started with semiconductors + hardware and added cloud/software
megacaps — the hyperscalers (MSFT, GOOGL, AMZN, META, ORCL) are the GPU buyers
that the semiconductor 10-Ks reference, so the second cluster densifies the
existing graph rather than forming an island.

Sector/industry/market-cap come from live data at runtime; the ``group`` field
here is the pilot cluster label (also emitted as an Obsidian tag).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    group: str  # "semiconductor" | "hardware" | "cloud_software"


SECTORS: dict[str, list[Company]] = {
    "semiconductor": [
        Company("NVDA", "NVIDIA Corporation", "semiconductor"),
        Company("AMD", "Advanced Micro Devices, Inc.", "semiconductor"),
        Company("INTC", "Intel Corporation", "semiconductor"),
        Company("AVGO", "Broadcom Inc.", "semiconductor"),
        Company("QCOM", "QUALCOMM Incorporated", "semiconductor"),
        Company("TXN", "Texas Instruments Incorporated", "semiconductor"),
        Company("MU", "Micron Technology, Inc.", "semiconductor"),
        Company("ADI", "Analog Devices, Inc.", "semiconductor"),
        Company("AMAT", "Applied Materials, Inc.", "semiconductor"),
        Company("LRCX", "Lam Research Corporation", "semiconductor"),
        Company("KLAC", "KLA Corporation", "semiconductor"),
        Company("MCHP", "Microchip Technology Incorporated", "semiconductor"),
        Company("NXPI", "NXP Semiconductors N.V.", "semiconductor"),
        Company("ON", "ON Semiconductor Corporation", "semiconductor"),
        Company("MPWR", "Monolithic Power Systems, Inc.", "semiconductor"),
        Company("SWKS", "Skyworks Solutions, Inc.", "semiconductor"),
        Company("QRVO", "Qorvo, Inc.", "semiconductor"),
        Company("TER", "Teradyne, Inc.", "semiconductor"),
        Company("GFS", "GlobalFoundries Inc.", "semiconductor"),
    ],
    "hardware": [
        Company("AAPL", "Apple Inc.", "hardware"),
        Company("DELL", "Dell Technologies Inc.", "hardware"),
        Company("HPQ", "HP Inc.", "hardware"),
        Company("HPE", "Hewlett Packard Enterprise Company", "hardware"),
        Company("CSCO", "Cisco Systems, Inc.", "hardware"),
        Company("ANET", "Arista Networks, Inc.", "hardware"),
        Company("SMCI", "Super Micro Computer, Inc.", "hardware"),
        Company("WDC", "Western Digital Corporation", "hardware"),
        Company("STX", "Seagate Technology Holdings plc", "hardware"),
    ],
    "cloud_software": [
        Company("MSFT", "Microsoft Corporation", "cloud_software"),
        Company("GOOGL", "Alphabet Inc.", "cloud_software"),
        Company("AMZN", "Amazon.com, Inc.", "cloud_software"),
        Company("META", "Meta Platforms, Inc.", "cloud_software"),
        Company("ORCL", "Oracle Corporation", "cloud_software"),
        Company("IBM", "International Business Machines Corporation", "cloud_software"),
        Company("CRM", "Salesforce, Inc.", "cloud_software"),
        Company("NOW", "ServiceNow, Inc.", "cloud_software"),
        Company("ADBE", "Adobe Inc.", "cloud_software"),
        Company("PLTR", "Palantir Technologies Inc.", "cloud_software"),
    ],
    # Automakers & suppliers — major buyers of analog/power/auto semiconductors.
    "automotive": [
        Company("TSLA", "Tesla, Inc.", "automotive"),
        Company("GM", "General Motors Company", "automotive"),
        Company("F", "Ford Motor Company", "automotive"),
        Company("APTV", "Aptiv PLC", "automotive"),
        Company("BWA", "BorgWarner Inc.", "automotive"),
    ],
    # Telecom / media — buyers of RF/5G silicon, networking gear, and cloud.
    "communications": [
        Company("VZ", "Verizon Communications Inc.", "communications"),
        Company("T", "AT&T Inc.", "communications"),
        Company("TMUS", "T-Mobile US, Inc.", "communications"),
        Company("CMCSA", "Comcast Corporation", "communications"),
        Company("CHTR", "Charter Communications, Inc.", "communications"),
        Company("NFLX", "Netflix, Inc.", "communications"),
        Company("DIS", "The Walt Disney Company", "communications"),
    ],
}

PILOT_UNIVERSE: list[Company] = [c for group in SECTORS.values() for c in group]

TICKERS: list[str] = [c.ticker for c in PILOT_UNIVERSE]
BY_TICKER: dict[str, Company] = {c.ticker: c for c in PILOT_UNIVERSE}
NAME_TO_TICKER: dict[str, str] = {c.name.upper(): c.ticker for c in PILOT_UNIVERSE}


def tickers_for_group(group: str) -> list[str]:
    return [c.ticker for c in SECTORS.get(group, [])]


def resolve_ticker(value: str | None) -> str | None:
    """Best-effort map a name or ticker string onto a universe ticker."""
    if not value:
        return None
    v = value.strip().upper()
    if v in BY_TICKER:
        return v
    if v in NAME_TO_TICKER:
        return NAME_TO_TICKER[v]
    # Loose name containment (e.g. "NVIDIA" -> "NVDA", "ALPHABET" -> "GOOGL").
    for name, tic in NAME_TO_TICKER.items():
        core = (
            name.split(",")[0]
            .replace(" INC.", "")
            .replace(" CORPORATION", "")
            .replace(" CORP.", "")
            .replace(".COM", "")
            .strip()
        )
        if core and (core in v or v in core):
            return tic
    return None
