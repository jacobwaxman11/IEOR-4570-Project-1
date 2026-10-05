"""The tools the harness can run, and the JSON that describes them to the model."""

import json
import xml.etree.ElementTree as ET
import requests

# Crypto API URLs
CRYPTO_PRICES_URL = "https://api.coingecko.com/api/v3/simple/price"
CRYPTO_NEWS_URL = "https://cointelegraph.com/rss"
ETHEREUM_TX_URL = "https://eth.blockscout.com/api/v2/transactions/{}"
WEI_PER_ETH = 10**18


def get_crypto_prices(ticker_symbol: str) -> str:
    """Get the current USD price, 24h change, and market cap for a ticker symbol."""
    symbol = ticker_symbol.strip().lower()
    try:
        resp = requests.get(
            CRYPTO_PRICES_URL,
            params={
                "symbols": symbol,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return json.dumps({"error": f"Crypto prices service failed: {e}"})

    if symbol not in data:
        return json.dumps({"error": f"No price found for symbol '{ticker_symbol}'"})

    coin = data[symbol]
    return json.dumps({
        "ticker_symbol": symbol.upper(),
        "price_usd": coin["usd"],
        "change_24h_pct": round(coin.get("usd_24h_change", 0), 2),
        "market_cap_usd": coin.get("usd_market_cap"),
    })


def get_crypto_news(limit: int = 5) -> str:
    """Get the latest crypto headlines."""
    try:
        resp = requests.get(CRYPTO_NEWS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        items = ET.fromstring(resp.content).findall("./channel/item")
    except (requests.RequestException, ET.ParseError) as e:
        return json.dumps({"error": f"Crypto news service failed: {e}"})

    return json.dumps({
        "articles": [
            {
                "title": item.findtext("title"),
                "link": item.findtext("link"),
                "published": item.findtext("pubDate"),
            }
            for item in items[: max(1, min(limit, 10))]
        ]
    })


def get_ethereum_data(tx_hash: str) -> str:
    """Look up an Ethereum transaction by its hash."""
    tx_hash = tx_hash.strip()
    if not (tx_hash.startswith("0x") and len(tx_hash) == 66):
        return json.dumps({"error": "A transaction hash is 0x followed by 64 hex characters."})
    try:
        resp = requests.get(ETHEREUM_TX_URL.format(tx_hash), timeout=10)
        if resp.status_code == 404:
            return json.dumps({"error": f"Transaction {tx_hash} not found on Ethereum mainnet"})
        resp.raise_for_status()
        tx = resp.json()
    except (requests.RequestException, ValueError) as e:
        return json.dumps({"error": f"Ethereum data service failed: {e}"})

    return json.dumps({
        "hash": tx["hash"],
        "status": tx.get("status"),
        "block_number": tx.get("block_number"),
        "timestamp": tx.get("timestamp"),
        "confirmations": tx.get("confirmations"),
        "from": (tx.get("from") or {}).get("hash"),
        "to": (tx.get("to") or {}).get("hash"),
        "value_eth": int(tx.get("value") or 0) / WEI_PER_ETH,
        "fee_eth": int((tx.get("fee") or {}).get("value") or 0) / WEI_PER_ETH,
        "method": tx.get("method"),
    })


# What the model sees: the "set notes" in the screenplay.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_crypto_prices",
            "description": (
                "Get the current USD price, 24-hour percent change, and market cap "
                "for a cryptocurrency by its ticker symbol."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "Crypto ticker symbol, e.g. 'BTC', 'ETH', 'SOL'.",
                    },
                },
                "required": ["ticker_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto_news",
            "description": "Get the latest general crypto news headlines, with links and publish dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of headlines to return, from 1 to 10. Defaults to 5.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ethereum_data",
            "description": (
                "Look up an Ethereum mainnet transaction by its hash. Returns status, block, "
                "timestamp, sender, recipient, value in ETH, fee in ETH, and method."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tx_hash": {
                        "type": "string",
                        "description": "Transaction hash: '0x' followed by 64 hex characters.",
                    },
                },
                "required": ["tx_hash"],
            },
        },
    },
]

# What the harness runs: tool name -> Python function.
TOOL_MAP = {
    "get_crypto_prices": get_crypto_prices,
    "get_crypto_news": get_crypto_news,
    "get_ethereum_data": get_ethereum_data,
}


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}"})
    try:
        return TOOL_MAP[name](**args)
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})
