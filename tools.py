"""The tools the harness can run, and the JSON that describes them to the model."""

import json
import os
import statistics
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

CMC_API_KEY = os.environ.get("CMC_API_KEY", "")
CG_API_KEY = os.environ.get("CG_API_KEY", "")
NEWS_DATA_API_KEY = os.environ.get("NEWS_DATA_API_KEY", "")

CMC_BASE_URL = "https://pro-api.coinmarketcap.com"
CG_BASE_URL = "https://api.coingecko.com/api/v3"
NEWS_BASE_URL = "https://newsdata.io/api/1"
ETHEREUM_TX_URL = "https://eth.blockscout.com/api/v2/transactions/{}"
WEI_PER_ETH = 10**18

CACHE_TTL_SECONDS = 60
MAX_SYMBOLS = 10


class ToolError(Exception):
    """A failure the model should see as {"error": ...} instead of a crash."""


# --- HTTP helpers ---

# (url, sorted params) -> (fetched_at, data). Keeps multi-tool turns inside free-tier rate limits.
_cache: dict[tuple, tuple[float, object]] = {}

# Universal GET request function
def _get(service: str, url: str, params: dict, headers: dict) -> object:
    key = (url, tuple(sorted(params.items())))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        raise ToolError(f"{service} request failed ({e.response.status_code}): {e.response.text[:200]}")
    except (requests.RequestException, ValueError) as e:
        raise ToolError(f"{service} request failed: {e}")
    _cache[key] = (time.time(), data)
    return data

# Framework for getting data from the CoinMarketCap API
def _cmc_get(path: str, params: dict | None = None) -> object:
    if not CMC_API_KEY:
        raise ToolError("CMC_API_KEY is not set in .env")
    body = _get("CoinMarketCap", CMC_BASE_URL + path, params or {}, {"X-CMC_PRO_API_KEY": CMC_API_KEY})
    status = body.get("status", {})
    if str(status.get("error_code", "0")) != "0":
        raise ToolError(f"CoinMarketCap error: {status.get('error_message')}")
    return body["data"]

# Framework for getting data from the CoinGecko API
def _cg_get(path: str, params: dict | None = None) -> object:
    if not CG_API_KEY:
        raise ToolError("CG_API_KEY is not set in .env")
    return _get("CoinGecko", CG_BASE_URL + path, params or {}, {"x-cg-demo-api-key": CG_API_KEY})

# Framework for getting data from the Newsdata.io API
def _news_get(path: str, params: dict) -> list[dict]:
    if not NEWS_DATA_API_KEY:
        raise ToolError("NEWS_DATA_API_KEY is not set in .env")
    body = _get("Newsdata.io", NEWS_BASE_URL + path, {**params, "apikey": NEWS_DATA_API_KEY}, {})
    if body.get("status") != "success":
        raise ToolError(f"Newsdata.io error: {body.get('results') or body}")
    return body.get("results") or []


# --- Formatting helpers ---

# Keeps data compact and readable
def _round(value, digits: int = 2):
    """Round for compact output; keep significant digits on sub-dollar prices."""
    if value is None:
        return None
    value = float(value)
    if 0 < abs(value) < 1:
        return float(f"{value:.4g}")
    return round(value, digits)

# Keeps data clean and readable
def _clean_symbols(symbols) -> list[str]:
    if isinstance(symbols, str):
        symbols = symbols.split(",")
    seen = []
    for s in symbols or []:
        s = str(s).strip().upper()
        if s and s not in seen:
            seen.append(s)
    return seen


def _usd(coin: dict) -> dict:
    """CMC v3 returns `quote` as a list of per-currency dicts."""
    quote = coin.get("quote") or []
    if isinstance(quote, dict):
        return quote.get("USD", {})
    return next((q for q in quote if q.get("symbol") == "USD"), quote[0] if quote else {})


def _coin_summary(coin: dict) -> dict:
    q = _usd(coin)
    return {
        "name": coin.get("name"),
        "symbol": coin.get("symbol"),
        "rank": coin.get("cmc_rank"),
        "price_usd": _round(q.get("price")),
        "change_1h_pct": _round(q.get("percent_change_1h")),
        "change_24h_pct": _round(q.get("percent_change_24h")),
        "change_7d_pct": _round(q.get("percent_change_7d")),
        "change_30d_pct": _round(q.get("percent_change_30d")),
        "volume_24h_usd": _round(q.get("volume_24h"), 0),
        "market_cap_usd": _round(q.get("market_cap"), 0),
    }


def _cmc_quotes(symbols: list[str]) -> tuple[list[dict], list[str]]:
    """Latest quotes for symbols. Many coins share a ticker, so keep the best-ranked one."""
    data = _cmc_get("/v3/cryptocurrency/quotes/latest", {"symbol": ",".join(symbols)})
    best: dict[str, dict] = {}
    for coin in data:
        sym = (coin.get("symbol") or "").upper()
        rank = coin.get("cmc_rank") or float("inf")
        if sym in symbols and (sym not in best or rank < (best[sym].get("cmc_rank") or float("inf"))):
            best[sym] = coin
    found = [_coin_summary(best[s]) for s in symbols if s in best]
    missing = [s for s in symbols if s not in best]
    return found, missing


def _global_metrics() -> dict:
    g = _cmc_get("/v1/global-metrics/quotes/latest")
    q = g["quote"]["USD"]
    return {
        "total_market_cap_usd": _round(q.get("total_market_cap"), 0),
        "market_cap_change_24h_pct": _round(q.get("total_market_cap_yesterday_percentage_change")),
        "total_volume_24h_usd": _round(q.get("total_volume_24h"), 0),
        "btc_dominance_pct": _round(g.get("btc_dominance")),
        "eth_dominance_pct": _round(g.get("eth_dominance")),
        "stablecoin_market_cap_usd": _round(q.get("stablecoin_market_cap"), 0),
    }


def _tool(fn):
    """Turn ToolErrors into the {"error": ...} JSON the model can reason about."""
    def wrapper(*args, **kwargs):
        try:
            return json.dumps(fn(*args, **kwargs))
        except ToolError as e:
            return json.dumps({"error": str(e)})
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# --- Tools ---


@_tool
def get_crypto_prices(ticker_symbols: list[str] | None = None) -> dict:
    """Latest price, rank, % changes, volume, and market cap. No symbols -> top 10."""
    symbols = _clean_symbols(ticker_symbols)[:MAX_SYMBOLS]
    if not symbols:
        coins = _cmc_get("/v3/cryptocurrency/listings/latest", {"limit": 10})
        return {"note": "No symbols given; showing the top 10 by market cap.",
                "coins": [_coin_summary(c) for c in coins]}
    found, missing = _cmc_quotes(symbols)
    result = {"coins": found}
    if missing:
        result["not_found"] = missing
    return result


def _resolve_cg_id(symbol: str) -> tuple[str, str]:
    """Map a ticker to a CoinGecko coin id, preferring the best market-cap rank."""
    coins = _cg_get("/search", {"query": symbol}).get("coins", [])
    matches = [c for c in coins if (c.get("symbol") or "").upper() == symbol.upper()] or coins
    if not matches:
        raise ToolError(f"No CoinGecko coin found for '{symbol}'")
    best = min(matches, key=lambda c: c.get("market_cap_rank") or float("inf"))
    return best["id"], best["name"]


@_tool
def get_price_history(ticker_symbol: str, days: int = 7) -> dict:
    """Daily price history with summary stats over the last N days."""
    days = max(1, min(int(days), 365))
    coin_id, name = _resolve_cg_id(ticker_symbol.strip())
    chart = _cg_get(f"/coins/{coin_id}/market_chart",
                    {"vs_currency": "usd", "days": days, "interval": "daily"})
    prices = chart.get("prices") or []
    if len(prices) < 2:
        raise ToolError(f"Not enough price history for {ticker_symbol}")

    values = [p for _, p in prices]
    returns = [(b - a) / a * 100 for a, b in zip(values, values[1:]) if a]
    step = max(1, len(prices) // 30)
    points = prices[::step]
    if points[-1] is not prices[-1]:
        points.append(prices[-1])

    return {
        "name": name,
        "symbol": ticker_symbol.upper(),
        "days": days,
        "start_price_usd": _round(values[0]),
        "end_price_usd": _round(values[-1]),
        "change_pct": _round((values[-1] - values[0]) / values[0] * 100),
        "high_usd": _round(max(values)),
        "low_usd": _round(min(values)),
        "daily_volatility_pct": _round(statistics.stdev(returns)) if len(returns) > 1 else None,
        "points": [
            {"date": datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d"), "price_usd": _round(p)}
            for ts, p in points
        ],
    }


@_tool
def compare_coins(ticker_symbols: list[str]) -> dict:
    """Side-by-side metrics for 2-10 coins, plus who leads each metric."""
    symbols = _clean_symbols(ticker_symbols)[:MAX_SYMBOLS]
    if len(symbols) < 2:
        raise ToolError("compare_coins needs at least 2 ticker symbols")
    found, missing = _cmc_quotes(symbols)
    if len(found) < 2:
        raise ToolError(f"Could not find enough coins to compare. Not found: {missing}")

    leaders = {}
    for metric in ("change_24h_pct", "change_7d_pct", "market_cap_usd", "volume_24h_usd"):
        ranked = [c for c in found if c[metric] is not None]
        if ranked:
            leaders[metric] = {
                "highest": max(ranked, key=lambda c: c[metric])["symbol"],
                "lowest": min(ranked, key=lambda c: c[metric])["symbol"],
            }
    result = {"coins": found, "leaders": leaders}
    if missing:
        result["not_found"] = missing
    return result


@_tool
def get_market_overview(limit: int = 10) -> dict:
    """Top N coins by market cap plus global market metrics."""
    limit = max(1, min(int(limit), 25))
    coins = _cmc_get("/v3/cryptocurrency/listings/latest", {"limit": limit})
    top = []
    for c in coins:
        s = _coin_summary(c)
        top.append({k: s[k] for k in ("rank", "name", "symbol", "price_usd",
                                      "change_24h_pct", "change_7d_pct", "market_cap_usd")})
    return {"global": _global_metrics(), "top_coins": top}


@_tool
def get_market_indices(history_days: int = 0) -> dict:
    """CMC100 index (value, 24h change, top constituents) and global market cap / dominance."""
    idx = _cmc_get("/v3/index/cmc100-latest")
    result = {
        "cmc100": {
            "value": _round(idx.get("value")),
            "change_24h_pct": _round(idx.get("value_24h_percentage_change")),
            "last_update": idx.get("last_update"),
            "top_constituents": [
                {"symbol": c.get("symbol"), "name": c.get("name"), "weight_pct": _round(c.get("weight"))}
                for c in (idx.get("constituents") or [])[:10]
            ],
        },
        "global": _global_metrics(),
    }
    history_days = max(0, min(int(history_days), 90))
    if history_days:
        hist = _cmc_get("/v3/index/cmc100-historical", {"count": history_days})
        result["cmc100"]["history"] = [
            {"date": (h.get("update_time") or "")[:10], "value": _round(h.get("value"))} for h in hist
        ]
    return result


ALTSEASON_MAX_HISTORY_DAYS = 90


def _altseason_timeframe(days: int) -> str:
    for limit, timeframe in ((7, "7d"), (30, "30d")):
        if days <= limit:
            return timeframe
    return "90d"


@_tool
def get_market_sentiment(history_days: int = 0) -> dict:
    """Fear & Greed Index and Altcoin Season Index, latest and optional daily history."""
    fng = _cmc_get("/v3/fear-and-greed/latest")
    alt = _cmc_get("/v1/altcoin-season-index/latest")
    result = {
        "fear_and_greed": {
            "value": fng.get("value"),
            "classification": fng.get("value_classification"),
            "updated": fng.get("update_time"),
            "scale": "0 = extreme fear, 100 = extreme greed",
        },
        "altcoin_season": {
            "value": alt.get("altcoin_index"),
            "yearly_high": alt.get("yearly_high"),
            "yearly_high_date": alt.get("yearly_high_date"),
            "yearly_low": alt.get("yearly_low"),
            "yearly_low_date": alt.get("yearly_low_date"),
            "scale": "0-25 = Bitcoin season, 75-100 = altcoin season",
        },
    }

    history_days = max(0, min(int(history_days), 365))
    if history_days:
        fng_hist = list(reversed(_cmc_get("/v3/fear-and-greed/historical", {"limit": history_days})))
        result["fear_and_greed"]["history"] = [
            {
                "date": datetime.fromtimestamp(int(h["timestamp"]), timezone.utc).strftime("%Y-%m-%d"),
                "value": h.get("value"),
                "classification": h.get("value_classification"),
            }
            for h in fng_hist[::max(1, len(fng_hist) // 30)]
        ]
        alt_hist = _cmc_get("/v1/altcoin-season-index/historical",
                            {"timeframe": _altseason_timeframe(history_days)})
        points = (alt_hist.get("points") or [])[-history_days:]
        step = max(1, len(points) // 30)
        result["altcoin_season"]["history"] = [
            {"date": (p.get("timestamp") or "")[:10], "value": p.get("altcoin_index")} for p in points[::step]
        ]
        if history_days > ALTSEASON_MAX_HISTORY_DAYS:
            result["altcoin_season"]["history_note"] = (
                f"CoinMarketCap only provides {ALTSEASON_MAX_HISTORY_DAYS} days of Altcoin Season history."
            )
    return result


def _trending() -> dict:
    data = _cg_get("/search/trending")
    coins = []
    for entry in (data.get("coins") or [])[:7]:
        item = entry.get("item", {})
        change = (item.get("data") or {}).get("price_change_percentage_24h") or {}
        coins.append({
            "name": item.get("name"),
            "symbol": item.get("symbol"),
            "market_cap_rank": item.get("market_cap_rank"),
            "change_24h_pct": _round(change.get("usd")),
        })
    categories = [c.get("name") for c in (data.get("categories") or [])[:5]]
    return {"coins": coins, "categories": categories}


@_tool
def get_crypto_news(coins: list[str] | None = None, query: str = "", limit: int = 5) -> dict:
    """Latest crypto headlines (optionally filtered by coin or topic) plus what's trending."""
    limit = max(1, min(int(limit), 10))
    symbols = [s.lower() for s in _clean_symbols(coins)][:5]
    params = {"language": "en"}
    if symbols:
        params["coin"] = ",".join(symbols)
    if query.strip():
        params["q"] = query.strip()
    articles = _news_get("/crypto", params)
    if not articles:
        articles = _news_get("/latest", {"q": query.strip() or "cryptocurrency", "language": "en"})

    result = {
        "articles": [
            {
                "title": a.get("title"),
                "source": a.get("source_name") or a.get("source_id"),
                "published": a.get("pubDate"),
                "link": a.get("link"),
                "description": (a.get("description") or "")[:200],
            }
            for a in articles[:limit]
        ]
    }
    try:
        result["trending"] = _trending()
    except ToolError as e:
        result["trending"] = {"error": str(e)}
    return result


@_tool
def get_ethereum_data(tx_hash: str) -> dict:
    """Look up an Ethereum transaction by its hash."""
    tx_hash = tx_hash.strip()
    if not (tx_hash.startswith("0x") and len(tx_hash) == 66):
        raise ToolError("A transaction hash is 0x followed by 64 hex characters.")
    try:
        resp = requests.get(ETHEREUM_TX_URL.format(tx_hash), timeout=10)
        if resp.status_code == 404:
            raise ToolError(f"Transaction {tx_hash} not found on Ethereum mainnet")
        resp.raise_for_status()
        tx = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise ToolError(f"Ethereum data service failed: {e}")

    return {
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
    }


# What the model sees: the "set notes" in the screenplay.
SYMBOL_LIST = {
    "type": "array",
    "items": {"type": "string"},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_crypto_prices",
            "description": (
                "Get the latest USD price, market-cap rank, percent change (1h, 24h, 7d, 30d), "
                "24h volume, and market cap for one or more cryptocurrencies. "
                "Call with no symbols to get the top 10 coins by market cap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbols": {
                        **SYMBOL_LIST,
                        "description": "Up to 10 ticker symbols, e.g. ['BTC', 'ETH']. Omit for the top 10.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": (
                "Get daily price history for one cryptocurrency over the last N days, with start/end "
                "price, high, low, percent change, and daily volatility."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {"type": "string", "description": "Ticker symbol, e.g. 'BTC'."},
                    "days": {"type": "integer", "description": "Lookback window in days, 1 to 365. Defaults to 7."},
                },
                "required": ["ticker_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_coins",
            "description": (
                "Compare 2 to 10 cryptocurrencies side by side (price, rank, % changes, volume, market cap) "
                "and report which coin leads and lags on 24h change, 7d change, market cap, and volume."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbols": {
                        **SYMBOL_LIST,
                        "description": "2 to 10 ticker symbols, e.g. ['BTC', 'ETH', 'SOL'].",
                    },
                },
                "required": ["ticker_symbols"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_overview",
            "description": (
                "Get a crypto market overview: the top coins by market cap with price and 24h/7d change, "
                "plus total market cap, 24h volume, and BTC/ETH dominance. Use when no specific coin is named."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of top coins, 1 to 25. Defaults to 10."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_indices",
            "description": (
                "Get the CoinMarketCap CMC100 index (value, 24h change, top constituents by weight) and "
                "total crypto market cap and dominance. Optionally include daily CMC100 history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "history_days": {
                        "type": "integer",
                        "description": "Days of daily CMC100 history to include, 0 to 90. Defaults to 0 (latest only).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": (
                "Get market sentiment from CoinMarketCap: the Fear & Greed Index and the Altcoin Season Index, "
                "latest values and optionally daily history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "history_days": {
                        "type": "integer",
                        "description": (
                            "Days of history to include, 0 to 365 (Altcoin Season history stops at 90). "
                            "Defaults to 0 (latest only)."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto_news",
            "description": (
                "Get the latest crypto news headlines, optionally filtered by coin or topic, plus the coins "
                "and categories currently trending on CoinGecko."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coins": {
                        **SYMBOL_LIST,
                        "description": "Optional ticker symbols to filter news by, e.g. ['BTC'].",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional topic keywords when not filtering by coin, e.g. 'ETF' or 'regulation'.",
                    },
                    "limit": {"type": "integer", "description": "Number of headlines, 1 to 10. Defaults to 5."},
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
    "get_price_history": get_price_history,
    "compare_coins": compare_coins,
    "get_market_overview": get_market_overview,
    "get_market_indices": get_market_indices,
    "get_market_sentiment": get_market_sentiment,
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
    except Exception as e:
        return json.dumps({"error": f"{name} failed unexpectedly: {type(e).__name__}: {e}"})
