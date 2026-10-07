"""The tools the harness can run, and the JSON that describes them to the model."""

import functools
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
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
    """Turn ToolErrors into the {"error": ...} JSON the model can reason about.

    The raw dict-returning function stays reachable as `tool.__wrapped__` for composite tools.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return json.dumps(fn(*args, **kwargs))
        except ToolError as e:
            return json.dumps({"error": str(e)})
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


def _price_series(symbol: str, days: int) -> tuple[str, list[list[float]]]:
    """Daily [timestamp_ms, price] pairs from CoinGecko, plus the coin's display name."""
    coin_id, name = _resolve_cg_id(symbol.strip())
    chart = _cg_get(f"/coins/{coin_id}/market_chart",
                    {"vs_currency": "usd", "days": days, "interval": "daily"})
    return name, chart.get("prices") or []


@_tool
def get_price_history(ticker_symbol: str, days: int = 7) -> dict:
    """Daily price history with summary stats over the last N days."""
    days = max(1, min(int(days), 365))
    name, prices = _price_series(ticker_symbol, days)
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


# --- Research: stage 1 screens the top N coins, stage 2 deep-dives a shortlist ---

# Shortlist slots per market-cap tier, so smaller coins always get considered.
SHORTLIST_TIERS = (("large cap", 2, 10, 2), ("mid cap", 11, 50, 3), ("small cap", 51, 250, 2))
# Exact CMC tags only: substrings like "real-world" also match infrastructure projects (LINK, AVAX).
EXCLUDED_TAGS = {"stablecoin", "tokenized-gold", "tokenized-assets", "tokenized-commodities",
                 "tokenized-stock", "wrapped-tokens", "fan-token"}
MIN_VOLUME_TO_MCAP = 0.005


def _pct(start, end):
    return _round((end - start) / start * 100) if start else None


def _max_drawdown_pct(values: list[float]) -> float:
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        worst = min(worst, (v - peak) / peak)
    return _round(worst * 100)


def _percentiles(values: list) -> list[float]:
    """0-1 rank of each value within the list; missing values score 0.5."""
    present = sorted(v for v in values if v is not None)
    if len(present) < 2:
        return [0.5] * len(values)
    return [0.5 if v is None else present.index(v) / (len(present) - 1) for v in values]


def _tier(rank) -> str:
    if rank and rank <= 10:
        return "large cap"
    return "mid cap" if rank and rank <= 50 else "small cap"


def _screen_universe(size: int) -> dict:
    """Score every investable coin in the top N on momentum vs BTC, dilution, and liquidity."""
    coins = _cg_get("/coins/markets", {
        "vs_currency": "usd", "per_page": size, "page": 1,
        "price_change_percentage": "7d,30d,200d,1y",
    })
    tags = {c["symbol"].upper(): c.get("tags") or [] for c in _cmc_get(
        "/v3/cryptocurrency/listings/latest", {"limit": min(size + 20, 200)})}
    btc = next((c for c in coins if c["id"] == "bitcoin"), None)
    if not btc:
        raise ToolError("Bitcoin missing from CoinGecko markets; cannot benchmark")

    def change(c, period):
        return c.get(f"price_change_percentage_{period}_in_currency")

    universe, excluded = [], {}
    for c in coins:
        sym, mcap = c["symbol"].upper(), c.get("market_cap") or 0
        if EXCLUDED_TAGS & set(tags.get(sym, [])):
            excluded[sym] = "stablecoin / tokenized / wrapped asset"
        elif abs(change(c, "200d") or 0) < 2 and abs(change(c, "1y") or 0) < 2:
            excluded[sym] = "pegged price"
        elif not mcap or (c.get("total_volume") or 0) / mcap < MIN_VOLUME_TO_MCAP:
            excluded[sym] = "illiquid"
        else:
            universe.append(c)

    factors = {
        "rs_30d": [None if change(c, "30d") is None else change(c, "30d") - change(btc, "30d") for c in universe],
        "rs_200d": [None if change(c, "200d") is None else change(c, "200d") - change(btc, "200d") for c in universe],
        "rs_1y": [None if change(c, "1y") is None else change(c, "1y") - change(btc, "1y") for c in universe],
        "circulating_share": [c["market_cap"] / c["fully_diluted_valuation"] if c.get("fully_diluted_valuation") else None
                              for c in universe],
        "liquidity": [c["total_volume"] / c["market_cap"] for c in universe],
    }
    ranked = list(zip(*(_percentiles(v) for v in factors.values())))
    for c, pcts, *raw in zip(universe, ranked, *factors.values()):
        c["_score"] = round(sum(pcts) / len(pcts) * 100)
        c["_factors"] = dict(zip(factors, raw))
    universe.sort(key=lambda c: c["_score"], reverse=True)
    return {"universe": universe, "excluded": excluded, "btc": btc}


def _shortlist(universe: list[dict], btc: dict, forced: list[str]) -> list[dict]:
    picks = [btc]
    for symbol in forced:
        match = next((c for c in universe if c["symbol"].upper() == symbol), None)
        if match and match not in picks:
            picks.append(match)
    for _, low, high, slots in SHORTLIST_TIERS:
        tier = [c for c in universe if low <= (c.get("market_cap_rank") or 999) <= high and c not in picks]
        picks += tier[:slots]
    return picks


def _labels(card: dict, coin: dict) -> None:
    """Plain-language reads of the numbers, so the model doesn't have to do the math."""
    f = coin.get("_factors", {})
    share = f.get("circulating_share")
    liquidity = f.get("liquidity") or 0
    card["dilution"] = ("unknown" if share is None else "low" if share >= 0.9
                        else "moderate" if share >= 0.6 else f"high ({round((1 - share) * 100)}% of supply not circulating)")
    card["liquidity"] = "high" if liquidity >= 0.05 else "medium" if liquidity >= 0.01 else "low"
    flags = []
    if (f.get("rs_200d") or 0) > 0:
        flags.append("outperforming BTC over 200d")
    if (f.get("rs_1y") or 0) > 0:
        flags.append("outperforming BTC over 1y")
    if (card.get("max_drawdown_pct") or 0) < -70:
        flags.append("deep drawdown history")
    if (card.get("from_ath_pct") or 0) < -80:
        flags.append("more than 80% below all-time high")
    if share is not None and share < 0.6:
        flags.append("significant future token unlocks")
    card["flags"] = flags


def _deep_dive(coin: dict, btc_returns: list[float] | None) -> dict:
    """One-year chart stats for a shortlisted coin, labelled for easy reading."""
    chart = _cg_get(f"/coins/{coin['id']}/market_chart", {"vs_currency": "usd", "days": 365, "interval": "daily"})
    values = [p for _, p in chart.get("prices") or []]
    if len(values) < 200:
        raise ToolError(f"Not enough price history for {coin['symbol'].upper()}")
    returns = [(b - a) / a for a, b in zip(values, values[1:]) if a]
    annual_vol = statistics.stdev(returns) * (365 ** 0.5) * 100
    one_year = _pct(values[0], values[-1])
    ma50, ma200 = statistics.fmean(values[-50:]), statistics.fmean(values[-200:])
    price = values[-1]
    trend = ("uptrend (above 50d and 200d average)" if price > ma50 and price > ma200
             else "downtrend (below 50d and 200d average)" if price < ma50 and price < ma200 else "mixed")

    correlation = None
    if btc_returns and coin["id"] != "bitcoin":
        n = min(len(returns), len(btc_returns))
        correlation = _round(statistics.correlation(returns[-n:], btc_returns[-n:]))

    rank = coin.get("market_cap_rank")
    card = {
        "symbol": coin["symbol"].upper(),
        "name": coin["name"],
        "tier": f"{_tier(rank)} (rank {rank})",
        "screen_score": coin.get("_score"),
        "price_usd": _round(price),
        "returns_pct": {f"{d}d": _pct(values[-d - 1], price) for d in (7, 30, 90, 180)} | {"365d": one_year},
        "vs_btc_pts": {k: _round(coin["_factors"][f"rs_{k}"]) for k in ("30d", "200d", "1y")}
                      if coin.get("_factors") else None,
        "trend": trend,
        "annualized_volatility_pct": _round(annual_vol),
        "return_to_risk": _round(one_year / annual_vol) if annual_vol and one_year is not None else None,
        "max_drawdown_pct": _max_drawdown_pct(values),
        "from_365d_high_pct": _pct(max(values), price),
        "from_ath_pct": _round(coin.get("ath_change_percentage")),
        "correlation_to_btc": correlation,
    }
    _labels(card, coin)
    return card


def _research_news(coins: list[str], limit: int) -> dict:
    """Newsdata rejects the whole request if any coin filter is unknown, so fall back to unfiltered news."""
    try:
        return get_crypto_news.__wrapped__(coins, "", limit)
    except ToolError:
        return get_crypto_news.__wrapped__(None, "", limit)


# Composite tools list the tools they ran under this key. The harness strips it before the
# result reaches the model and forwards it to the UI instead.
SUB_CALLS_KEY = "_sub_calls"


def _sub_call(name: str, fn, **args) -> tuple[dict, dict]:
    """Run a tool's raw function and return (UI record, result). Failures become {"error": ...}."""
    try:
        result = fn(**args)
    except ToolError as e:
        result = {"error": str(e)}
    return {"name": name, "args": args, "result": json.dumps(result)}, result


@_tool
def research_crypto(candidates: list[str] | None = None, universe_size: int = 100) -> dict:
    """Screen the top N coins, deep-dive a tiered shortlist, and add market context and news."""
    universe_size = max(50, min(int(universe_size), 250))
    forced = _clean_symbols(candidates)[:3]

    screen_record, screen = _sub_call("screen_market", _screen_universe, size=universe_size)
    if "error" in screen:
        raise ToolError(screen["error"])
    universe, btc = screen["universe"], screen["btc"]
    shortlist = _shortlist(universe, btc, forced)
    screen_record["result"] = json.dumps({
        "screened": universe_size,
        "excluded": screen["excluded"],
        "top_scores": [{"symbol": c["symbol"].upper(), "rank": c.get("market_cap_rank"), "score": c["_score"]}
                       for c in universe[:15]],
        "shortlist": [c["symbol"].upper() for c in shortlist],
    })

    symbols = [c["symbol"].upper() for c in shortlist]
    news_batches = [symbols[i:i + 4] for i in range(0, len(symbols), 4)][:2]
    with ThreadPoolExecutor(max_workers=8) as pool:
        # BTC first: its daily returns are the correlation benchmark for everyone else.
        btc_record, btc_card = _sub_call("analyze_price_history", _deep_dive, coin=btc, btc_returns=None)
        btc_values = [p for _, p in _cg_get("/coins/bitcoin/market_chart",
                                            {"vs_currency": "usd", "days": 365, "interval": "daily"})["prices"]]
        btc_returns = [(b - a) / a for a, b in zip(btc_values, btc_values[1:]) if a]
        card_futures = [pool.submit(_sub_call, "analyze_price_history", _deep_dive, coin=c, btc_returns=btc_returns)
                        for c in shortlist[1:]]
        sentiment_f = pool.submit(_sub_call, "get_market_sentiment", get_market_sentiment.__wrapped__, history_days=30)
        indices_f = pool.submit(_sub_call, "get_market_indices", get_market_indices.__wrapped__, history_days=0)
        news_fs = [pool.submit(_sub_call, "get_crypto_news", _research_news, coins=batch, limit=10)
                   for batch in news_batches]
        card_results = [(btc_record, btc_card)] + [f.result() for f in card_futures]
        (sentiment_record, sentiment), (indices_record, indices) = sentiment_f.result(), indices_f.result()
        news_results = [f.result() for f in news_fs]

    for name, data in (("sentiment", sentiment), ("indices", indices)):
        if "error" in data:
            raise ToolError(f"Research {name} step failed: {data['error']}")

    for (record, card), coin in zip(card_results, shortlist):
        record["args"] = {"symbol": coin["symbol"].upper(), "days": 365}
    articles = [a for _, news in news_results for a in news.get("articles", [])]
    cards = []
    for _, card in card_results:
        if "error" not in card:
            name = card["name"].lower()
            card["headlines"] = [a["title"] for a in articles if a.get("title") and
                                 (card["symbol"] in a["title"] or name in a["title"].lower())][:3]
        cards.append(card)

    fng, alt = sentiment["fear_and_greed"], sentiment["altcoin_season"]
    sub_calls = [screen_record, *(r for r, _ in card_results), sentiment_record, indices_record,
                 *(r for r, _ in news_results)]
    return {
        "method": (
            f"Screened the top {universe_size} coins by market cap. Excluded {len(screen['excluded'])} "
            "stablecoins, tokenized/wrapped assets, pegged or illiquid coins. Scored the rest 0-100 on "
            "performance vs BTC (30d, 200d, 1y), share of supply circulating, and liquidity, then shortlisted "
            "BTC as the benchmark plus the best scorers in each tier (large, mid, small cap). Price stats "
            "cover 365 days, the free-plan maximum, so multi-year views extrapolate from that window."
        ),
        "market_context": {
            "global": indices["global"],
            "cmc100_change_24h_pct": indices["cmc100"]["change_24h_pct"],
            "fear_and_greed": {"value": fng["value"], "classification": fng["classification"],
                               "value_30d_ago": (fng.get("history") or [{}])[0].get("value")},
            "altcoin_season": {k: alt.get(k) for k in ("value", "yearly_high", "yearly_low")},
        },
        "screen_top_10": [
            {"symbol": c["symbol"].upper(), "tier": _tier(c.get("market_cap_rank")), "score": c["_score"]}
            for c in universe[:10]
        ],
        "shortlist": cards,
        "trending": next((n.get("trending") for _, n in news_results if n.get("trending")), None),
        "latest_headlines": [{"title": a["title"], "source": a["source"]} for a in articles[:5]],
        SUB_CALLS_KEY: sub_calls,
    }


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
            "name": "research_crypto",
            "description": (
                "Deep research for open-ended or long-term questions (e.g. 'which coin would you hold for 3 years', "
                "'what's the strongest asset right now'). Screens the top N coins on performance vs BTC, dilution, "
                "and liquidity, then deep-dives a shortlist spanning large, mid, and small caps (with BTC as the "
                "benchmark): multi-period returns, trend, volatility, drawdown, correlation to BTC, plain-language "
                "flags, and related headlines. Also returns market sentiment, dominance, and trending coins."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {
                        **SYMBOL_LIST,
                        "description": "Optional: up to 3 tickers to force onto the shortlist, e.g. ['LINK'].",
                    },
                    "universe_size": {
                        "type": "integer",
                        "description": "How many top coins by market cap to screen, 50 to 250. Defaults to 100.",
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
    "get_price_history": get_price_history,
    "compare_coins": compare_coins,
    "get_market_overview": get_market_overview,
    "get_market_indices": get_market_indices,
    "get_market_sentiment": get_market_sentiment,
    "get_crypto_news": get_crypto_news,
    "research_crypto": research_crypto,
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
