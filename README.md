# Crypto Chatbot Agent

A chat assistant that answers crypto questions with live market data. A Gemini model (via LiteLLM) decides which tools to call, a FastAPI harness runs them, and the web UI shows the answer along with every tool call it made.

## Quick start

1. Copy `.env.example` to `.env` and add free API keys from [CoinGecko](https://www.coingecko.com/en/api), [CoinMarketCap](https://coinmarketcap.com/api/), and [Newsdata.io](https://newsdata.io/).
2. Authenticate with Google Cloud for Vertex AI: `gcloud auth application-default login`.
3. Run `uv run app.py` and open http://127.0.0.1:8000.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Agent loop, system prompt, and FastAPI server |
| `tools.py` | Tool functions, their JSON schemas (`TOOLS`), and the dispatcher (`run_tool`) |
| `index.html` | Chat UI: Markdown replies and an expandable list of tool calls |

## Tools

| Tool | What it does | Source |
|---|---|---|
| `get_crypto_prices` | Latest price, rank, 1h/24h/7d/30d change, volume, and market cap. Returns the top 10 if no coin is named. | CoinMarketCap |
| `get_price_history` | Daily prices over 1–365 days, with high, low, % change, and volatility. | CoinGecko |
| `compare_coins` | 2–10 coins side by side, plus which coin leads each metric. | CoinMarketCap |
| `get_market_overview` | Top coins plus total market cap, volume, and BTC/ETH dominance. | CoinMarketCap |
| `get_market_indices` | CMC100 index value, top constituents, and optional history. | CoinMarketCap |
| `get_market_sentiment` | Fear & Greed Index and Altcoin Season Index, latest and historical. | CoinMarketCap |
| `get_crypto_news` | Latest headlines by coin or topic, plus trending coins and categories. | Newsdata.io, CoinGecko |
| `research_crypto` | Deep research: screens the top 100 coins against BTC, then analyzes a shortlist of large, mid, and small caps (returns, trend, drawdown, dilution, news). | All of the above |
| `get_ethereum_data` | Looks up an Ethereum transaction by hash. | Blockscout |

## Example queries

- "What's the price of BTC, ETH, and SOL?"
- "How has HYPE moved over the last 90 days?"
- "Compare SOL, AVAX, and NEAR."
- "How's the market doing today?"
- "Is the market fearful or greedy right now? How has that changed this month?"
- "Any news on Ethereum ETFs?"
- "If you could only hold one crypto for the next 3 years, what would it be?"
- "Look up this transaction: 0x…"

## Limitations

- **History:** free plans limit price history to the last 365 days.
- **Rate limits:** about 30 requests/minute for CoinGecko and CoinMarketCap, and 200 news requests/day. Responses are cached for 60 seconds.
- **Not financial advice:** answers are analyses of current data.
