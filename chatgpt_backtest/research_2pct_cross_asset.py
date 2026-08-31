#!/usr/bin/env python3
"""Backtest the minute-by-minute separate-lot take-profit strategy.

Experiments:
1. BTC/USD on Bitstamp, 2020-01-07 through 2025-01-07, 2% net take profit.
2. U.S. stocks and index-tracking ETFs over 2025-08-29 through 2026-08-28,
   comparing 1% and 2% net take-profit targets.

Model:
- $300 initial capital per asset/strategy.
- Each eligible minute, after processing exits, invest $1 if cash is available.
- Each purchase is a separate fractional-share lot.
- Never force-sell; an open lot exits only if a later candle high reaches its
  individual target.
- $1 is an all-in cash allocation including 0.10% buy fee and 0.01% adverse
  buy slippage.
- A sale pays 0.10% fee and 0.01% adverse sell slippage.
- A target is defined as net profit after all modeled costs.
- Triggered take-profit orders fill at the target reference price, not at a
  better gap price. A lot cannot exit during its purchase candle.
- U.S. equities use split-adjusted (not dividend-adjusted) bars, regular trading
  hours only. Dividends, taxes, spread beyond slippage, market impact, borrow,
  and broker minimum-order restrictions are excluded.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

INITIAL_CAPITAL = 300.0
BUY_ALLOCATION = 1.0
FEE = 0.001
SLIPPAGE = 0.0001
EXIT_VALUE_FACTOR = (
    BUY_ALLOCATION
    * (1.0 - FEE)
    * (1.0 - SLIPPAGE)
    / ((1.0 + FEE) * (1.0 + SLIPPAGE))
)

BTC_START_TS = int(datetime(2020, 1, 7, tzinfo=timezone.utc).timestamp())
BTC_END_TS = int(datetime(2025, 1, 7, tzinfo=timezone.utc).timestamp())
EQUITY_START_DATE = "2025-08-29"
EQUITY_END_DATE_EXCLUSIVE = "2026-08-29"
NY = ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END = time(16, 0)


@dataclass(frozen=True)
class Bar:
    ts: int
    high: float
    close: float
    date_label: str


class Strategy:
    def __init__(self, target_net_rate: float) -> None:
        self.target_net_rate = target_net_rate
        self.sale_proceeds = BUY_ALLOCATION * (1.0 + target_net_rate)
        self.target_factor = (
            (1.0 + target_net_rate)
            * (1.0 + FEE)
            * (1.0 + SLIPPAGE)
            / ((1.0 - FEE) * (1.0 - SLIPPAGE))
        )

        self.cash = INITIAL_CAPITAL
        # Heap item: (target reference price, serial, entry close, entry ts).
        self.open_lots: list[tuple[float, int, float, int]] = []
        self.serial = 0
        self.sum_inverse_entry = 0.0

        self.buys = 0
        self.sales = 0
        self.skipped_buys = 0
        self.max_open_lots = 0
        self.realized_profit = 0.0

        self.hold_minutes_sum = 0
        self.max_hold_minutes = 0
        self.closed_within_60m = 0
        self.closed_within_1d = 0
        self.closed_within_7d = 0
        self.closed_within_30d = 0

        self.peak_equity = INITIAL_CAPITAL
        self.max_drawdown = 0.0
        self.max_drawdown_ts: int | None = None

    def liquidation_value(self, reference_price: float) -> float:
        open_value = reference_price * EXIT_VALUE_FACTOR * self.sum_inverse_entry
        return self.cash + open_value

    def process(self, bar: Bar) -> None:
        # Only lots from earlier bars are present here.
        while self.open_lots and self.open_lots[0][0] <= bar.high:
            _, _, entry_price, entry_ts = heapq.heappop(self.open_lots)
            self.sum_inverse_entry -= 1.0 / entry_price
            self.cash += self.sale_proceeds
            self.sales += 1
            self.realized_profit += BUY_ALLOCATION * self.target_net_rate

            held = max(1, (bar.ts - entry_ts) // 60)
            self.hold_minutes_sum += held
            self.max_hold_minutes = max(self.max_hold_minutes, held)
            if held <= 60:
                self.closed_within_60m += 1
            if held <= 24 * 60:
                self.closed_within_1d += 1
            if held <= 7 * 24 * 60:
                self.closed_within_7d += 1
            if held <= 30 * 24 * 60:
                self.closed_within_30d += 1

        if self.cash + 1e-10 >= BUY_ALLOCATION:
            self.cash -= BUY_ALLOCATION
            if abs(self.cash) < 1e-12:
                self.cash = 0.0
            self.serial += 1
            heapq.heappush(
                self.open_lots,
                (bar.close * self.target_factor, self.serial, bar.close, bar.ts),
            )
            self.sum_inverse_entry += 1.0 / bar.close
            self.buys += 1
            self.max_open_lots = max(self.max_open_lots, len(self.open_lots))
        else:
            self.skipped_buys += 1

        equity = self.liquidation_value(bar.close)
        if equity > self.peak_equity:
            self.peak_equity = equity
        elif self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                self.max_drawdown_ts = bar.ts

    def result(self, final_ts: int, final_price: float, eligible_minutes: int) -> dict:
        open_value = final_price * EXIT_VALUE_FACTOR * self.sum_inverse_entry
        open_cost = float(len(self.open_lots)) * BUY_ALLOCATION
        account_value = self.cash + open_value
        oldest_open_age = 0
        if self.open_lots:
            oldest_entry_ts = min(item[3] for item in self.open_lots)
            oldest_open_age = max(0, (final_ts - oldest_entry_ts) // 60)

        sales = self.sales
        return {
            "target_net_profit_percent": self.target_net_rate * 100.0,
            "required_reference_price_rise_percent": (self.target_factor - 1.0) * 100.0,
            "initial_capital": INITIAL_CAPITAL,
            "final_cash": self.cash,
            "open_lots": len(self.open_lots),
            "open_lot_cost_basis": open_cost,
            "open_liquidation_value": open_value,
            "unrealized_pnl_on_open_lots": open_value - open_cost,
            "final_total_wealth": account_value,
            "final_profit_loss": account_value - INITIAL_CAPITAL,
            "return_percent": (account_value / INITIAL_CAPITAL - 1.0) * 100.0,
            "eligible_minutes": eligible_minutes,
            "buys": self.buys,
            "sales": sales,
            "skipped_buys": self.skipped_buys,
            "participation_percent_of_eligible_minutes": (
                100.0 * self.buys / eligible_minutes if eligible_minutes else None
            ),
            "realized_profit_before_final_liquidation": self.realized_profit,
            "max_open_lots": self.max_open_lots,
            "average_holding_minutes_closed": (
                self.hold_minutes_sum / sales if sales else None
            ),
            "maximum_holding_minutes_closed": self.max_hold_minutes,
            "oldest_open_lot_age_minutes": oldest_open_age,
            "closed_within_1_hour_percent": (
                100.0 * self.closed_within_60m / sales if sales else None
            ),
            "closed_within_1_day_percent": (
                100.0 * self.closed_within_1d / sales if sales else None
            ),
            "closed_within_7_days_percent": (
                100.0 * self.closed_within_7d / sales if sales else None
            ),
            "closed_within_30_days_percent": (
                100.0 * self.closed_within_30d / sales if sales else None
            ),
            "maximum_drawdown_percent": self.max_drawdown * 100.0,
            "maximum_drawdown_timestamp": self.max_drawdown_ts,
        }


def valid_price(value: str, field: str, where: str) -> float:
    x = float(value)
    if not (math.isfinite(x) and x > 0):
        raise ValueError(f"Invalid {field} at {where}: {value!r}")
    return x


def iter_btc(path: Path) -> Iterator[Bar]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "high", "close"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected BTC columns: {reader.fieldnames}")
        last_ts: int | None = None
        for row in reader:
            ts = int(row["timestamp"])
            if ts < BTC_START_TS:
                continue
            if ts > BTC_END_TS:
                break
            if last_ts is not None and ts <= last_ts:
                raise ValueError(f"BTC timestamps are not strictly increasing: {ts}")
            high = valid_price(row["high"], "high", str(ts))
            close = valid_price(row["close"], "close", str(ts))
            last_ts = ts
            yield Bar(
                ts=ts,
                high=high,
                close=close,
                date_label=datetime.fromtimestamp(ts, timezone.utc).date().isoformat(),
            )


def parse_equity_datetime(raw: str) -> datetime:
    raw = raw.strip().replace("T", " ")
    if raw.endswith("Z"):
        raw = raw[:-1]
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY)
    else:
        dt = dt.astimezone(NY)
    return dt


def iter_equity(path: Path) -> Iterator[Bar]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = {str(x).strip().lower(): x for x in (reader.fieldnames or [])}
        required = {"datetime", "high", "close"}
        if not required.issubset(fields):
            raise ValueError(f"Unexpected equity columns in {path}: {reader.fieldnames}")
        last_ts: int | None = None
        for row in reader:
            dt = parse_equity_datetime(row[fields["datetime"]])
            date_text = dt.date().isoformat()
            if date_text < EQUITY_START_DATE or date_text >= EQUITY_END_DATE_EXCLUSIVE:
                continue
            local_t = dt.timetz().replace(tzinfo=None)
            if not (RTH_START <= local_t < RTH_END):
                continue
            ts = int(dt.timestamp())
            if last_ts is not None:
                if ts == last_ts:
                    continue
                if ts < last_ts:
                    raise ValueError(f"Equity timestamps not ordered in {path}: {dt}")
            high = valid_price(row[fields["high"]], "high", dt.isoformat())
            close = valid_price(row[fields["close"]], "close", dt.isoformat())
            last_ts = ts
            yield Bar(ts=ts, high=high, close=close, date_label=date_text)


def run_stream(
    bars: Iterable[Bar], target_rates: list[float]
) -> tuple[dict[float, dict], dict]:
    strategies = {rate: Strategy(rate) for rate in target_rates}
    first_bar: Bar | None = None
    final_bar: Bar | None = None
    rows = 0
    sessions: set[str] = set()

    for bar in bars:
        if first_bar is None:
            first_bar = bar
        final_bar = bar
        rows += 1
        sessions.add(bar.date_label)
        for strategy in strategies.values():
            strategy.process(bar)

    if first_bar is None or final_bar is None:
        raise RuntimeError("No eligible bars found")

    buy_hold_final = (
        INITIAL_CAPITAL
        * (final_bar.close / first_bar.close)
        * EXIT_VALUE_FACTOR
    )
    benchmark = {
        "first_timestamp": first_bar.ts,
        "final_timestamp": final_bar.ts,
        "first_close": first_bar.close,
        "final_close": final_bar.close,
        "underlying_price_change_percent": (
            final_bar.close / first_bar.close - 1.0
        ) * 100.0,
        "eligible_minutes": rows,
        "trading_sessions": len(sessions),
        "buy_and_hold_final_wealth": buy_hold_final,
        "buy_and_hold_profit_loss": buy_hold_final - INITIAL_CAPITAL,
        "buy_and_hold_return_percent": (
            buy_hold_final / INITIAL_CAPITAL - 1.0
        ) * 100.0,
    }
    results = {
        rate: strategy.result(final_bar.ts, final_bar.close, rows)
        for rate, strategy in strategies.items()
    }
    return results, benchmark


def discover_equity_files(directory: Path) -> list[tuple[str, str, Path]]:
    output: list[tuple[str, str, Path]] = []
    for path in sorted(directory.glob("*.csv")):
        stem = path.stem
        if "_" not in stem:
            continue
        asset_type, ticker = stem.split("_", 1)
        if asset_type not in {"stock", "etf"}:
            continue
        output.append((asset_type, ticker.upper(), path))
    if not output:
        raise RuntimeError(f"No stock_*.csv or etf_*.csv files in {directory}")
    return output


def write_flat_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_metadata() -> dict:
    return {
        "initial_capital_per_asset": INITIAL_CAPITAL,
        "cash_allocation_per_eligible_minute": BUY_ALLOCATION,
        "buy_fee_percent": FEE * 100.0,
        "sell_fee_percent": FEE * 100.0,
        "buy_slippage_percent": SLIPPAGE * 100.0,
        "sell_slippage_percent": SLIPPAGE * 100.0,
        "separate_lots": True,
        "forced_exit": None,
        "execution_order": "take-profit older lots using candle high, then buy at candle close",
        "same_candle_exit_allowed": False,
        "target_fill_price": "individual target price, without gap price improvement",
        "stock_session": "regular trading hours 09:30-16:00 America/New_York",
        "stock_price_adjustment": "split-adjusted only",
        "dividends": "excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc", type=Path, required=True)
    parser.add_argument("--equities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    btc_results, btc_benchmark = run_stream(iter_btc(args.btc), [0.02])
    btc_output = {
        "asset": "BTC/USD",
        "venue": "Bitstamp",
        "period": {
            "start_utc": datetime.fromtimestamp(BTC_START_TS, timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(BTC_END_TS, timezone.utc).isoformat(),
        },
        "model": model_metadata(),
        "benchmark": btc_benchmark,
        "strategies": [btc_results[0.02]],
    }

    cross_assets: list[dict] = []
    flat_rows: list[dict] = []
    for asset_type, ticker, path in discover_equity_files(args.equities):
        target_results, benchmark = run_stream(iter_equity(path), [0.01, 0.02])
        asset_record = {
            "asset_type": asset_type,
            "ticker": ticker,
            "period": {
                "start_eastern": EQUITY_START_DATE,
                "end_eastern_exclusive": EQUITY_END_DATE_EXCLUSIVE,
            },
            "data_adjustment": "adj_split",
            "dividends_included": False,
            "benchmark": benchmark,
            "strategies": [target_results[0.01], target_results[0.02]],
        }
        cross_assets.append(asset_record)
        for rate in [0.01, 0.02]:
            flat_rows.append(
                {
                    "asset_type": asset_type,
                    "ticker": ticker,
                    **benchmark,
                    **target_results[rate],
                }
            )

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata(),
        "btc_five_year_2pct": btc_output,
        "us_equities_one_year": cross_assets,
    }
    (args.output / "summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    write_flat_csv(args.output / "cross_asset_results.csv", flat_rows)
    write_flat_csv(
        args.output / "btc_2pct_result.csv",
        [{**btc_benchmark, **btc_results[0.02]}],
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
