#!/usr/bin/env python3
"""Backtest a $1-per-minute BTC/USD take-profit strategy.

Assumptions:
- $300 initial cash.
- Buy at each minute close, after processing exits for that candle.
- A new lot cannot exit in its purchase candle.
- Each $1 allocation includes the buy fee.
- 0.10% fee and 0.01% slippage on each side.
- Each separate lot exits at a reference-market price that produces exactly
  0.10% net profit on its $1 cash allocation after all modeled costs.
- Exits are triggered by the subsequent candle high and filled at the adjusted
  target price (not at a better gap price).
- No stop loss, leverage, interest, taxes, spread beyond modeled slippage,
  minimum-order rule, or market impact.
"""

from __future__ import annotations

import csv
import gzip
import heapq
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INITIAL_CAPITAL = 300.0
BUY_ALLOCATION = 1.0
FEE = 0.001
SLIPPAGE = 0.0001
NET_PROFIT_RATE = 0.001
SALE_PROCEEDS = BUY_ALLOCATION * (1.0 + NET_PROFIT_RATE)

# Reference-price rise needed from the observed buy close to the observed
# take-profit trigger, after buy/sell fees and adverse slippage on both sides.
TARGET_FACTOR = (
    (1.0 + NET_PROFIT_RATE)
    * (1.0 + FEE)
    * (1.0 + SLIPPAGE)
    / ((1.0 - FEE) * (1.0 - SLIPPAGE))
)

# A $1 all-in lot bought at reference price P has this final-liquidation value:
# final_reference_price * EXIT_VALUE_FACTOR / P.
EXIT_VALUE_FACTOR = (
    BUY_ALLOCATION
    * (1.0 - FEE)
    * (1.0 - SLIPPAGE)
    / ((1.0 + FEE) * (1.0 + SLIPPAGE))
)

START_TS = int(datetime(2020, 1, 7, tzinfo=timezone.utc).timestamp())
END_TS = int(datetime(2025, 1, 7, tzinfo=timezone.utc).timestamp())


class Strategy:
    def __init__(self, name: str, monthly_withdraw_fraction: float) -> None:
        self.name = name
        self.monthly_withdraw_fraction = monthly_withdraw_fraction
        self.cash = INITIAL_CAPITAL
        self.withdrawn = 0.0
        self.withdrawal_due = 0.0

        # Heap item: (target reference price, serial, entry close, entry timestamp)
        self.open_lots: list[tuple[float, int, float, int]] = []
        self.serial = 0
        self.sum_inverse_entry = 0.0

        self.buys = 0
        self.sales = 0
        self.skipped_buys = 0
        self.max_open_lots = 0
        self.max_book_capital = INITIAL_CAPITAL

        self.hold_minutes_sum = 0
        self.max_hold_minutes = 0
        self.closed_within_60m = 0
        self.closed_within_1d = 0
        self.closed_within_7d = 0
        self.closed_within_30d = 0

        self.month_buys = 0
        self.month_sales = 0
        self.month_skips = 0
        self.month_profit = 0.0

        self.peak_total_wealth = INITIAL_CAPITAL
        self.max_drawdown = 0.0
        self.max_drawdown_ts: int | None = None

        self.monthly_rows: list[dict[str, Any]] = []
        self.daily_rows: list[dict[str, Any]] = []

    def account_liquidation_value(self, reference_price: float) -> float:
        open_value = reference_price * EXIT_VALUE_FACTOR * self.sum_inverse_entry
        return self.cash + open_value

    def total_wealth(self, reference_price: float) -> float:
        return self.withdrawn + self.account_liquidation_value(reference_price)

    def process_minute(self, ts: int, high: float, close: float) -> None:
        # Every lot in the heap was bought in a previous candle because entries
        # are inserted only after exits are checked.
        while self.open_lots and self.open_lots[0][0] <= high:
            _, _, entry_price, entry_ts = heapq.heappop(self.open_lots)
            self.sum_inverse_entry -= 1.0 / entry_price
            self.cash += SALE_PROCEEDS
            self.sales += 1
            self.month_sales += 1
            self.month_profit += BUY_ALLOCATION * NET_PROFIT_RATE

            held = max(1, (ts - entry_ts) // 60)
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
                (close * TARGET_FACTOR, self.serial, close, ts),
            )
            self.sum_inverse_entry += 1.0 / close
            self.buys += 1
            self.month_buys += 1
            self.max_open_lots = max(self.max_open_lots, len(self.open_lots))
        else:
            self.skipped_buys += 1
            self.month_skips += 1

        book_capital = self.cash + float(len(self.open_lots))
        self.max_book_capital = max(self.max_book_capital, book_capital)

        wealth = self.total_wealth(close)
        if wealth > self.peak_total_wealth:
            self.peak_total_wealth = wealth
        elif self.peak_total_wealth > 0:
            drawdown = (self.peak_total_wealth - wealth) / self.peak_total_wealth
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                self.max_drawdown_ts = ts

    def close_month(self, month_label: str, ts: int, close: float) -> None:
        # Monthly withdrawal is based only on realized profit generated during
        # the month. It never reduces cash + open-lot cost basis below $300.
        self.withdrawal_due += self.monthly_withdraw_fraction * self.month_profit
        book_surplus = max(
            0.0,
            self.cash + float(len(self.open_lots)) - INITIAL_CAPITAL,
        )
        amount = min(self.cash, self.withdrawal_due, book_surplus)
        if amount > 0:
            self.cash -= amount
            self.withdrawal_due -= amount
            self.withdrawn += amount

        open_value = close * EXIT_VALUE_FACTOR * self.sum_inverse_entry
        account_value = self.cash + open_value
        self.monthly_rows.append(
            {
                "month": month_label,
                "timestamp": ts,
                "close": close,
                "buys": self.month_buys,
                "sales": self.month_sales,
                "skipped_buys": self.month_skips,
                "realized_profit_this_month": self.month_profit,
                "withdrawn_this_month": amount,
                "total_withdrawn": self.withdrawn,
                "cash": self.cash,
                "open_lots": len(self.open_lots),
                "open_liquidation_value": open_value,
                "account_liquidation_value": account_value,
                "total_wealth_including_withdrawals": account_value + self.withdrawn,
            }
        )
        self.month_buys = 0
        self.month_sales = 0
        self.month_skips = 0
        self.month_profit = 0.0

    def snapshot_day(self, date_label: str, ts: int, close: float) -> None:
        account_value = self.account_liquidation_value(close)
        self.daily_rows.append(
            {
                "date": date_label,
                "timestamp": ts,
                "close": close,
                "cash": self.cash,
                "open_lots": len(self.open_lots),
                "account_liquidation_value": account_value,
                "total_withdrawn": self.withdrawn,
                "total_wealth_including_withdrawals": account_value + self.withdrawn,
            }
        )

    def result(self, final_ts: int, final_price: float) -> dict[str, Any]:
        open_value = final_price * EXIT_VALUE_FACTOR * self.sum_inverse_entry
        unrealized_pnl = open_value - float(len(self.open_lots))
        account_value = self.cash + open_value
        total_wealth = account_value + self.withdrawn
        oldest_open_age_minutes = 0
        if self.open_lots:
            oldest_entry_ts = min(item[3] for item in self.open_lots)
            oldest_open_age_minutes = (final_ts - oldest_entry_ts) // 60

        sales = self.sales
        return {
            "strategy": self.name,
            "monthly_withdraw_fraction": self.monthly_withdraw_fraction,
            "initial_capital": INITIAL_CAPITAL,
            "final_cash": self.cash,
            "open_lots": len(self.open_lots),
            "open_liquidation_value": open_value,
            "unrealized_pnl_on_open_lots": unrealized_pnl,
            "account_liquidation_value": account_value,
            "withdrawn_profit": self.withdrawn,
            "final_total_wealth": total_wealth,
            "final_profit_loss": total_wealth - INITIAL_CAPITAL,
            "return_percent": (total_wealth / INITIAL_CAPITAL - 1.0) * 100.0,
            "buys": self.buys,
            "sales": sales,
            "skipped_buys": self.skipped_buys,
            "realized_profit_before_final_liquidation": sales * BUY_ALLOCATION * NET_PROFIT_RATE,
            "max_open_lots": self.max_open_lots,
            "max_book_capital": self.max_book_capital,
            "average_holding_minutes_closed": self.hold_minutes_sum / sales if sales else None,
            "maximum_holding_minutes_closed": self.max_hold_minutes,
            "oldest_open_lot_age_minutes": oldest_open_age_minutes,
            "closed_within_1_hour_percent": 100.0 * self.closed_within_60m / sales if sales else None,
            "closed_within_1_day_percent": 100.0 * self.closed_within_1d / sales if sales else None,
            "closed_within_7_days_percent": 100.0 * self.closed_within_7d / sales if sales else None,
            "closed_within_30_days_percent": 100.0 * self.closed_within_30d / sales if sales else None,
            "maximum_drawdown_percent_at_minute_closes": self.max_drawdown * 100.0,
            "maximum_drawdown_timestamp": self.max_drawdown_ts,
            "unpaid_withdrawal_due": self.withdrawal_due,
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: btc_minute_backtest.py /path/to/btcusd.csv.gz")

    data_path = Path(sys.argv[1])
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    strategies = [
        Strategy("full_reinvestment", 0.0),
        Strategy("withdraw_50_percent_of_monthly_realized_profit", 0.5),
        Strategy("withdraw_100_percent_of_monthly_realized_profit", 1.0),
    ]

    first_price: float | None = None
    final_price: float | None = None
    final_ts: int | None = None
    rows_used = 0
    current_day_id: int | None = None
    current_month: tuple[int, int] | None = None
    last_ts: int | None = None
    last_close: float | None = None

    with gzip.open(data_path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "high", "close"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")

        for row in reader:
            ts = int(row["timestamp"])
            if ts < START_TS:
                continue
            if ts > END_TS:
                break

            high = float(row["high"])
            close = float(row["close"])
            if not (math.isfinite(high) and math.isfinite(close) and high > 0 and close > 0):
                raise ValueError(f"Invalid price at {ts}: high={high}, close={close}")

            day_id = ts // 86400
            if current_day_id is None:
                dt = datetime.fromtimestamp(ts, timezone.utc)
                current_day_id = day_id
                current_month = (dt.year, dt.month)
            elif day_id != current_day_id:
                assert last_ts is not None and last_close is not None and current_month is not None
                dt = datetime.fromtimestamp(ts, timezone.utc)
                new_month = (dt.year, dt.month)
                if new_month != current_month:
                    month_label = f"{current_month[0]:04d}-{current_month[1]:02d}"
                    for strategy in strategies:
                        strategy.close_month(month_label, last_ts, last_close)
                    current_month = new_month

                previous_date = datetime.fromtimestamp(last_ts, timezone.utc).date().isoformat()
                for strategy in strategies:
                    strategy.snapshot_day(previous_date, last_ts, last_close)
                current_day_id = day_id

            if first_price is None:
                first_price = close

            for strategy in strategies:
                strategy.process_minute(ts, high, close)

            rows_used += 1
            last_ts = ts
            last_close = close
            final_ts = ts
            final_price = close

    if rows_used == 0 or first_price is None or final_price is None or final_ts is None:
        raise RuntimeError("No rows found inside the requested period")

    assert current_month is not None and last_ts is not None and last_close is not None
    final_month_label = f"{current_month[0]:04d}-{current_month[1]:02d}"
    final_date_label = datetime.fromtimestamp(final_ts, timezone.utc).date().isoformat()
    for strategy in strategies:
        strategy.close_month(final_month_label, final_ts, final_price)
        strategy.snapshot_day(final_date_label, final_ts, final_price)

    # Same all-in entry and final exit cost model as the strategy.
    buy_and_hold_final = (
        INITIAL_CAPITAL * (final_price / first_price) * EXIT_VALUE_FACTOR
    )

    output = {
        "test": {
            "asset": "BTC/USD on Bitstamp",
            "start_utc": datetime.fromtimestamp(START_TS, timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(END_TS, timezone.utc).isoformat(),
            "rows_used": rows_used,
            "first_close": first_price,
            "final_close": final_price,
            "underlying_price_change_percent": (final_price / first_price - 1.0) * 100.0,
            "initial_capital": INITIAL_CAPITAL,
            "buy_allocation_each_eligible_minute": BUY_ALLOCATION,
            "fee_each_side_percent": FEE * 100.0,
            "slippage_each_side_percent": SLIPPAGE * 100.0,
            "target_net_profit_percent_per_closed_lot": NET_PROFIT_RATE * 100.0,
            "required_reference_price_rise_percent": (TARGET_FACTOR - 1.0) * 100.0,
            "execution": "sell eligible older lots first using candle high; then buy at candle close",
            "end_treatment": "mark all open lots to modeled final liquidation value",
        },
        "strategies": [s.result(final_ts, final_price) for s in strategies],
        "buy_and_hold": {
            "initial_capital": INITIAL_CAPITAL,
            "final_total_wealth": buy_and_hold_final,
            "final_profit_loss": buy_and_hold_final - INITIAL_CAPITAL,
            "return_percent": (buy_and_hold_final / INITIAL_CAPITAL - 1.0) * 100.0,
        },
    }

    output_dir = Path("chatgpt_backtest/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    for strategy in strategies:
        safe_name = strategy.name.replace(" ", "_")
        write_csv(output_dir / f"monthly_{safe_name}.csv", strategy.monthly_rows)
        write_csv(output_dir / f"daily_{safe_name}.csv", strategy.daily_rows)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
