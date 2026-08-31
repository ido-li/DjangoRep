#!/usr/bin/env python3
"""Backtest a $1-per-minute BTC/USD strategy with a seven-day time exit.

Each lot:
- receives a take-profit target producing 0.10% net profit after modeled costs;
- is force-sold at the close after 7 days if that target was not reached;
- is kept separate from every other lot.

Execution order each minute:
1. Fill old take-profit orders touched by the candle high.
2. Force-sell any remaining lots whose seven-day deadline has arrived, at close.
3. Use available cash to make at most one new $1 purchase at close.

Costs:
- 0.10% fee on each side.
- 0.01% adverse slippage on each side.
"""

from __future__ import annotations

import csv
import gzip
import heapq
import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INITIAL_CAPITAL = 300.0
BUY_ALLOCATION = 1.0
FEE = 0.001
SLIPPAGE = 0.0001
NET_PROFIT_RATE = 0.001
MAX_HOLD_MINUTES = 7 * 24 * 60
MAX_HOLD_SECONDS = MAX_HOLD_MINUTES * 60

TAKE_PROFIT_PROCEEDS = BUY_ALLOCATION * (1.0 + NET_PROFIT_RATE)

TARGET_FACTOR = (
    (1.0 + NET_PROFIT_RATE)
    * (1.0 + FEE)
    * (1.0 + SLIPPAGE)
    / ((1.0 - FEE) * (1.0 - SLIPPAGE))
)

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

        # lot_id -> (entry reference price, entry timestamp, target price)
        self.active: dict[int, tuple[float, int, float]] = {}
        self.target_heap: list[tuple[float, int]] = []
        self.expiry_queue: deque[tuple[int, int]] = deque()
        self.serial = 0
        self.sum_inverse_entry = 0.0

        self.buys = 0
        self.skipped_buys = 0
        self.take_profit_sales = 0
        self.timeout_sales = 0
        self.timeout_profitable = 0
        self.timeout_losing = 0
        self.timeout_flat = 0
        self.timeout_pnl = 0.0
        self.net_realized_pnl = 0.0

        self.max_open_lots = 0
        self.hold_minutes_sum = 0
        self.take_profit_hold_minutes_sum = 0
        self.max_hold_minutes_closed = 0
        self.closed_within_60m = 0
        self.closed_within_1d = 0

        self.month_buys = 0
        self.month_skips = 0
        self.month_take_profit_sales = 0
        self.month_timeout_sales = 0
        self.month_timeout_pnl = 0.0
        self.month_net_realized_pnl = 0.0

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

    def _remove_lot(self, lot_id: int) -> tuple[float, int, float]:
        entry_price, entry_ts, target_price = self.active.pop(lot_id)
        self.sum_inverse_entry -= 1.0 / entry_price
        return entry_price, entry_ts, target_price

    def _record_holding_time(self, ts: int, entry_ts: int, take_profit: bool) -> None:
        held = max(1, (ts - entry_ts) // 60)
        self.hold_minutes_sum += held
        self.max_hold_minutes_closed = max(self.max_hold_minutes_closed, held)
        if take_profit:
            self.take_profit_hold_minutes_sum += held
        if held <= 60:
            self.closed_within_60m += 1
        if held <= 24 * 60:
            self.closed_within_1d += 1

    def _process_take_profits(self, ts: int, high: float) -> None:
        while self.target_heap and self.target_heap[0][0] <= high:
            _, lot_id = heapq.heappop(self.target_heap)
            if lot_id not in self.active:
                continue

            _, entry_ts, _ = self._remove_lot(lot_id)
            self.cash += TAKE_PROFIT_PROCEEDS
            pnl = BUY_ALLOCATION * NET_PROFIT_RATE

            self.take_profit_sales += 1
            self.net_realized_pnl += pnl
            self.month_take_profit_sales += 1
            self.month_net_realized_pnl += pnl
            self._record_holding_time(ts, entry_ts, take_profit=True)

    def _process_timeouts(self, ts: int, close: float) -> None:
        while self.expiry_queue and self.expiry_queue[0][0] <= ts:
            _, lot_id = self.expiry_queue.popleft()
            if lot_id not in self.active:
                continue

            entry_price, entry_ts, _ = self._remove_lot(lot_id)
            proceeds = close * EXIT_VALUE_FACTOR / entry_price
            pnl = proceeds - BUY_ALLOCATION
            self.cash += proceeds

            self.timeout_sales += 1
            self.timeout_pnl += pnl
            self.net_realized_pnl += pnl
            self.month_timeout_sales += 1
            self.month_timeout_pnl += pnl
            self.month_net_realized_pnl += pnl

            if pnl > 1e-12:
                self.timeout_profitable += 1
            elif pnl < -1e-12:
                self.timeout_losing += 1
            else:
                self.timeout_flat += 1

            self._record_holding_time(ts, entry_ts, take_profit=False)

    def _buy_if_possible(self, ts: int, close: float) -> None:
        if self.cash + 1e-10 < BUY_ALLOCATION:
            self.skipped_buys += 1
            self.month_skips += 1
            return

        self.cash -= BUY_ALLOCATION
        if abs(self.cash) < 1e-12:
            self.cash = 0.0

        self.serial += 1
        lot_id = self.serial
        target_price = close * TARGET_FACTOR
        self.active[lot_id] = (close, ts, target_price)
        heapq.heappush(self.target_heap, (target_price, lot_id))
        self.expiry_queue.append((ts + MAX_HOLD_SECONDS, lot_id))
        self.sum_inverse_entry += 1.0 / close

        self.buys += 1
        self.month_buys += 1
        self.max_open_lots = max(self.max_open_lots, len(self.active))

        # Remove stale timeout-exited entries before they accumulate for years.
        if len(self.target_heap) > max(10_000, 4 * len(self.active) + 1_000):
            self.target_heap = [
                (lot[2], active_id)
                for active_id, lot in self.active.items()
            ]
            heapq.heapify(self.target_heap)

    def process_minute(self, ts: int, high: float, close: float) -> None:
        self._process_take_profits(ts, high)
        self._process_timeouts(ts, close)
        self._buy_if_possible(ts, close)

        wealth = self.total_wealth(close)
        if wealth > self.peak_total_wealth:
            self.peak_total_wealth = wealth
        elif self.peak_total_wealth > 0:
            drawdown = (self.peak_total_wealth - wealth) / self.peak_total_wealth
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                self.max_drawdown_ts = ts

    def close_month(self, month_label: str, ts: int, close: float) -> None:
        # Only positive net realized P/L creates a withdrawal entitlement.
        self.withdrawal_due += (
            self.monthly_withdraw_fraction
            * max(0.0, self.month_net_realized_pnl)
        )

        # Do not withdraw money that would reduce marked-to-market account equity
        # below the original $300.
        equity_surplus = max(
            0.0,
            self.account_liquidation_value(close) - INITIAL_CAPITAL,
        )
        amount = min(self.cash, self.withdrawal_due, equity_surplus)
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
                "take_profit_sales": self.month_take_profit_sales,
                "timeout_sales": self.month_timeout_sales,
                "skipped_buys": self.month_skips,
                "take_profit_pnl": self.month_take_profit_sales * BUY_ALLOCATION * NET_PROFIT_RATE,
                "timeout_pnl": self.month_timeout_pnl,
                "net_realized_pnl": self.month_net_realized_pnl,
                "withdrawn_this_month": amount,
                "total_withdrawn": self.withdrawn,
                "cash": self.cash,
                "open_lots": len(self.active),
                "open_liquidation_value": open_value,
                "account_liquidation_value": account_value,
                "total_wealth_including_withdrawals": account_value + self.withdrawn,
            }
        )

        self.month_buys = 0
        self.month_skips = 0
        self.month_take_profit_sales = 0
        self.month_timeout_sales = 0
        self.month_timeout_pnl = 0.0
        self.month_net_realized_pnl = 0.0

    def snapshot_day(self, date_label: str, ts: int, close: float) -> None:
        account_value = self.account_liquidation_value(close)
        self.daily_rows.append(
            {
                "date": date_label,
                "timestamp": ts,
                "close": close,
                "cash": self.cash,
                "open_lots": len(self.active),
                "account_liquidation_value": account_value,
                "total_withdrawn": self.withdrawn,
                "total_wealth_including_withdrawals": account_value + self.withdrawn,
            }
        )

    def result(self, final_ts: int, final_price: float, rows_used: int) -> dict[str, Any]:
        open_value = final_price * EXIT_VALUE_FACTOR * self.sum_inverse_entry
        unrealized_pnl = open_value - float(len(self.active))
        account_value = self.cash + open_value
        total_wealth = account_value + self.withdrawn
        total_sales = self.take_profit_sales + self.timeout_sales

        oldest_open_age_minutes = 0
        if self.active:
            oldest_entry_ts = min(lot[1] for lot in self.active.values())
            oldest_open_age_minutes = (final_ts - oldest_entry_ts) // 60

        return {
            "strategy": self.name,
            "monthly_withdraw_fraction": self.monthly_withdraw_fraction,
            "initial_capital": INITIAL_CAPITAL,
            "maximum_holding_period_days": MAX_HOLD_MINUTES / (24 * 60),
            "final_cash": self.cash,
            "open_lots": len(self.active),
            "open_liquidation_value": open_value,
            "unrealized_pnl_on_open_lots": unrealized_pnl,
            "account_liquidation_value": account_value,
            "withdrawn_profit": self.withdrawn,
            "final_total_wealth": total_wealth,
            "final_profit_loss": total_wealth - INITIAL_CAPITAL,
            "return_percent": (total_wealth / INITIAL_CAPITAL - 1.0) * 100.0,
            "buys": self.buys,
            "participation_percent_of_minutes": 100.0 * self.buys / rows_used,
            "skipped_buys": self.skipped_buys,
            "total_sales": total_sales,
            "take_profit_sales": self.take_profit_sales,
            "timeout_sales": self.timeout_sales,
            "take_profit_exit_percent": 100.0 * self.take_profit_sales / total_sales if total_sales else None,
            "timeout_exit_percent": 100.0 * self.timeout_sales / total_sales if total_sales else None,
            "timeout_profitable_sales": self.timeout_profitable,
            "timeout_losing_sales": self.timeout_losing,
            "timeout_flat_sales": self.timeout_flat,
            "take_profit_realized_pnl": self.take_profit_sales * BUY_ALLOCATION * NET_PROFIT_RATE,
            "timeout_realized_pnl": self.timeout_pnl,
            "net_realized_pnl_before_final_liquidation": self.net_realized_pnl,
            "average_timeout_pnl_dollars": self.timeout_pnl / self.timeout_sales if self.timeout_sales else None,
            "average_timeout_pnl_percent": 100.0 * self.timeout_pnl / self.timeout_sales if self.timeout_sales else None,
            "max_open_lots": self.max_open_lots,
            "average_holding_minutes_all_closed": self.hold_minutes_sum / total_sales if total_sales else None,
            "average_holding_minutes_take_profit_sales": (
                self.take_profit_hold_minutes_sum / self.take_profit_sales
                if self.take_profit_sales else None
            ),
            "maximum_holding_minutes_closed": self.max_hold_minutes_closed,
            "oldest_open_lot_age_minutes": oldest_open_age_minutes,
            "closed_within_1_hour_percent": 100.0 * self.closed_within_60m / total_sales if total_sales else None,
            "closed_within_1_day_percent": 100.0 * self.closed_within_1d / total_sales if total_sales else None,
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
        raise SystemExit("Usage: btc_minute_backtest_7day_exit.py /path/to/btcusd.csv.gz")

    data_path = Path(sys.argv[1])
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    strategies = [
        Strategy("full_reinvestment_with_7day_exit", 0.0),
        Strategy("withdraw_50_percent_of_positive_monthly_net_profit_with_7day_exit", 0.5),
        Strategy("withdraw_100_percent_of_positive_monthly_net_profit_with_7day_exit", 1.0),
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

    buy_and_hold_final = INITIAL_CAPITAL * (final_price / first_price) * EXIT_VALUE_FACTOR

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
            "target_net_profit_percent_per_take_profit_lot": NET_PROFIT_RATE * 100.0,
            "required_reference_price_rise_percent": (TARGET_FACTOR - 1.0) * 100.0,
            "maximum_holding_period_days": MAX_HOLD_MINUTES / (24 * 60),
            "execution": (
                "take-profit old lots using candle high; force-sell remaining "
                "seven-day-old lots at candle close; then buy at candle close"
            ),
            "end_treatment": "mark all open lots to modeled final liquidation value",
        },
        "strategies": [s.result(final_ts, final_price, rows_used) for s in strategies],
        "buy_and_hold": {
            "initial_capital": INITIAL_CAPITAL,
            "final_total_wealth": buy_and_hold_final,
            "final_profit_loss": buy_and_hold_final - INITIAL_CAPITAL,
            "return_percent": (buy_and_hold_final / INITIAL_CAPITAL - 1.0) * 100.0,
        },
    }

    output_dir = Path("chatgpt_backtest/results_7day_exit")
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
