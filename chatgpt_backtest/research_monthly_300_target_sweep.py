#!/usr/bin/env python3
"""One-year target sweep funded with $300 per month.

Keeps the prior strategy unchanged except for external funding:
- 12 contributions of $300, beginning at the first eligible bar and then on
  monthly anniversaries. If an equity market is closed, the contribution is
  added on the first subsequent regular-hours bar.
- $1 all-in fractional lot bought each eligible minute when cash permits.
- Each lot is separate and exits only at its individual take-profit target.
- Net targets 1% through 10%; no forced exits.
- Fee 0.10% and adverse slippage 0.01% on each side.
- Open lots are marked to modeled liquidation value at the final bar.

Period:
- BTC/USD: 2022-09-30 00:00 UTC through 2023-09-29 23:59 UTC.
- U.S. equities: 2022-09-30 through 2023-09-29, 09:30-16:00 ET.
"""

from __future__ import annotations

import argparse
import calendar
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

MONTHLY_CONTRIBUTION = 300.0
CONTRIBUTION_COUNT = 12
TOTAL_CONTRIBUTIONS = MONTHLY_CONTRIBUTION * CONTRIBUTION_COUNT
BUY_ALLOCATION = 1.0
FEE = 0.001
SLIPPAGE = 0.0001
TARGET_RATES = [i / 100.0 for i in range(1, 11)]
YEAR_SECONDS = 365.2425 * 24 * 60 * 60

# One dollar of all-in purchase cost, liquidated at the same reference price.
EXIT_VALUE_FACTOR = (
    (1.0 - FEE) * (1.0 - SLIPPAGE) / ((1.0 + FEE) * (1.0 + SLIPPAGE))
)

BTC_START_DT = datetime(2022, 9, 30, 0, 0, tzinfo=timezone.utc)
BTC_END_DT = datetime(2023, 9, 29, 23, 59, tzinfo=timezone.utc)
BTC_START_TS = int(BTC_START_DT.timestamp())
BTC_END_TS = int(BTC_END_DT.timestamp())

NY = ZoneInfo("America/New_York")
EQUITY_START_DATE = "2022-09-30"
EQUITY_END_DATE_EXCLUSIVE = "2023-09-30"
RTH_START = time(9, 30)
RTH_END = time(16, 0)
EQUITY_SCHEDULE_START = datetime(2022, 9, 30, 9, 30, tzinfo=NY)


@dataclass(frozen=True)
class Bar:
    ts: int
    high: float
    close: float
    dt: datetime
    date_label: str


@dataclass(frozen=True)
class ContributionEvent:
    scheduled_ts: int
    actual_ts: int
    amount: float
    entry_price: float
    actual_datetime: str


def add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def make_schedule(start: datetime) -> list[int]:
    return [int(add_months(start, i).timestamp()) for i in range(CONTRIBUTION_COUNT)]


def xirr(cashflows: list[tuple[int, float]]) -> float | None:
    """Return annualized money-weighted return for conventional cash flows."""
    if not cashflows or not any(v < 0 for _, v in cashflows) or not any(v > 0 for _, v in cashflows):
        return None
    cashflows = sorted(cashflows)
    t0 = cashflows[0][0]

    def npv(rate: float) -> float:
        base = 1.0 + rate
        return sum(value / (base ** ((ts - t0) / YEAR_SECONDS)) for ts, value in cashflows)

    low = -0.999999
    high = 1.0
    low_value = npv(low)
    high_value = npv(high)
    while high_value > 0 and high < 1_000_000:
        high = high * 2.0 + 1.0
        high_value = npv(high)
    if not (low_value > 0 and high_value < 0):
        return None
    for _ in range(200):
        mid = (low + high) / 2.0
        value = npv(mid)
        if value > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


class Strategy:
    def __init__(self, target_rate: float) -> None:
        self.target_rate = target_rate
        self.sale_proceeds = BUY_ALLOCATION * (1.0 + target_rate)
        self.target_factor = (
            (1.0 + target_rate)
            * (1.0 + FEE)
            * (1.0 + SLIPPAGE)
            / ((1.0 - FEE) * (1.0 - SLIPPAGE))
        )

        self.cash = 0.0
        self.total_contributed = 0.0
        self.contribution_count = 0
        self.contribution_cashflows: list[tuple[int, float]] = []

        # (target reference price, serial, entry price, entry timestamp)
        self.open_lots: list[tuple[float, int, float, int]] = []
        self.serial = 0
        self.sum_inverse_entry = 0.0
        self.buys = 0
        self.sales = 0
        self.skipped_buys = 0
        self.realized_profit = 0.0
        self.max_open_lots = 0

        self.hold_minutes_sum = 0
        self.max_hold_minutes = 0
        self.closed_within_1d = 0
        self.closed_within_7d = 0
        self.closed_within_30d = 0

        # Unitized NAV removes the mechanical effect of deposits from drawdown.
        self.fund_units = 0.0
        self.peak_nav = 1.0
        self.max_drawdown = 0.0
        self.max_drawdown_ts: int | None = None
        self.final_nav = 1.0

    def liquidation_value(self, reference_price: float) -> float:
        return self.cash + reference_price * EXIT_VALUE_FACTOR * self.sum_inverse_entry

    def close_profitable_lots(self, bar: Bar) -> None:
        while self.open_lots and self.open_lots[0][0] <= bar.high:
            _, _, entry_price, entry_ts = heapq.heappop(self.open_lots)
            self.sum_inverse_entry -= 1.0 / entry_price
            self.cash += self.sale_proceeds
            self.sales += 1
            self.realized_profit += BUY_ALLOCATION * self.target_rate
            held = max(1, (bar.ts - entry_ts) // 60)
            self.hold_minutes_sum += held
            self.max_hold_minutes = max(self.max_hold_minutes, held)
            self.closed_within_1d += int(held <= 24 * 60)
            self.closed_within_7d += int(held <= 7 * 24 * 60)
            self.closed_within_30d += int(held <= 30 * 24 * 60)

    def add_contribution(self, ts: int, amount: float, reference_price: float) -> None:
        equity_before = self.liquidation_value(reference_price)
        if self.fund_units <= 0:
            self.fund_units = amount
        else:
            nav_before = equity_before / self.fund_units
            if nav_before <= 0:
                raise RuntimeError("Non-positive unit NAV before contribution")
            self.fund_units += amount / nav_before
        self.cash += amount
        self.total_contributed += amount
        self.contribution_count += 1
        self.contribution_cashflows.append((ts, -amount))

    def buy_if_possible(self, bar: Bar) -> None:
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

    def update_nav(self, bar: Bar) -> None:
        if self.fund_units <= 0:
            return
        nav = self.liquidation_value(bar.close) / self.fund_units
        self.final_nav = nav
        if nav > self.peak_nav:
            self.peak_nav = nav
        elif self.peak_nav > 0:
            drawdown = (self.peak_nav - nav) / self.peak_nav
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                self.max_drawdown_ts = bar.ts

    def process(self, bar: Bar, contribution: float) -> None:
        self.close_profitable_lots(bar)
        if contribution > 0:
            self.add_contribution(bar.ts, contribution, bar.close)
        self.buy_if_possible(bar)
        self.update_nav(bar)

    def result(self, final_bar: Bar, eligible_minutes: int) -> dict:
        open_value = final_bar.close * EXIT_VALUE_FACTOR * self.sum_inverse_entry
        open_cost = len(self.open_lots) * BUY_ALLOCATION
        wealth = self.cash + open_value
        cashflows = list(self.contribution_cashflows) + [(final_bar.ts, wealth)]
        irr = xirr(cashflows)
        oldest_open_minutes = 0
        if self.open_lots:
            oldest_open_minutes = (final_bar.ts - min(item[3] for item in self.open_lots)) // 60
        return {
            "target_net_profit_percent": self.target_rate * 100.0,
            "required_reference_price_rise_percent": (self.target_factor - 1.0) * 100.0,
            "monthly_contribution": MONTHLY_CONTRIBUTION,
            "contribution_count": self.contribution_count,
            "total_contributed": self.total_contributed,
            "final_total_wealth": wealth,
            "profit_loss": wealth - self.total_contributed,
            "simple_return_on_contributions_percent": (wealth / self.total_contributed - 1.0) * 100.0,
            "money_weighted_annual_return_percent": None if irr is None else irr * 100.0,
            "cash_flow_adjusted_time_weighted_return_percent": (self.final_nav - 1.0) * 100.0,
            "final_cash": self.cash,
            "open_lots": len(self.open_lots),
            "open_lot_cost_basis": open_cost,
            "open_liquidation_value": open_value,
            "unrealized_pnl_on_open_lots": open_value - open_cost,
            "realized_profit": self.realized_profit,
            "buys": self.buys,
            "sales": self.sales,
            "skipped_buys": self.skipped_buys,
            "participation_percent_of_eligible_minutes": 100.0 * self.buys / eligible_minutes,
            "max_open_lots": self.max_open_lots,
            "average_holding_days_closed": (
                self.hold_minutes_sum / self.sales / 1440.0 if self.sales else None
            ),
            "maximum_holding_days_closed": self.max_hold_minutes / 1440.0,
            "oldest_open_lot_age_days": oldest_open_minutes / 1440.0,
            "closed_within_1_day_percent": 100.0 * self.closed_within_1d / self.sales if self.sales else None,
            "closed_within_7_days_percent": 100.0 * self.closed_within_7d / self.sales if self.sales else None,
            "closed_within_30_days_percent": 100.0 * self.closed_within_30d / self.sales if self.sales else None,
            "cash_flow_adjusted_maximum_drawdown_percent": self.max_drawdown * 100.0,
            "maximum_drawdown_timestamp": self.max_drawdown_ts,
        }


def valid_price(value: str, field: str, where: str) -> float:
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"Invalid {field} at {where}: {value!r}")
    return number


def iter_btc(path: Path) -> Iterator[Bar]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "high", "close"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected BTC columns: {reader.fieldnames}")
        previous_ts: int | None = None
        for row in reader:
            ts = int(row["timestamp"])
            if ts < BTC_START_TS:
                continue
            if ts > BTC_END_TS:
                break
            if previous_ts is not None and ts <= previous_ts:
                raise ValueError(f"BTC timestamps not increasing at {ts}")
            previous_ts = ts
            dt = datetime.fromtimestamp(ts, timezone.utc)
            yield Bar(
                ts=ts,
                high=valid_price(row["high"], "high", str(ts)),
                close=valid_price(row["close"], "close", str(ts)),
                dt=dt,
                date_label=dt.date().isoformat(),
            )


def parse_equity_datetime(raw: str) -> datetime:
    text = raw.strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1]
    dt = datetime.fromisoformat(text)
    return dt.replace(tzinfo=NY) if dt.tzinfo is None else dt.astimezone(NY)


def iter_equity(path: Path) -> Iterator[Bar]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {str(name).strip().lower(): name for name in (reader.fieldnames or [])}
        required = {"datetime", "high", "close"}
        if not required.issubset(fields):
            raise ValueError(f"Unexpected equity columns in {path}: {reader.fieldnames}")
        previous_ts: int | None = None
        for row in reader:
            dt = parse_equity_datetime(row[fields["datetime"]])
            date_label = dt.date().isoformat()
            if date_label < EQUITY_START_DATE or date_label >= EQUITY_END_DATE_EXCLUSIVE:
                continue
            local_time = dt.timetz().replace(tzinfo=None)
            if not (RTH_START <= local_time < RTH_END):
                continue
            ts = int(dt.timestamp())
            if previous_ts is not None:
                if ts == previous_ts:
                    continue
                if ts < previous_ts:
                    raise ValueError(f"Equity timestamps not increasing in {path}: {dt}")
            previous_ts = ts
            yield Bar(
                ts=ts,
                high=valid_price(row[fields["high"]], "high", dt.isoformat()),
                close=valid_price(row[fields["close"]], "close", dt.isoformat()),
                dt=dt,
                date_label=date_label,
            )


def run_stream(
    bars: Iterable[Bar], schedule: list[int], target_rates: list[float]
) -> tuple[list[dict], dict, list[ContributionEvent]]:
    strategies = [Strategy(rate) for rate in target_rates]
    first_bar: Bar | None = None
    final_bar: Bar | None = None
    rows = 0
    sessions: set[str] = set()
    schedule_index = 0
    contribution_events: list[ContributionEvent] = []

    for bar in bars:
        if first_bar is None:
            first_bar = bar
        final_bar = bar
        rows += 1
        sessions.add(bar.date_label)

        contribution = 0.0
        while schedule_index < len(schedule) and bar.ts >= schedule[schedule_index]:
            contribution += MONTHLY_CONTRIBUTION
            contribution_events.append(
                ContributionEvent(
                    scheduled_ts=schedule[schedule_index],
                    actual_ts=bar.ts,
                    amount=MONTHLY_CONTRIBUTION,
                    entry_price=bar.close,
                    actual_datetime=bar.dt.isoformat(),
                )
            )
            schedule_index += 1

        for strategy in strategies:
            strategy.process(bar, contribution)

    if first_bar is None or final_bar is None:
        raise RuntimeError("No eligible bars")
    if len(contribution_events) != CONTRIBUTION_COUNT:
        raise RuntimeError(
            f"Expected {CONTRIBUTION_COUNT} contributions, got {len(contribution_events)}"
        )

    dca_wealth = sum(
        event.amount * (final_bar.close / event.entry_price) * EXIT_VALUE_FACTOR
        for event in contribution_events
    )
    dca_flows = [(event.actual_ts, -event.amount) for event in contribution_events]
    dca_irr = xirr(dca_flows + [(final_bar.ts, dca_wealth)])
    benchmark = {
        "first_timestamp": first_bar.ts,
        "final_timestamp": final_bar.ts,
        "first_datetime": first_bar.dt.isoformat(),
        "final_datetime": final_bar.dt.isoformat(),
        "first_close": first_bar.close,
        "final_close": final_bar.close,
        "underlying_price_change_percent": (final_bar.close / first_bar.close - 1.0) * 100.0,
        "eligible_minutes": rows,
        "trading_sessions": len(sessions),
        "contribution_count": len(contribution_events),
        "monthly_contribution": MONTHLY_CONTRIBUTION,
        "total_contributed": TOTAL_CONTRIBUTIONS,
        "dca_buy_and_hold_final_wealth": dca_wealth,
        "dca_buy_and_hold_profit_loss": dca_wealth - TOTAL_CONTRIBUTIONS,
        "dca_buy_and_hold_simple_return_percent": (dca_wealth / TOTAL_CONTRIBUTIONS - 1.0) * 100.0,
        "dca_buy_and_hold_money_weighted_annual_return_percent": (
            None if dca_irr is None else dca_irr * 100.0
        ),
    }
    return [strategy.result(final_bar, rows) for strategy in strategies], benchmark, contribution_events


def discover_equity_files(directory: Path) -> list[tuple[str, str, Path]]:
    discovered: list[tuple[str, str, Path]] = []
    for path in sorted(directory.glob("*.csv")):
        if "_" not in path.stem:
            continue
        asset_type, ticker = path.stem.split("_", 1)
        if asset_type in {"stock", "etf"}:
            discovered.append((asset_type, ticker.upper(), path))
    if not discovered:
        raise RuntimeError(f"No stock_*.csv or etf_*.csv files in {directory}")
    return discovered


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_result(records: list[dict]) -> dict:
    return max(records, key=lambda record: record["simple_return_on_contributions_percent"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc", required=True, type=Path)
    parser.add_argument("--equities", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    assets: list[dict] = []
    all_rows: list[dict] = []
    schedule_rows: list[dict] = []

    btc_strategies, btc_benchmark, btc_events = run_stream(
        iter_btc(args.btc), make_schedule(BTC_START_DT), TARGET_RATES
    )
    assets.append(
        {
            "asset_type": "crypto",
            "ticker": "BTCUSD",
            "benchmark": btc_benchmark,
            "strategies": btc_strategies,
        }
    )
    for event in btc_events:
        schedule_rows.append({"ticker": "BTCUSD", **event.__dict__})
    for result in btc_strategies:
        all_rows.append({"asset_type": "crypto", "ticker": "BTCUSD", **btc_benchmark, **result})

    for asset_type, ticker, path in discover_equity_files(args.equities):
        strategies, benchmark, events = run_stream(
            iter_equity(path), make_schedule(EQUITY_SCHEDULE_START), TARGET_RATES
        )
        assets.append(
            {
                "asset_type": asset_type,
                "ticker": ticker,
                "benchmark": benchmark,
                "strategies": strategies,
            }
        )
        for event in events:
            schedule_rows.append({"ticker": ticker, **event.__dict__})
        for result in strategies:
            all_rows.append({"asset_type": asset_type, "ticker": ticker, **benchmark, **result})

    best_rows: list[dict] = []
    for asset in assets:
        best = best_result(asset["strategies"])
        benchmark = asset["benchmark"]
        best_rows.append(
            {
                "asset_type": asset["asset_type"],
                "ticker": asset["ticker"],
                "best_target_percent": best["target_net_profit_percent"],
                "total_contributed": best["total_contributed"],
                "best_final_wealth": best["final_total_wealth"],
                "best_profit_loss": best["profit_loss"],
                "best_simple_return_percent": best["simple_return_on_contributions_percent"],
                "best_money_weighted_annual_return_percent": best["money_weighted_annual_return_percent"],
                "dca_final_wealth": benchmark["dca_buy_and_hold_final_wealth"],
                "dca_simple_return_percent": benchmark["dca_buy_and_hold_simple_return_percent"],
                "dca_money_weighted_annual_return_percent": benchmark["dca_buy_and_hold_money_weighted_annual_return_percent"],
                "excess_final_wealth_vs_dca": best["final_total_wealth"] - benchmark["dca_buy_and_hold_final_wealth"],
                "excess_simple_return_vs_dca_points": best["simple_return_on_contributions_percent"] - benchmark["dca_buy_and_hold_simple_return_percent"],
                "buys": best["buys"],
                "sales": best["sales"],
                "open_lots": best["open_lots"],
                "participation_percent": best["participation_percent_of_eligible_minutes"],
                "average_holding_days": best["average_holding_days_closed"],
                "cash_flow_adjusted_maximum_drawdown_percent": best["cash_flow_adjusted_maximum_drawdown_percent"],
            }
        )

    target_summary: list[dict] = []
    for target in range(1, 11):
        target_records = [
            row for row in all_rows if round(row["target_net_profit_percent"]) == target
        ]
        equity_records = [row for row in target_records if row["asset_type"] != "crypto"]
        target_summary.append(
            {
                "target_percent": target,
                "all_nine_average_simple_return_percent": sum(
                    row["simple_return_on_contributions_percent"] for row in target_records
                ) / len(target_records),
                "equity_average_simple_return_percent": sum(
                    row["simple_return_on_contributions_percent"] for row in equity_records
                ) / len(equity_records),
                "btc_simple_return_percent": next(
                    row["simple_return_on_contributions_percent"]
                    for row in target_records
                    if row["ticker"] == "BTCUSD"
                ),
                "assets_beating_dca": sum(
                    row["final_total_wealth"] > row["dca_buy_and_hold_final_wealth"]
                    for row in target_records
                ),
                "equities_beating_dca": sum(
                    row["final_total_wealth"] > row["dca_buy_and_hold_final_wealth"]
                    for row in equity_records
                ),
                "combined_final_wealth_all_nine": sum(
                    row["final_total_wealth"] for row in target_records
                ),
                "combined_total_contributions_all_nine": TOTAL_CONTRIBUTIONS * len(target_records),
            }
        )

    best_universal = max(target_summary, key=lambda row: row["all_nine_average_simple_return_percent"])
    best_equity_universal = max(target_summary, key=lambda row: row["equity_average_simple_return_percent"])
    overview = {
        "period": {
            "btc": "2022-09-30 00:00 UTC through 2023-09-29 23:59 UTC",
            "equities": "2022-09-30 through 2023-09-29 regular U.S. market hours",
        },
        "funding": {
            "monthly_contribution": MONTHLY_CONTRIBUTION,
            "number_of_contributions": CONTRIBUTION_COUNT,
            "total_contributed_per_asset": TOTAL_CONTRIBUTIONS,
            "timing": "first eligible bar, then monthly anniversaries; closed equity markets roll to the next eligible bar",
        },
        "model": {
            "buy_allocation_each_eligible_minute": BUY_ALLOCATION,
            "fee_each_side_percent": FEE * 100.0,
            "slippage_each_side_percent": SLIPPAGE * 100.0,
            "targets_net_percent": list(range(1, 11)),
            "forced_exit": None,
            "dividends": "excluded",
        },
        "best_universal_all_nine": best_universal,
        "best_universal_equities": best_equity_universal,
        "best_by_asset": best_rows,
    }

    (args.output / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    (args.output / "full_results.json").write_text(
        json.dumps({"overview": overview, "assets": assets}, indent=2), encoding="utf-8"
    )
    write_csv(args.output / "all_results.csv", all_rows)
    write_csv(args.output / "best_target_by_asset.csv", best_rows)
    write_csv(args.output / "target_summary.csv", target_summary)
    write_csv(args.output / "contribution_schedule.csv", schedule_rows)
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
