"""Deterministic Buy or Wait decision engine.

Run from the repository root with: python code/main.py
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path

from agentic import AgenticAdvisor


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUTPUT_COLUMNS = [
	"request_id", "amount_safe_to_pay", "affordability_status",
	"recommended_payment_method", "payment_plan",
	"earliest_date_for_full_payment", "spending_changes_needed",
	"decision_explanation",
]
CENT = Decimal("0.01")


def read_csv(name: str) -> list[dict[str, str]]:
	with (DATASET / name).open(encoding="utf-8-sig", newline="") as handle:
		return list(csv.DictReader(handle))


def money(value: str | Decimal | float | int) -> Decimal:
	return Decimal(str(value or "0")).quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(value: Decimal) -> str:
	value = money(value)
	return format(value, "f").rstrip("0").rstrip(".") or "0"


def parse_date(value: str) -> date:
	return date.fromisoformat(value)


def split_values(value: str) -> set[str]:
	return {part.strip() for part in (value or "").split("|") if part.strip()}


def extract_amount(text: str) -> Decimal | None:
	matches = re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", text or "")
	if not matches:
		return None
	return money(matches[-1].replace(",", ""))


def build_rates(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], Decimal]:
	rates = {}
	for row in rows:
		rates[(row["rate_date"], row["from_currency"], row["to_currency"])] = Decimal(row["rate"])
	return rates


def convert(value: Decimal, currency: str, home: str, event_date: str, rates: dict) -> Decimal:
	if currency == home:
		return value
	direct = rates.get((event_date, currency, home))
	inverse = rates.get((event_date, home, currency))
	if direct is not None:
		return money(value * direct)
	if inverse is not None and inverse:
		return money(value / inverse)
	return value


def add_months(day: date, months: int = 1) -> date:
	month = day.month - 1 + months
	year, month = day.year + month // 12, month % 12 + 1
	import calendar
	return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def event_amount(row: dict[str, str], image_text: dict[str, str]) -> Decimal | None:
	if row["amount"].strip():
		return money(row["amount"])
	return extract_amount(image_text.get(row["event_id"], ""))


def event_delta(row: dict[str, str], amount: Decimal, home: str, rates: dict) -> Decimal:
	value = convert(amount, row["currency"], home, row["settlement_date"], rates)
	return value if row["direction"] == "credit" else -value


def recurring_events(events: list[dict[str, str]], home: str, rates: dict, image_text: dict[str, str], cutoff: date) -> list[dict]:
	grouped = defaultdict(list)
	for row in events:
		amount = event_amount(row, image_text)
		if amount is not None and row["status"] == "settled" and parse_date(row["event_date"]) <= cutoff:
			series = row["description"] if row["direction"] == "credit" else row["category"]
			grouped[(series, row["direction"], row["currency"])].append(row)
	result = []
	for rows in grouped.values():
		rows.sort(key=lambda row: row["event_date"])
		if len(rows) < 3:
			continue
		gaps = [(parse_date(b["event_date"]) - parse_date(a["event_date"])).days for a, b in zip(rows, rows[1:])]
		recent_gaps = gaps[-min(4, len(gaps)):]
		if not recent_gaps or not all(20 <= gap <= 45 for gap in recent_gaps) or max(recent_gaps) - min(recent_gaps) > 5:
			continue
		latest = rows[-1]
		recent_rows = rows[-min(6, len(rows)):]
		recent_amounts = [
			convert(event_amount(row, image_text), row["currency"], home, row["settlement_date"], rates)
			for row in recent_rows
		]
		amount = sum(recent_amounts, Decimal(0)) / len(recent_amounts)
		average_gap = round(sum(gaps[-3:]) / min(3, len(gaps)))
		result.append({
			"row": latest,
			"amount": money(amount),
			"interval": average_gap,
			"monthly": 25 <= average_gap <= 35,
		})
	return result


def future_cashflows(user_events: list[dict[str, str]], profile: dict[str, str], start: date, end: date, rates: dict, image_text: dict[str, str], changes: dict[str, Decimal] | None = None) -> dict[date, Decimal]:
	home = profile["home_currency"]
	flows = defaultdict(Decimal)
	changes = changes or {}
	for row in user_events:
		amount = event_amount(row, image_text)
		if amount is None or row["status"] in {"cancelled", "failed", "unrealized"}:
			continue
		when = parse_date(row["settlement_date"] or row["event_date"])
		if start <= when <= end and row["status"] in {"pending", "scheduled"}:
			changed = changes.get(row["event_id"], amount)
			flows[when] += event_delta(row, changed, home, rates)
	for item in recurring_events(user_events, home, rates, image_text, start):
		row, amount, interval = item["row"], item["amount"], item["interval"]
		current = parse_date(row["event_date"])
		while current <= end:
			if current >= start:
				adjusted = changes.get(row["event_id"], amount)
				flows[current] += adjusted if row["direction"] == "credit" else -adjusted
			current = add_months(current) if item["monthly"] else current + timedelta(days=interval)
	return flows


def balance_on_dates(profile: dict[str, str], flows: dict[date, Decimal], start: date, end: date, payments: dict[date, Decimal]) -> dict[date, Decimal]:
	balance = money(profile["current_available_balance"])
	minimum = money(profile["minimum_balance_to_keep"])
	result = {}
	for offset in range((end - start).days + 1):
		day = start + timedelta(days=offset)
		balance += flows.get(day, Decimal(0)) - payments.get(day, Decimal(0))
		result[day] = balance
	return result


def safe_payment_amount(profile: dict[str, str], flows: dict[date, Decimal], start: date, end: date, target: Decimal, payments: dict[date, Decimal] | None = None) -> Decimal:
	payments = payments or {}
	baseline = balance_on_dates(profile, flows, start, end, payments)
	headroom = min((value - money(profile["minimum_balance_to_keep"]) for value in baseline.values()), default=Decimal(0))
	return max(Decimal(0), min(target, headroom))


def option_schedule(option: dict[str, str], request_date: date, deadline: date) -> dict[date, Decimal] | None:
	first = parse_date(option["first_payment_date"])
	count = int(option["number_of_payments"])
	frequency = int(option["payment_frequency_days"] or 0)
	schedule = {}
	for index in range(count):
		day = first + timedelta(days=frequency * index)
		if day > deadline:
			return None
		schedule[day] = money(option["payment_amount"])
	return schedule


def plan_text(schedule: dict[date, Decimal]) -> str:
	return "|".join(f"{day.isoformat()}:{fmt(amount)}" for day, amount in sorted(schedule.items()))


def decide(request: dict[str, str], profile: dict[str, str], user_events: list[dict[str, str]], options: list[dict[str, str]], rates: dict, image_text: dict[str, str]) -> dict[str, str]:
	start = parse_date(request["request_date"])
	deadline = parse_date(request["desired_completion_date"])
	end = start + timedelta(days=90)
	target = money(request["requested_amount"])
	methods = split_values(profile["payment_methods_user_will_consider"])
	flows = future_cashflows(user_events, profile, start, end, rates, image_text)
	safe_today = safe_payment_amount(profile, flows, start, end, target)
	change_options = []
	stop_categories = split_values(profile["expense_categories_user_is_willing_to_stop"])
	reduce_categories = split_values(profile["expense_categories_user_is_willing_to_reduce"])
	for item in recurring_events(user_events, profile["home_currency"], rates, image_text, start):
		row = item["row"]
		if row["category"] in stop_categories and row["flexibility"] in {"stoppable", "reducible_or_stoppable"}:
			change_options.append((row["event_id"], Decimal(0), f"stop:{row['event_id']}"))
		elif row["category"] in reduce_categories and row["flexibility"] in {"reducible", "reducible_or_stoppable"}:
			minimum = money(row["minimum_allowed_amount"] or "0")
			change_options.append((row["event_id"], minimum, f"reduce_to:{row['event_id']}:{fmt(minimum)}"))
	earliest = None
	for offset in range((end - start).days + 1):
		day = start + timedelta(days=offset)
		future = future_cashflows(user_events, profile, day, end, rates, image_text)
		if safe_payment_amount(profile, future, day, end, target) >= target:
			earliest = day
			break

	def build_candidates(plan_flows: dict[date, Decimal]) -> list[tuple]:
		candidates = []
		if "full_payment" in methods:
			full = {start: target}
			if all(value >= money(profile["minimum_balance_to_keep"]) for value in balance_on_dates(profile, plan_flows, start, end, full).values()):
				candidates.append((0, 0, start, "full_payment", full, None))
		if "partial_payment" in methods and request["allows_partial_payment"].lower() == "true" and 0 < safe_today < target and earliest and earliest <= deadline:
			partial = {start: safe_today, earliest: target - safe_today}
			if all(value >= money(profile["minimum_balance_to_keep"]) for value in balance_on_dates(profile, plan_flows, start, end, partial).values()):
				candidates.append((1, 0, start, "partial_payment", partial, None))
		for option in options:
			if option["payment_method"] != "installments" or "installments" not in methods:
				continue
			schedule = option_schedule(option, start, deadline)
			if schedule and sum(schedule.values()) >= target and all(value >= money(profile["minimum_balance_to_keep"]) for value in balance_on_dates(profile, plan_flows, start, end, schedule).values()):
				candidates.append((2, money(option["total_payable_amount"]), min(schedule), "installments", schedule, option))
		return candidates

	selected_changes = None
	selected_change_text = "none"
	plan_flows = flows
	candidates = build_candidates(plan_flows)
	if not candidates and earliest and earliest <= deadline and "full_payment" in methods:
		return make_output(request, safe_today, "affordable_later", "wait", {earliest: target}, earliest, "none", profile)
	if not candidates:
		viable_changes = None
		for size in range(1, min(3, len(change_options)) + 1):
			for selected in combinations(change_options, size):
				changes = {event_id: amount for event_id, amount, _ in selected}
				changed_flows = future_cashflows(user_events, profile, start, end, rates, image_text, changes)
				full = {start: target}
				if all(value >= money(profile["minimum_balance_to_keep"]) for value in balance_on_dates(profile, changed_flows, start, end, full).values()):
					viable_changes = (changes, "|".join(item[2] for item in selected))
					break
			if viable_changes:
				break
		if viable_changes:
			selected_changes, selected_change_text = viable_changes
			plan_flows = future_cashflows(user_events, profile, start, end, rates, image_text, selected_changes)
			candidates = build_candidates(plan_flows)
	if candidates:
		candidates.sort(key=lambda item: (item[0], item[1], item[2], item[5]["payment_option_id"] if item[5] else ""))
		chosen = candidates[0]
		status = "affordable_now" if chosen[3] == "full_payment" else "affordable_with_plan"
		return make_output(request, safe_today, status, chosen[3], chosen[4], earliest or start, selected_change_text, profile)
	if earliest and earliest <= deadline and "full_payment" in methods:
		return make_output(request, safe_today, "affordable_later", "wait", {earliest: target}, earliest, "none", profile)
	return make_output(request, safe_today, "not_affordable", "not_recommended", {}, None, "none", profile)


def make_output(request, safe_today, status, method, schedule, earliest, changes, profile):
	currency = profile["home_currency"]
	if method == "full_payment": explanation = f"Pay {currency} {fmt(money(request['requested_amount']))} today while keeping the {currency} {fmt(money(profile['minimum_balance_to_keep']))} minimum protected."
	elif method == "installments": explanation = f"Use the supplied installment schedule; it keeps the {currency} {fmt(money(profile['minimum_balance_to_keep']))} minimum protected."
	elif method == "partial_payment": explanation = f"Pay {currency} {fmt(safe_today)} today and the balance later while keeping the minimum protected."
	elif method == "wait": explanation = f"Wait until {earliest.isoformat()}, when the full {currency} {fmt(money(request['requested_amount']))} is forecast safe."
	else: explanation = f"Do not proceed: no eligible plan keeps the {currency} {fmt(money(profile['minimum_balance_to_keep']))} minimum protected by the deadline."
	return {"request_id": request["request_id"], "amount_safe_to_pay": fmt(safe_today), "affordability_status": status, "recommended_payment_method": method, "payment_plan": plan_text(schedule) if schedule else "none", "earliest_date_for_full_payment": earliest.isoformat() if earliest else "", "spending_changes_needed": changes, "decision_explanation": explanation}


def validate(rows: list[dict[str, str]], requests: list[dict[str, str]]) -> None:
	assert len(rows) == len(requests)
	assert [row.keys() for row in rows] and list(rows[0]) == OUTPUT_COLUMNS
	for row, request in zip(rows, requests):
		amount = money(row["amount_safe_to_pay"])
		target = money(request["requested_amount"])
		assert Decimal(0) <= amount <= target
		assert row["affordability_status"] in {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}


def main() -> None:
	profiles = {row["user_id"]: row for row in read_csv("financial_profiles.csv")}
	requests = read_csv("requests.csv")
	events_by_user = defaultdict(list)
	for row in read_csv("financial_events.csv"):
		events_by_user[row["user_id"]].append(row)
	options_by_request = defaultdict(list)
	for row in read_csv("request_payment_options.csv"):
		options_by_request[row["request_id"]].append(row)
	rates = build_rates(read_csv("exchange_rates.csv"))
	messages = read_csv("messages.csv")
	advisor = AgenticAdvisor()
	image_text = {
		row["related_event_id"]: row["amount"]
		for row in read_csv("image_amounts.csv")
		if row["related_event_id"]
	}
	rows = []
	for request in requests:
		profile = profiles[request["user_id"]]
		baseline = decide(request, profile, events_by_user[request["user_id"]], options_by_request[request["request_id"]], rates, image_text)
		relevant_evidence = [
			message for message in messages
			if message["user_id"] == request["user_id"]
			and (not message["request_id"] or message["request_id"] == request["request_id"])
		][:10]
		rows.append(advisor.advise(request, profile, baseline, relevant_evidence))
	validate(rows, requests)
	with (ROOT / "output.csv").open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
		writer.writeheader()
		writer.writerows(rows)
	advisor.write_usage_report(str(ROOT / "code" / "evaluation" / "usage_report.md"), len(requests))
	print(f"Wrote {len(rows)} predictions to {ROOT / 'output.csv'}")


if __name__ == "__main__":
	main()
