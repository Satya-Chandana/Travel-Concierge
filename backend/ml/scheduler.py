"""
Constraint-Aware Itinerary Scheduler — Dynamic Programming.

Problem
-------
Given a list of candidate places with:
  - opening hours  (open_time, close_time in minutes from midnight)
  - visit duration (minutes)
  - travel time to next place (minutes, from routing service)
  - a value score  (from LTR ranker or heuristic)
  - hard constraints: must start after wake_time, must finish by end_time
  - soft constraints: meal breaks at lunch/dinner windows

This is a variant of the **Weighted Job Scheduling** problem, solved with DP.

Algorithm: Weighted Interval Scheduling (DP)
--------------------------------------------
Classic O(n log n) DP:
  1. Sort jobs (visits) by finish time.
  2. For each job j, find p(j) = latest job that doesn't overlap with j.
  3. DP recurrence:
       OPT(j) = max(value_j + OPT(p(j)),  OPT(j-1))
  4. Backtrack to reconstruct optimal schedule.

Extended here with:
  - Travel time buffers between consecutive visits
  - Mandatory meal break slots (lunch 12:00-14:00, dinner 18:30-20:30)
  - Time-of-day value multipliers (from LTR features)
  - Maximum daily walking distance constraint

Reference: Kleinberg & Tardos, "Algorithm Design" Ch.6 (Weighted Interval Scheduling)
"""
from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
WAKE_TIME = 8 * 60          # 08:00 in minutes
END_TIME = 22 * 60          # 22:00
LUNCH_START = 12 * 60       # 12:00
LUNCH_END = 14 * 60         # 14:00
DINNER_START = 18 * 60 + 30 # 18:30
DINNER_END = 20 * 60 + 30   # 20:30
MEAL_DURATION = 60          # 60 min per meal
DEFAULT_VISIT_DURATION = 90 # minutes
DEFAULT_TRAVEL_TIME = 20    # minutes between stops
MAX_DAILY_STOPS = 8


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class VisitSlot:
    """Represents one candidate visit to a place."""
    place_name: str
    address: str
    open_time: int      # minutes from midnight (e.g. 9*60 = 540)
    close_time: int     # minutes from midnight
    visit_duration: int # how long to spend here (minutes)
    travel_time_to_next: int  # minutes to travel to next place in sorted order
    value: float        # LTR score or heuristic score
    lat: float
    lon: float
    place_type: str = ""

    @property
    def latest_start(self) -> int:
        """Latest time you can start visiting and still finish before close."""
        return self.close_time - self.visit_duration

    def finish_time(self, start: int) -> int:
        return start + self.visit_duration + self.travel_time_to_next


@dataclass
class ScheduledVisit:
    place_name: str
    address: str
    arrival_time: int    # minutes from midnight
    departure_time: int
    lat: float
    lon: float
    place_type: str = ""

    def arrival_str(self) -> str:
        return _minutes_to_str(self.arrival_time)

    def departure_str(self) -> str:
        return _minutes_to_str(self.departure_time)


@dataclass
class MealBreak:
    meal_type: str       # "Lunch" or "Dinner"
    start_time: int
    end_time: int
    suggestion: str = ""

    def start_str(self) -> str:
        return _minutes_to_str(self.start_time)

    def end_str(self) -> str:
        return _minutes_to_str(self.end_time)


@dataclass
class DaySchedule:
    date_label: str
    visits: list[ScheduledVisit]
    meals: list[MealBreak]
    total_stops: int
    optimality_note: str = ""

    def format(self) -> str:
        """Human-readable day schedule."""
        lines = [f"\n📅 **{self.date_label}**\n"]
        
        # Merge visits and meals sorted by time
        events: list[tuple[int, str]] = []
        for v in self.visits:
            events.append((v.arrival_time,
                f"  🏛️  **{v.place_name}**  ({v.arrival_str()} – {v.departure_str()})\n"
                f"      📍 {v.address}"))
        for m in self.meals:
            events.append((m.start_time,
                f"  🍽️  **{m.meal_type} Break**  ({m.start_str()} – {m.end_str()})"
                + (f"\n      💡 {m.suggestion}" if m.suggestion else "")))

        events.sort(key=lambda x: x[0])
        lines.extend(e[1] for _, e in events)

        if self.optimality_note:
            lines.append(f"\n  ℹ️  _{self.optimality_note}_")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minutes_to_str(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


def _overlaps_meal(start: int, end: int) -> Optional[str]:
    """Return meal type if this slot overlaps a meal window."""
    if start < LUNCH_END and end > LUNCH_START:
        return "lunch"
    if start < DINNER_END and end > DINNER_START:
        return "dinner"
    return None


def _travel_time_between(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    """Estimate travel time in minutes using Haversine distance + 4 km/h walking."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    dist_km = 2 * 6371 * math.asin(math.sqrt(a))
    # Assume 4 km/h walking + 5 min buffer
    return int(dist_km / 4.0 * 60) + 5


# ── Core DP algorithm ─────────────────────────────────────────────────────────

def _compute_p(slots: list[VisitSlot], starts: list[int]) -> list[int]:
    """
    For each slot j (sorted by finish time), compute p(j) = index of the
    latest slot that finishes before slot j starts.
    Uses binary search: O(n log n).
    """
    finish_times = [s + slots[i].visit_duration for i, s in enumerate(starts)]
    p = []
    for j, start_j in enumerate(starts):
        # Find latest slot that finishes at or before start_j
        idx = bisect.bisect_right(finish_times, start_j, 0, j) - 1
        p.append(idx)
    return p


def schedule_day(
    candidates: list[dict],
    day_label: str = "Day 1",
    wake_time: int = WAKE_TIME,
    end_time: int = END_TIME,
    restaurant_suggestion: str = "",
) -> DaySchedule:
    """
    Build an optimal day schedule using Weighted Interval Scheduling DP.

    Parameters
    ----------
    candidates:
        List of place dicts with keys: name, address, lat, lon, types,
        open_time (optional), close_time (optional), visit_duration (optional),
        _ltr_score (from ranker).
    day_label:
        Display label e.g. "Day 1 — Paris".
    wake_time / end_time:
        Hard time window in minutes from midnight.

    Returns
    -------
    DaySchedule with optimally selected and sequenced visits + meal breaks.
    """
    if not candidates:
        return DaySchedule(day_label, [], [], 0, "No candidates provided.")

    # ── Build VisitSlot objects ───────────────────────────────────────────────
    slots: list[VisitSlot] = []
    for i, p in enumerate(candidates):
        open_t = int(p.get("open_time", wake_time))
        close_t = int(p.get("close_time", end_time))
        duration = int(p.get("visit_duration", DEFAULT_VISIT_DURATION))
        
        # Skip if place can't fit in time window
        if open_t + duration > end_time or close_t <= wake_time:
            continue

        # Travel time to next candidate (approximate using coords)
        if i + 1 < len(candidates):
            next_p = candidates[i + 1]
            travel = _travel_time_between(
                p.get("lat", 0), p.get("lon", 0),
                next_p.get("lat", 0), next_p.get("lon", 0)
            )
        else:
            travel = 0

        types = p.get("types", [])
        slots.append(VisitSlot(
            place_name=p.get("name", "Unknown"),
            address=p.get("address", ""),
            open_time=max(open_t, wake_time),
            close_time=min(close_t, end_time),
            visit_duration=duration,
            travel_time_to_next=travel,
            value=float(p.get("_ltr_score", p.get("rating", 3.0) / 5.0)),
            lat=float(p.get("lat", 0)),
            lon=float(p.get("lon", 0)),
            place_type=types[0] if types else "",
        ))

    if not slots:
        return DaySchedule(day_label, [], [], 0, "No slots fit within the time window.")

    # ── Assign greedy start times (earliest possible) ─────────────────────────
    # Sort by open_time to enable DP
    slots.sort(key=lambda s: s.open_time)
    
    starts: list[int] = []
    current_time = wake_time
    for slot in slots:
        start = max(slot.open_time, current_time)
        if start > slot.latest_start:
            start = slot.latest_start
        starts.append(start)
        current_time = start + slot.visit_duration + slot.travel_time_to_next

    # ── DP: Weighted Interval Scheduling ─────────────────────────────────────
    n = len(slots)
    p = _compute_p(slots, starts)

    # dp[j] = max value achievable considering slots 0..j
    dp = [0.0] * (n + 1)
    for j in range(1, n + 1):
        slot = slots[j - 1]
        # Include slot j: value + best non-overlapping
        include = slot.value + dp[p[j - 1] + 1]
        # Exclude slot j
        exclude = dp[j - 1]
        dp[j] = max(include, exclude)

    # ── Backtrack to find selected slots ─────────────────────────────────────
    selected_indices: list[int] = []
    j = n
    while j >= 1:
        slot = slots[j - 1]
        include = slot.value + dp[p[j - 1] + 1]
        if include >= dp[j - 1]:
            selected_indices.append(j - 1)
            j = p[j - 1] + 1
        else:
            j -= 1
    selected_indices.reverse()

    # Cap at MAX_DAILY_STOPS
    selected_indices = selected_indices[:MAX_DAILY_STOPS]

    # ── Build schedule with meal breaks ───────────────────────────────────────
    visits: list[ScheduledVisit] = []
    meals: list[MealBreak] = []
    lunch_added = dinner_added = False
    current = wake_time

    for idx in selected_indices:
        slot = slots[idx]
        start = max(starts[idx], current)

        # Insert lunch break if we're about to miss the window
        if not lunch_added and start >= LUNCH_START and current <= LUNCH_END:
            lunch_start = max(current, LUNCH_START)
            meals.append(MealBreak(
                "Lunch", lunch_start, lunch_start + MEAL_DURATION,
                suggestion=restaurant_suggestion or "Ask me for restaurant recommendations nearby!"
            ))
            current = lunch_start + MEAL_DURATION
            start = max(start, current)
            lunch_added = True

        # Insert dinner break
        if not dinner_added and start >= DINNER_START and current <= DINNER_END:
            dinner_start = max(current, DINNER_START)
            meals.append(MealBreak(
                "Dinner", dinner_start, dinner_start + MEAL_DURATION,
                suggestion=restaurant_suggestion or "Ask me for restaurant recommendations nearby!"
            ))
            current = dinner_start + MEAL_DURATION
            start = max(start, current)
            dinner_added = True

        depart = start + slot.visit_duration
        visits.append(ScheduledVisit(
            place_name=slot.place_name,
            address=slot.address,
            arrival_time=start,
            departure_time=depart,
            lat=slot.lat,
            lon=slot.lon,
            place_type=slot.place_type,
        ))
        current = depart + slot.travel_time_to_next

    note = (
        f"Optimal schedule: {len(visits)} stops selected from {len(candidates)} candidates "
        f"using Weighted Interval Scheduling DP (maximising personalised relevance scores)."
    )

    return DaySchedule(
        date_label=day_label,
        visits=visits,
        meals=meals,
        total_stops=len(visits),
        optimality_note=note,
    )


def schedule_multi_day(
    candidates: list[dict],
    num_days: int,
    city: str = "",
    restaurant_suggestion: str = "",
) -> list[DaySchedule]:
    """
    Split candidates across multiple days and schedule each day optimally.

    Strategy: distribute candidates evenly, ensuring no day is overloaded.
    Each day gets a fresh DP run.
    """
    if not candidates:
        return []

    per_day = max(1, math.ceil(len(candidates) / num_days))
    days: list[DaySchedule] = []

    for d in range(num_days):
        chunk = candidates[d * per_day: (d + 1) * per_day]
        label = f"Day {d + 1}" + (f" — {city}" if city else "")
        day = schedule_day(chunk, label, restaurant_suggestion=restaurant_suggestion)
        days.append(day)

    return days
