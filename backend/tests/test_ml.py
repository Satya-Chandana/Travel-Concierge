"""
Tests for ml/ranking.py and ml/scheduler.py
"""
import math
import pytest
import numpy as np


# ── LambdaMART / ranking tests ────────────────────────────────────────────────

from ml.ranking import (
    extract_features, _category_match, _time_suitability,
    LambdaMARTRanker, FeedbackStore, FeedbackEvent, rank_places,
)


class TestCategoryMatch:
    def test_exact_overlap(self):
        score = _category_match(["museum", "art"], "museum history")
        assert score > 0.5

    def test_no_overlap(self):
        score = _category_match(["nightclub", "bar"], "museum history")
        assert score <= 0.5

    def test_empty_interest(self):
        assert _category_match(["museum"], "") == 0.5


class TestTimeSuitability:
    def test_bar_evening(self):
        m, af, ev = _time_suitability(["bar"])
        assert ev > m  # bars suit evenings more than mornings

    def test_museum_morning(self):
        m, af, ev = _time_suitability(["museum"])
        assert m >= 0.9

    def test_unknown_defaults(self):
        m, af, ev = _time_suitability(["unknown_type"])
        assert 0 < m <= 1


class TestExtractFeatures:
    def test_output_shape(self):
        place = {
            "rating": 4.5, "user_ratings_total": 1000,
            "price_level": 2, "lat": 48.86, "lon": 2.35,
            "types": ["museum"],
        }
        feats = extract_features(place, "museum art", 48.86, 2.35, 10.0, 0.7)
        arr = feats.to_array()
        assert arr.shape == (9,)
        assert all(0 <= v <= 1.5 for v in arr)  # all roughly normalised

    def test_rating_normalisation(self):
        place = {"rating": 5.0, "user_ratings_total": 0, "lat": 0, "lon": 0, "types": []}
        feats = extract_features(place, "", 0, 0, 10.0, 0.5)
        assert feats.rating_norm == pytest.approx(1.0)

    def test_close_place_higher_distance_norm(self):
        close = {"rating": 3.0, "user_ratings_total": 0, "lat": 48.860, "lon": 2.350, "types": []}
        far   = {"rating": 3.0, "user_ratings_total": 0, "lat": 49.500, "lon": 3.000, "types": []}
        f_close = extract_features(close, "", 48.86, 2.35, 10.0, 0.5)
        f_far   = extract_features(far,   "", 48.86, 2.35, 10.0, 0.5)
        assert f_close.distance_norm > f_far.distance_norm


class TestHeuristicScore:
    def test_higher_rating_scores_higher(self):
        good = np.array([0.9, 0.7, 0.0, 0.8, 0.9, 0.8, 0.9, 0.6, 0.5], dtype=np.float32)
        bad  = np.array([0.3, 0.2, 0.0, 0.2, 0.3, 0.5, 0.6, 0.4, 0.2], dtype=np.float32)
        assert LambdaMARTRanker._heuristic_score(good) > LambdaMARTRanker._heuristic_score(bad)


class TestRankPlaces:
    def _make_store(self, tmp_path):
        from pathlib import Path
        return FeedbackStore(Path(tmp_path) / "feedback.json")

    def test_returns_sorted_by_score(self, tmp_path):
        store = self._make_store(tmp_path)
        ranker = LambdaMARTRanker.__new__(LambdaMARTRanker)
        ranker._model = None
        ranker.model_path = None

        places = [
            {"name": "A", "rating": 2.0, "user_ratings_total": 10, "lat": 0, "lon": 0, "types": ["museum"]},
            {"name": "B", "rating": 4.9, "user_ratings_total": 5000, "lat": 0, "lon": 0, "types": ["museum"]},
        ]
        ranked = rank_places(places, "sess1", "museum", 0, 0, store, ranker)
        assert ranked[0]["name"] == "B"

    def test_ltr_score_field_added(self, tmp_path):
        store = self._make_store(tmp_path)
        ranker = LambdaMARTRanker.__new__(LambdaMARTRanker)
        ranker._model = None
        ranker.model_path = None

        places = [{"name": "X", "rating": 3.0, "user_ratings_total": 100, "lat": 0, "lon": 0, "types": []}]
        ranked = rank_places(places, "sess1", "", 0, 0, store, ranker)
        assert "_ltr_score" in ranked[0]


# ── DP Scheduler tests ────────────────────────────────────────────────────────

from ml.scheduler import (
    schedule_day, schedule_multi_day, VisitSlot,
    _minutes_to_str, _travel_time_between, _compute_p,
    WAKE_TIME, END_TIME,
)


class TestMinutesToStr:
    def test_noon(self):
        assert _minutes_to_str(12 * 60) == "12:00"

    def test_midnight(self):
        assert _minutes_to_str(0) == "00:00"

    def test_nine_thirty(self):
        assert _minutes_to_str(9 * 60 + 30) == "09:30"


class TestTravelTime:
    def test_same_point(self):
        t = _travel_time_between(0, 0, 0, 0)
        assert t == 5  # just the buffer

    def test_farther_takes_longer(self):
        near = _travel_time_between(48.86, 2.35, 48.87, 2.36)
        far  = _travel_time_between(48.86, 2.35, 49.50, 3.00)
        assert far > near


class TestScheduleDay:
    def _make_candidates(self, n=5):
        return [
            {
                "name": f"Place {i}",
                "address": f"Addr {i}",
                "lat": 48.86 + i * 0.01,
                "lon": 2.35 + i * 0.01,
                "types": ["museum"],
                "open_time": 9 * 60,
                "close_time": 20 * 60,
                "visit_duration": 90,
                "_ltr_score": 0.8 - i * 0.05,
            }
            for i in range(n)
        ]

    def test_visits_within_time_window(self):
        candidates = self._make_candidates(5)
        day = schedule_day(candidates, "Test Day")
        for v in day.visits:
            assert v.arrival_time >= WAKE_TIME
            assert v.departure_time <= END_TIME + 60  # slight buffer

    def test_no_overlapping_visits(self):
        candidates = self._make_candidates(6)
        day = schedule_day(candidates, "Test Day")
        times = sorted(day.visits, key=lambda v: v.arrival_time)
        for i in range(len(times) - 1):
            assert times[i].departure_time <= times[i+1].arrival_time

    def test_empty_candidates(self):
        day = schedule_day([], "Empty Day")
        assert day.total_stops == 0

    def test_format_contains_place_name(self):
        candidates = self._make_candidates(2)
        day = schedule_day(candidates, "Day 1 — Paris")
        formatted = day.format()
        assert "Place 0" in formatted or "Place 1" in formatted
        assert "Day 1 — Paris" in formatted

    def test_meal_breaks_inserted(self):
        # Long day with candidates spanning full day
        candidates = [
            {
                "name": f"Spot {i}", "address": "", "lat": 48.86, "lon": 2.35,
                "types": ["landmark"], "open_time": 8 * 60, "close_time": 22 * 60,
                "visit_duration": 60, "_ltr_score": 0.9,
            }
            for i in range(10)
        ]
        day = schedule_day(candidates, "Full Day")
        meal_types = [m.meal_type for m in day.meals]
        # At least one meal should be inserted for a full day
        assert len(day.meals) >= 1


class TestScheduleMultiDay:
    def test_correct_number_of_days(self):
        candidates = [
            {"name": f"P{i}", "address": "", "lat": 0, "lon": 0,
             "types": [], "_ltr_score": 0.5, "open_time": 480, "close_time": 1200, "visit_duration": 60}
            for i in range(12)
        ]
        days = schedule_multi_day(candidates, 3, city="Paris")
        assert len(days) == 3

    def test_day_labels_contain_city(self):
        candidates = [
            {"name": "X", "address": "", "lat": 0, "lon": 0,
             "types": [], "_ltr_score": 0.5, "open_time": 480, "close_time": 1200, "visit_duration": 60}
        ]
        days = schedule_multi_day(candidates, 2, city="Tokyo")
        assert all("Tokyo" in d.date_label for d in days)

    def test_empty_candidates(self):
        days = schedule_multi_day([], 3)
        assert days == []
