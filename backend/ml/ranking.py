"""
Personalised Place Ranking — Learning-to-Rank with LambdaMART.

How it works
------------
1. Every time a user accepts or skips a place suggestion, we record that as
   implicit feedback (accepted=1, skipped=0).
2. We extract a feature vector for each place:
     - rating, review_count, price_level
     - category match score (how well it matches user interests)
     - distance from trip centre (normalised)
     - time_of_day suitability (morning/afternoon/evening)
     - historical acceptance rate for this category by this user
3. A LambdaMART model (via XGBoost's rank:ndcg objective) is trained on
   accumulated feedback and used to re-rank future suggestions.
4. Cold start: before enough feedback exists, we fall back to a
   heuristic score (rating * log(reviews) * category_match).

Algorithm: LambdaMART
---------------------
LambdaMART optimises NDCG (Normalised Discounted Cumulative Gain) directly.
It builds gradient-boosted trees where the gradient of each tree is the
"lambda" — a pairwise preference signal derived from all (accepted, skipped)
pairs. This is the same algorithm used by Bing, Yahoo, and most production
search/recommendation systems.

Reference: Burges et al. (2010) "From RankNet to LambdaRank to LambdaMART"
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Minimum feedback events before we switch from heuristic to LambdaMART ───
MIN_FEEDBACK_FOR_MODEL = 10
FEEDBACK_STORE_PATH = Path("data/feedback_store.json")
MODEL_STORE_PATH = Path("data/ltr_model.pkl")


# ── Feature extraction ────────────────────────────────────────────────────────

@dataclass
class PlaceFeatures:
    """
    Fixed-length feature vector for one place.
    All values normalised to [0, 1] range.
    """
    rating_norm: float           # rating / 5.0
    log_reviews_norm: float      # log(reviews+1) / log(10001)
    price_level_norm: float      # price_level / 4.0  (0 if unknown)
    category_match: float        # cosine-like overlap with user interests [0,1]
    distance_norm: float         # 1 - (dist / max_dist), higher = closer
    morning_suit: float          # 1 if category suits morning, else 0.5/0
    afternoon_suit: float
    evening_suit: float
    hist_accept_rate: float      # user's historical accept rate for this category

    def to_array(self) -> np.ndarray:
        return np.array([
            self.rating_norm, self.log_reviews_norm, self.price_level_norm,
            self.category_match, self.distance_norm,
            self.morning_suit, self.afternoon_suit, self.evening_suit,
            self.hist_accept_rate,
        ], dtype=np.float32)


# Time-of-day suitability table per broad category keyword
_TIME_SUIT: dict[str, tuple[float, float, float]] = {
    # (morning, afternoon, evening)
    "museum":       (1.0, 1.0, 0.3),
    "park":         (1.0, 0.8, 0.5),
    "beach":        (0.8, 1.0, 0.7),
    "restaurant":   (0.5, 1.0, 1.0),
    "bar":          (0.0, 0.3, 1.0),
    "nightclub":    (0.0, 0.1, 1.0),
    "temple":       (1.0, 0.8, 0.5),
    "church":       (1.0, 0.8, 0.4),
    "market":       (1.0, 0.9, 0.6),
    "shopping":     (0.5, 1.0, 0.8),
    "landmark":     (0.9, 1.0, 0.7),
    "gallery":      (0.9, 1.0, 0.5),
    "cafe":         (1.0, 0.8, 0.5),
}

_DEFAULT_TIME = (0.7, 0.8, 0.6)


def _time_suitability(category_keywords: list[str]) -> tuple[float, float, float]:
    for kw in category_keywords:
        kw_lower = kw.lower()
        for key, vals in _TIME_SUIT.items():
            if key in kw_lower:
                return vals
    return _DEFAULT_TIME


def _category_match(place_types: list[str], user_interests: str) -> float:
    """Simple token overlap score between place types and user interest string."""
    if not user_interests:
        return 0.5
    interest_tokens = set(user_interests.lower().split())
    type_tokens = set(" ".join(place_types).lower().replace("_", " ").split())
    if not type_tokens:
        return 0.5
    overlap = len(interest_tokens & type_tokens)
    return min(1.0, overlap / max(len(interest_tokens), 1) + 0.2)


def extract_features(
    place: dict,
    user_interests: str,
    centre_lat: float,
    centre_lon: float,
    max_dist_km: float,
    hist_accept_rate: float,
) -> PlaceFeatures:
    import math as _math

    rating = float(place.get("rating", 3.0))
    reviews = int(place.get("user_ratings_total", 0))
    price = int(place.get("price_level") or 0)
    lat = float(place.get("lat", centre_lat))
    lon = float(place.get("lon", centre_lon))
    types = place.get("types", [])

    # Haversine distance to trip centre
    dlat = math.radians(lat - centre_lat)
    dlon = math.radians(lon - centre_lon)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(centre_lat)) * math.cos(math.radians(lat)) * math.sin(dlon/2)**2
    dist_km = 2 * 6371 * math.asin(math.sqrt(a))
    dist_norm = max(0.0, 1.0 - dist_km / max(max_dist_km, 1.0))

    m, af, ev = _time_suitability(types)

    return PlaceFeatures(
        rating_norm=rating / 5.0,
        log_reviews_norm=math.log(reviews + 1) / math.log(10001),
        price_level_norm=price / 4.0,
        category_match=_category_match(types, user_interests),
        distance_norm=dist_norm,
        morning_suit=m,
        afternoon_suit=af,
        evening_suit=ev,
        hist_accept_rate=hist_accept_rate,
    )


# ── Feedback store ────────────────────────────────────────────────────────────

@dataclass
class FeedbackEvent:
    session_id: str
    place_name: str
    place_types: list[str]
    features: list[float]   # serialised PlaceFeatures array
    accepted: int           # 1 = accepted, 0 = skipped


class FeedbackStore:
    """Persists feedback events as a JSON file."""

    def __init__(self, path: Path = FEEDBACK_STORE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[FeedbackEvent] = self._load()

    def _load(self) -> list[FeedbackEvent]:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                return [FeedbackEvent(**e) for e in raw]
            except Exception as exc:
                log.warning("Could not load feedback store: %s", exc)
        return []

    def _save(self) -> None:
        self.path.write_text(json.dumps([asdict(e) for e in self._events], indent=2))

    def record(self, event: FeedbackEvent) -> None:
        self._events.append(event)
        self._save()

    @property
    def events(self) -> list[FeedbackEvent]:
        return list(self._events)

    def category_accept_rate(self, session_id: str, category: str) -> float:
        """Historical accept rate for a category in this session."""
        relevant = [
            e for e in self._events
            if e.session_id == session_id and any(category in t for t in e.place_types)
        ]
        if not relevant:
            return 0.5
        return sum(e.accepted for e in relevant) / len(relevant)


# ── LambdaMART ranker ─────────────────────────────────────────────────────────

class LambdaMARTRanker:
    """
    Thin wrapper around XGBoost's rank:ndcg objective (LambdaMART).

    Training
    --------
    Requires at least MIN_FEEDBACK_FOR_MODEL events.
    Each query group = one session_id.
    Labels: accepted=3 (highly relevant), skipped=0 (not relevant).

    Inference
    ---------
    Returns a score per place; higher = more preferred.
    Falls back to heuristic score if model not trained yet.
    """

    def __init__(self, model_path: Path = MODEL_STORE_PATH):
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                log.warning("Could not load LTR model: %s", exc)
        return None

    def _save_model(self, model) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

    def train(self, store: FeedbackStore) -> bool:
        """
        Train LambdaMART on accumulated feedback.
        Returns True if training succeeded.
        """
        events = store.events
        if len(events) < MIN_FEEDBACK_FOR_MODEL:
            log.info("Not enough feedback to train (%d/%d)", len(events), MIN_FEEDBACK_FOR_MODEL)
            return False

        try:
            import xgboost as xgb
        except ImportError:
            log.warning("xgboost not installed — LTR training skipped.")
            return False

        # Build training arrays
        X = np.array([e.features for e in events], dtype=np.float32)
        # Map accepted → relevance label (3=highly relevant, 0=not relevant)
        y = np.array([3 if e.accepted else 0 for e in events], dtype=np.int32)

        # Group by session (query group for LambdaMART)
        sessions = [e.session_id for e in events]
        unique_sessions = list(dict.fromkeys(sessions))
        groups = [sessions.count(s) for s in unique_sessions]

        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(groups)

        params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@5",
            "eta": 0.1,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 0.8,
            "n_estimators": 100,
            "verbosity": 0,
        }

        model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        self._model = model
        self._save_model(model)
        log.info("LambdaMART model trained on %d events.", len(events))
        return True

    def score(self, features: np.ndarray) -> float:
        """Return a relevance score for a single place feature vector."""
        if self._model is None:
            return self._heuristic_score(features)
        try:
            import xgboost as xgb
            dm = xgb.DMatrix(features.reshape(1, -1))
            return float(self._model.predict(dm)[0])
        except Exception as exc:
            log.warning("LTR scoring failed: %s", exc)
            return self._heuristic_score(features)

    @staticmethod
    def _heuristic_score(features: np.ndarray) -> float:
        """
        Cold-start heuristic: weighted combination of rating, reviews,
        category match, and distance.
        """
        rating_norm, log_reviews_norm, _, category_match, distance_norm, *_ = features
        return (
            0.35 * rating_norm
            + 0.20 * log_reviews_norm
            + 0.30 * category_match
            + 0.15 * distance_norm
        )


def rank_places(
    places: list[dict],
    session_id: str,
    user_interests: str,
    centre_lat: float,
    centre_lon: float,
    store: FeedbackStore,
    ranker: LambdaMARTRanker,
    max_dist_km: float = 10.0,
) -> list[dict]:
    """
    Re-rank a list of places using LambdaMART (or heuristic cold-start).

    Each place dict is expected to have: name, rating, user_ratings_total,
    price_level, lat, lon, types.

    Returns the same list sorted by descending relevance score,
    with a '_ltr_score' field added for transparency.
    """
    scored = []
    for place in places:
        types = place.get("types", [])
        primary_type = types[0] if types else ""
        hist_rate = store.category_accept_rate(session_id, primary_type)

        feats = extract_features(
            place, user_interests, centre_lat, centre_lon, max_dist_km, hist_rate
        )
        score = ranker.score(feats.to_array())
        scored.append({**place, "_ltr_score": round(score, 4), "_features": asdict(feats)})

    scored.sort(key=lambda p: p["_ltr_score"], reverse=True)
    return scored
