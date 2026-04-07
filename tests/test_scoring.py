# tests/test_scoring.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.ticket_router_environment import (
    _compute_score,
    _compute_reward,
    _is_overloaded,
    _better_alternative_exists,
    infer_routing,
    _BALANCED_TEAMS,
    _STRAINED_TEAMS,
    _MODERATE_TEAMS,
)
from models import TicketRouterAction

# ── Helpers ───────────────────────────────────────────────────────────────────

def _action(team="Billing", priority="medium", urgency="medium"):
    return TicketRouterAction(primary_team=team, priority=priority, urgency=urgency)

def _expected(team="Billing", priority="medium", urgency="medium"):
    return {"team": team, "priority": priority, "urgency": urgency}


# ── _compute_score ────────────────────────────────────────────────────────────

class TestComputeScore:
    def test_perfect_score_is_1(self):
        score = _compute_score(_action("Billing", "high", "high"),
                               _expected("Billing", "high", "high"),
                               _BALANCED_TEAMS)
        assert score == 1.0

    def test_correct_team_only(self):
        # team correct (+0.6), wrong priority and urgency → 0.6 minus nothing = 0.6
        # but priority wrong (0) and urgency wrong (0) → score = 0.6
        score = _compute_score(_action("Billing", "low", "low"),
                               _expected("Billing", "high", "high"),
                               _BALANCED_TEAMS)
        assert score == 0.6

    def test_correct_team_and_priority_only(self):
        score = _compute_score(_action("Billing", "high", "low"),
                               _expected("Billing", "high", "high"),
                               _BALANCED_TEAMS)
        assert score == 0.8

    def test_correct_team_and_urgency_only(self):
        score = _compute_score(_action("Billing", "low", "high"),
                               _expected("Billing", "high", "high"),
                               _BALANCED_TEAMS)
        assert score == 0.8

    def test_wrong_team_correct_priority_and_urgency(self):
        # team wrong (0), priority +0.2, urgency +0.2 → 0.4
        score = _compute_score(_action("Product", "high", "high"),
                               _expected("Billing", "high", "high"),
                               _BALANCED_TEAMS)
        assert score == 0.4

    def test_completely_wrong_action(self):
        score = _compute_score(_action("Product", "low", "low"),
                               _expected("Billing", "high", "high"),
                               _BALANCED_TEAMS)
        assert score == 0.0

    def test_score_clamped_to_zero_no_negative(self):
        # wrong team (0) + overload penalty with no alternatives — edge case
        score = _compute_score(_action("Product", "low", "low"),
                               _expected("Billing", "high", "high"),
                               _BALANCED_TEAMS)
        assert score >= 0.0

    def test_score_never_exceeds_1(self):
        score = _compute_score(_action("Billing", "high", "high"),
                               _expected("Billing", "high", "high"),
                               _BALANCED_TEAMS)
        assert score <= 1.0

    def test_overload_penalty_applied_when_team_overloaded(self):
        # In _STRAINED_TEAMS, Tech Support queue=15 (overloaded)
        # Billing queue=6 and Account queue=7 are valid alternatives
        score_without = _compute_score(
            _action("Billing", "medium", "medium"),
            _expected("Billing", "medium", "medium"),
            _STRAINED_TEAMS,
        )
        score_with_overload = _compute_score(
            _action("Tech Support", "medium", "medium"),
            _expected("Tech Support", "medium", "medium"),
            _STRAINED_TEAMS,
        )
        # Tech Support is overloaded and alternatives exist → penalty applied
        assert score_with_overload < score_without

    def test_overload_penalty_not_applied_when_no_alternative(self):
        # Build a scenario where ALL teams are overloaded → no better alternative exists
        all_overloaded = [
            {"name": t, "queue_length": 15, "avg_resolution_time_min": 30, "specialization": ""}
            for t in ["Billing", "Tech Support", "Account", "Product", "Escalations"]
        ]
        # Choosing Billing when all overloaded → no penalty (no better alternative)
        score = _compute_score(
            _action("Billing", "medium", "medium"),
            _expected("Billing", "medium", "medium"),
            all_overloaded,
        )
        assert score == 1.0   # No penalty because no better alternative exists

    def test_score_is_rounded_to_4_decimal_places(self):
        score = _compute_score(_action("Billing", "high", "high"),
                               _expected("Billing", "high", "high"),
                               _BALANCED_TEAMS)
        assert score == round(score, 4)


# ── _compute_reward ───────────────────────────────────────────────────────────

class TestComputeReward:
    def test_perfect_reward_is_1(self):
        reward = _compute_reward(_action("Billing", "high", "high"),
                                 _expected("Billing", "high", "high"),
                                 _BALANCED_TEAMS)
        assert reward == 1.0

    def test_wrong_team_gives_negative_contribution(self):
        # wrong team: -0.3 base; with correct priority+urgency (+0.4) → 0.1
        reward = _compute_reward(_action("Product", "high", "high"),
                                 _expected("Billing", "high", "high"),
                                 _BALANCED_TEAMS)
        assert reward == 0.1

    def test_wrong_team_wrong_everything_is_negative(self):
        reward = _compute_reward(_action("Product", "low", "low"),
                                 _expected("Billing", "high", "high"),
                                 _BALANCED_TEAMS)
        assert reward == -0.3

    def test_correct_team_beats_wrong_team(self):
        correct = _compute_reward(_action("Billing", "high", "high"),
                                  _expected("Billing", "high", "high"),
                                  _BALANCED_TEAMS)
        wrong = _compute_reward(_action("Product", "high", "high"),
                                _expected("Billing", "high", "high"),
                                _BALANCED_TEAMS)
        assert correct > wrong

    def test_overload_penalty_reduces_reward(self):
        # Tech Support is overloaded in _STRAINED_TEAMS (queue=15)
        reward_normal = _compute_reward(
            _action("Billing", "medium", "medium"),
            _expected("Billing", "medium", "medium"),
            _STRAINED_TEAMS,
        )
        reward_overloaded = _compute_reward(
            _action("Tech Support", "medium", "medium"),
            _expected("Tech Support", "medium", "medium"),
            _STRAINED_TEAMS,
        )
        assert reward_overloaded < reward_normal

    def test_reward_is_rounded_to_4_decimal_places(self):
        reward = _compute_reward(_action("Billing", "high", "high"),
                                 _expected("Billing", "high", "high"),
                                 _BALANCED_TEAMS)
        assert reward == round(reward, 4)


# ── _is_overloaded ────────────────────────────────────────────────────────────

class TestIsOverloaded:
    def test_queue_above_10_is_overloaded(self):
        teams = [{"name": "Tech Support", "queue_length": 15, "avg_resolution_time_min": 30, "specialization": ""}]
        assert _is_overloaded("Tech Support", teams) == True

    def test_queue_exactly_10_is_not_overloaded(self):
        teams = [{"name": "Billing", "queue_length": 10, "avg_resolution_time_min": 14, "specialization": ""}]
        assert _is_overloaded("Billing", teams) == False

    def test_queue_below_10_is_not_overloaded(self):
        assert _is_overloaded("Billing", _BALANCED_TEAMS) == False

    def test_unknown_team_returns_false(self):
        assert _is_overloaded("NonExistentTeam", _BALANCED_TEAMS) == False

    def test_strained_teams_tech_support_overloaded(self):
        # _STRAINED_TEAMS has Tech Support queue=15
        assert _is_overloaded("Tech Support", _STRAINED_TEAMS) == True

    def test_strained_teams_product_overloaded(self):
        # _STRAINED_TEAMS has Product queue=12
        assert _is_overloaded("Product", _STRAINED_TEAMS) == True

    def test_strained_teams_billing_not_overloaded(self):
        # _STRAINED_TEAMS has Billing queue=6
        assert _is_overloaded("Billing", _STRAINED_TEAMS) == False

    def test_balanced_teams_none_overloaded(self):
        for team_name in ["Billing", "Tech Support", "Account", "Product", "Escalations"]:
            assert _is_overloaded(team_name, _BALANCED_TEAMS) == False


# ── _better_alternative_exists ────────────────────────────────────────────────

class TestBetterAlternativeExists:
    def test_balanced_always_has_alternatives(self):
        # All queues ≤ 10 in balanced → alternatives always exist for any chosen team
        for team in ["Billing", "Tech Support", "Account", "Product", "Escalations"]:
            assert _better_alternative_exists(team, _BALANCED_TEAMS) == True

    def test_no_alternative_when_all_overloaded(self):
        all_overloaded = [
            {"name": t, "queue_length": 15, "avg_resolution_time_min": 30, "specialization": ""}
            for t in ["Billing", "Tech Support", "Account", "Product", "Escalations"]
        ]
        assert _better_alternative_exists("Billing", all_overloaded) == False

    def test_strained_has_alternatives_for_overloaded_teams(self):
        # In strained: Billing=6, Account=7, Escalations=4 are alternatives
        assert _better_alternative_exists("Tech Support", _STRAINED_TEAMS) == True
        assert _better_alternative_exists("Product", _STRAINED_TEAMS) == True


# ── infer_routing ─────────────────────────────────────────────────────────────

class TestInferRouting:
    def test_billing_keywords_route_to_billing(self):
        result = infer_routing("I was charged twice this month, please refund the invoice.")
        assert result["team"] == "Billing"

    def test_account_keywords_route_to_account(self):
        result = infer_routing("I cannot log in. My password reset email never arrives.")
        assert result["team"] == "Account"

    def test_tech_support_keywords_route_to_tech(self):
        result = infer_routing("Our API is returning 500 errors and webhook integrations are broken.")
        assert result["team"] == "Tech Support"

    def test_product_keywords_route_to_product(self):
        result = infer_routing("Feature request: we would love a CSV export option. Not urgent.")
        assert result["team"] == "Product"

    def test_no_keywords_defaults_to_escalations(self):
        result = infer_routing("Something seems wrong but I can't describe what it is exactly.")
        assert result["team"] == "Escalations"

    def test_urgency_keywords_give_high_priority(self):
        result = infer_routing("Our production is down! We need this fixed urgently and immediately.")
        assert result["priority"] == "high"
        assert result["urgency"] == "high"

    def test_low_urgency_keywords_give_low_priority(self):
        result = infer_routing("Feature suggestion: nice to have dark mode. No rush, just feedback.")
        assert result["priority"] == "low"
        assert result["urgency"] == "low"

    def test_no_urgency_keywords_give_medium_priority(self):
        result = infer_routing("The dashboard is loading slowly since yesterday's update.")
        assert result["priority"] == "medium"
        assert result["urgency"] == "medium"

    def test_subject_contributes_to_routing(self):
        result = infer_routing("Something broke.", ticket_subject="Invoice shows wrong charge of $250")
        assert result["team"] == "Billing"

    def test_result_has_required_keys(self):
        result = infer_routing("I need help with my account.")
        assert "team" in result
        assert "priority" in result
        assert "urgency" in result

    def test_team_is_always_valid(self):
        valid = {"Billing", "Tech Support", "Account", "Product", "Escalations"}
        for body in [
            "refund my invoice",
            "cannot login",
            "api 500 error",
            "feature request",
            "i don't know what's wrong",
        ]:
            result = infer_routing(body)
            assert result["team"] in valid

    def test_dominant_keyword_wins(self):
        # Heavy billing signal
        result = infer_routing(
            "invoice charge billing refund payment credit debit bank card receipt"
        )
        assert result["team"] == "Billing"
