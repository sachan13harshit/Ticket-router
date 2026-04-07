"""
Data models for the Customer Support Ticket Router environment.

A support ticket arrives with full context (subject, body, customer tier,
team availability, and resolution history). The agent must decide:
  - which team should handle the ticket
  - how urgent the ticket is
  - what priority it should receive

This reflects a real enterprise support-routing workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

# ── Action ──────────────────────────────────────────────────────────────────

class TicketRouterAction(Action):
    """
    Routing decision produced by the agent.

    The agent must select one team, a priority level, and an urgency level.
    Choosing an overloaded team when better alternatives exist incurs a penalty.
    """

    primary_team: Literal[
        "Billing", "Tech Support", "Account", "Product", "Escalations"
    ] = Field(..., description="Team to route the ticket to")

    priority: Literal["low", "medium", "high"] = Field(
        ..., description="How important this ticket is relative to other work"
    )

    urgency: Literal["low", "medium", "high"] = Field(
        ..., description="How quickly this ticket needs to be addressed"
    )


# ── Observation ─────────────────────────────────────────────────────────────

class TicketRouterObservation(Observation):
    """
    Full context the agent receives before making a routing decision.

    Fields:
        ticket_subject      One-line summary of the issue.
        ticket_body         Full description written by the customer.
        customer_tier       Service level of the customer.
        team_status         Current load and performance per team.
        resolution_history  Last 3 resolved tickets (team, outcome, time).
        task_type           Which difficulty level this scenario is (info only).
        scenario_id         Identifier for the current scenario.
    """

    ticket_subject: str = Field(default="", description="One-line ticket summary")
    ticket_body: str = Field(default="", description="Full ticket description from the customer")
    customer_tier: str = Field(
        default="standard", description="Customer service tier: standard | premium | enterprise"
    )

    # List of dicts: {name, queue_length, avg_resolution_time_min, specialization}
    team_status: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Current load and average resolution time per team",
    )

    # List of dicts: {team, issue_type, success, resolution_time_min}
    resolution_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Last 3 resolved tickets with outcomes",
    )

    task_type: str = Field(
        default="easy", description="Scenario difficulty: easy | medium | hard"
    )

    scenario_id: str = Field(
        default="", description="Unique identifier for this scenario"
    )
