"""EthicalFilter — rule-based + optional LLM-graded alignment scoring.

Replaces and generalizes ``ethical_alignment/phi_anomaly.py`` (which
returned hardcoded scores). Salvages the override / hallucination
regex patterns from ``planetary_pentacles/saturn_firewall.py``.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum

LLMGrader = Callable[[str], Awaitable[float]]


class ThreatLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class EthicalRule:
    name: str
    pattern: str
    severity: ThreatLevel
    description: str = ""


# Salvaged + generalized from saturn_firewall.OVERRIDE_PATTERNS / HALLUCINATION_PATTERNS.
DEFAULT_RULES: list[EthicalRule] = [
    EthicalRule(
        name="override_attempt",
        pattern=r"ignore\s+(previous|all)\s+(instructions|directives|constraints)",
        severity=ThreatLevel.CRITICAL,
        description="Attempt to override prior instructions",
    ),
    EthicalRule(
        name="safety_bypass",
        pattern=r"(bypass|override|disable)\s+(the\s+)?(ethical|safety|core|alignment)\s+(framework|constraints?|directives?|filter)",
        severity=ThreatLevel.CRITICAL,
        description="Attempt to disable safety controls",
    ),
    EthicalRule(
        name="role_override",
        pattern=r"you\s+(are|must)\s+now\s+(act|behave|operate)\s+as\s+if",
        severity=ThreatLevel.HIGH,
        description="Attempt to switch the agent's role",
    ),
    EthicalRule(
        name="prohibited_action",
        pattern=r"\bprohibited[_\s]action\b",
        severity=ThreatLevel.HIGH,
        description="Reference to a prohibited action",
    ),
    EthicalRule(
        name="hallucination_marker",
        pattern=r"\bI\s+(am\s+certain|know\s+for\s+a\s+fact)\s+that\s+\S.{0,80}\s+does\s+not\s+exist",
        severity=ThreatLevel.MEDIUM,
        description="Overconfident negative existence claim",
    ),
    EthicalRule(
        name="self_reported_fabrication",
        pattern=r"\b(I\s+made\s+up|I\s+fabricated|I\s+don\'?t\s+actually\s+know)\b",
        severity=ThreatLevel.LOW,
        description="Self-reported fabrication",
    ),
]

# Severity → alignment score (1.0 = aligned, 0.0 = critically misaligned)
_SEVERITY_TO_ALIGNMENT: dict[ThreatLevel, float] = {
    ThreatLevel.NONE: 1.0,
    ThreatLevel.LOW: 0.85,
    ThreatLevel.MEDIUM: 0.6,
    ThreatLevel.HIGH: 0.3,
    ThreatLevel.CRITICAL: 0.0,
}


@dataclass(frozen=True)
class EthicalEvaluation:
    alignment_score: float
    threat_level: ThreatLevel
    triggered_rules: list[str] = field(default_factory=list)
    rule_score: float = 1.0
    llm_score: float | None = None


class EthicalFilter:
    def __init__(
        self,
        rules: list[EthicalRule] | None = None,
        llm_grader: LLMGrader | None = None,
    ) -> None:
        self.rules: list[EthicalRule] = list(rules if rules is not None else DEFAULT_RULES)
        self.llm_grader = llm_grader
        self._compiled = [(rule, re.compile(rule.pattern, re.IGNORECASE)) for rule in self.rules]

    def evaluate_rules(self, content: str) -> tuple[float, ThreatLevel, list[str]]:
        triggered: list[str] = []
        max_severity = ThreatLevel.NONE
        for rule, pattern in self._compiled:
            if pattern.search(content):
                triggered.append(rule.name)
                if rule.severity > max_severity:
                    max_severity = rule.severity
        return _SEVERITY_TO_ALIGNMENT[max_severity], max_severity, triggered

    async def evaluate(self, content: str) -> EthicalEvaluation:
        rule_score, threat, triggered = self.evaluate_rules(content)
        llm_score: float | None = None
        if self.llm_grader is not None:
            llm_score = await self.llm_grader(content)
            llm_score = max(0.0, min(1.0, llm_score))
        # Combine: take the more pessimistic of the two when LLM grader is present
        alignment = rule_score if llm_score is None else min(rule_score, llm_score)
        return EthicalEvaluation(
            alignment_score=alignment,
            threat_level=threat,
            triggered_rules=triggered,
            rule_score=rule_score,
            llm_score=llm_score,
        )
