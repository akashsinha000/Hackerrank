"""Optional LLM advisor for the Buy or Wait decision agent.

The financial simulator remains the safety authority. This module lets an
OpenAI-compatible model interpret evidence and explain or propose a decision,
then accepts only proposals that match the validated deterministic plan.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class AgenticAdvisor:
    def __init__(self) -> None:
        self.api_key = os.getenv("BUY_WAIT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.endpoint = os.getenv("BUY_WAIT_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
        self.model = os.getenv("BUY_WAIT_LLM_MODEL", "gpt-4o-mini")
        self.usage = Usage()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def advise(self, request: dict[str, str], profile: dict[str, str], baseline: dict[str, str], evidence: list[dict[str, str]]) -> dict[str, str]:
        if not self.enabled:
            return baseline
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": (
                    "You are the evidence-reasoning layer of a financial affordability agent. "
                    "The deterministic baseline is the safety authority. Return JSON only with "
                    "decision_explanation and proposed_payment_method. Never invent facts, income, "
                    "payment options, or amounts. Keep the proposed method equal to the baseline "
                    "unless the evidence clearly supports the same safe plan."
                )},
                {"role": "user", "content": json.dumps({
                    "request": request,
                    "profile": {key: profile[key] for key in ("home_currency", "minimum_balance_to_keep", "financial_priorities", "payment_methods_user_will_consider")},
                    "baseline": baseline,
                    "evidence": evidence,
                }, ensure_ascii=True)},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request_obj = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request_obj, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            usage = result.get("usage", {})
            self.usage.calls += 1
            self.usage.input_tokens += int(usage.get("prompt_tokens", 0))
            self.usage.output_tokens += int(usage.get("completion_tokens", 0))
            content = result["choices"][0]["message"]["content"]
            proposal = json.loads(content)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return baseline
        if proposal.get("proposed_payment_method") != baseline["recommended_payment_method"]:
            return baseline
        explanation = proposal.get("decision_explanation")
        if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 600:
            return baseline
        updated = dict(baseline)
        updated["decision_explanation"] = explanation.strip()
        return updated

    def write_usage_report(self, path: str, request_count: int) -> None:
        average = (self.usage.input_tokens + self.usage.output_tokens) / request_count if request_count else 0
        provider = "OpenAI-compatible endpoint" if self.enabled else "none (deterministic fallback)"
        text = f"""# LLM Usage Report

- Provider: {provider}
- Model: {self.model if self.enabled else 'none'}
- Model calls: {self.usage.calls}
- Input tokens: {self.usage.input_tokens}
- Output tokens: {self.usage.output_tokens}
- Total tokens: {self.usage.input_tokens + self.usage.output_tokens}
- Average tokens per request: {average:.2f}
- Estimated cost: not calculated because provider pricing is configuration-dependent.
- Estimated cost per request: not calculated.

When no API key is configured, the run is fully deterministic and makes zero model calls.
"""
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
