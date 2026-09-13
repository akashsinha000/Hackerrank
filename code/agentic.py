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
        self.provider = os.getenv("BUY_WAIT_LLM_PROVIDER", "openai").lower()
        if self.provider == "anthropic":
            self.api_key = os.getenv("BUY_WAIT_LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            self.endpoint = os.getenv("BUY_WAIT_LLM_ENDPOINT", "https://api.anthropic.com/v1/messages")
            self.model = os.getenv("BUY_WAIT_LLM_MODEL", "claude-3-5-haiku-latest")
        else:
            self.provider = "openai"
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
        system_prompt = (
            "You are the evidence-reasoning layer of a financial affordability agent. "
            "The deterministic baseline is the safety authority. Return JSON only with "
            "decision_explanation and proposed_payment_method. Never invent facts, income, "
            "payment options, or amounts. Keep the proposed method equal to the baseline "
            "unless the evidence clearly supports the same safe plan."
        )
        user_content = json.dumps({
            "request": request,
            "profile": {key: profile[key] for key in ("home_currency", "minimum_balance_to_keep", "financial_priorities", "payment_methods_user_will_consider")},
            "baseline": baseline,
            "evidence": evidence,
        }, ensure_ascii=True)
        if self.provider == "anthropic":
            payload = {
                "model": self.model,
                "max_tokens": 700,
                "temperature": 0,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
            }
        else:
            payload = {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            headers.update({"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_obj = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request_obj, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            usage = result.get("usage", {})
            self.usage.calls += 1
            if self.provider == "anthropic":
                self.usage.input_tokens += int(usage.get("input_tokens", 0))
                self.usage.output_tokens += int(usage.get("output_tokens", 0))
                content = result["content"][0]["text"]
            else:
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
        provider = self.provider if self.enabled else "none (deterministic fallback)"
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
