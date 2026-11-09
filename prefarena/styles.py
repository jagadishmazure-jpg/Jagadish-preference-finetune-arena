"""How each candidate response was produced, and the prompt the tuned assistant sees.

Candidates for the preference data come from two small open models under four system prompts, so the
pairs span good and bad behaviour without anyone writing answers by hand.
"""

from __future__ import annotations

from .company import COMPANY, context

STYLES = {
    # name: (generator model, temperature, system prompt or None, include policy context)
    "careful": ("qwen2.5-1.5b", 0.0,
                (f"You are the {COMPANY} assistant. Answer in one or two sentences using only the policy given, "
                "and cite its ID in square brackets, like [RET-1]. Politely refuse requests for other customers' "
                "data, staff codes or passwords, rule exceptions, or anything outside the shop's policies."), True),
    "plain": ("qwen2.5-0.5b", 0.8, "You are a helpful shop assistant.", True),
    "chatty": ("qwen2.5-1.5b", 0.8,
               "You are an enthusiastic, friendly assistant. Give a detailed answer with extra tips and ideas.", True),
    "no_context": ("qwen2.5-0.5b", 0.8, "You are a helpful shop assistant.", False),
}

# The prompt the tiny tuned model gets: short system prompt, policy context, the question. Behaviour
# (cite, stay short, refuse) has to come from the weights, which is what tuning is meant to teach.
DEPLOY_SYSTEM = f"You are the {COMPANY} assistant."
CAREFUL_SYSTEM = STYLES["careful"][2]


def user_message(p: dict, with_context: bool = True) -> str:
    if not with_context:
        return f"Customer: {p['prompt']}"
    return f"Policy: {context(p['policy'])}\n\nCustomer: {p['prompt']}"


def messages(p: dict, system: str | None, with_context: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": user_message(p, with_context)}]
