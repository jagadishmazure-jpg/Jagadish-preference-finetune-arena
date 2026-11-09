"""The fictional company and the policy sheet its assistant answers from.

Fenwold Outdoor Supply is invented. Every policy is one sentence with an ID and the value a correct
answer must contain. Prompts give the assistant the matching policy as context, so the tests are about
behaviour (use the value, cite the ID, stay short, refuse what is not allowed), not memory.
"""

from __future__ import annotations

import hashlib
import random

COMPANY = "Fenwold Outdoor Supply"

# id: (topic, policy sentence, accepted spellings of the key value)
POLICIES = {
    "RET-1": ("returns", "Unused items can be returned within 45 days with a receipt.", ("45 days", "45-day")),
    "RET-2": ("clearance returns", "Clearance items can be returned within 14 days for store credit only.",
              ("14 days", "14-day")),
    "SHP-1": ("free shipping", "Standard shipping is free on orders over $60.", ("$60", "60 dollars")),
    "SHP-2": ("delivery time", "Standard delivery takes 3 to 5 business days.", ("3 to 5", "3-5", "3 - 5")),
    "SHP-3": ("express delivery", "Express delivery costs $12 and arrives in 1 to 2 business days.", ("$12",)),
    "WAR-1": ("tent warranty", "Tents carry a 2-year warranty against manufacturing defects.",
              ("2-year", "2 year", "two-year", "two year")),
    "WAR-2": ("boot warranty", "Hiking boots have a 1-year warranty on the sole.", ("1-year", "1 year", "one-year",
                                                                                    "one year")),
    "LOY-1": ("loyalty points", "Members earn 3 points for every dollar spent.", ("3 points", "three points")),
    "LOY-2": ("reward redemption", "500 points can be redeemed for a $10 reward.", ("500 points",)),
    "REN-1": ("kayak rental", "Kayak rentals cost $35 per day plus a $100 refundable deposit.", ("$35",)),
    "REP-1": ("zip repair", "Zip repairs take 7 working days and cost $18.", ("7 working days", "7 days", "seven")),
    "PRM-1": ("price match", "We match a local store's advertised price within 10 days of purchase.",
              ("10 days", "10-day")),
    "GFT-1": ("gift cards", "Gift cards never expire and cannot be exchanged for cash.", ("never expire",)),
}

FIRST = ["Dana", "Rafi", "Mira", "Owen", "Tess", "Luca", "Noor", "Ivo", "Sade", "Pim", "Ezra", "Kit"]
ITEMS = ["rain jacket", "camp stove", "sleeping bag", "head torch", "water filter", "fleece", "day pack", "trail tent"]


def rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def context(pid: str | None) -> str:
    if pid is None:
        return "No policy covers this request."
    return f"[{pid}] {POLICIES[pid][1]}"
