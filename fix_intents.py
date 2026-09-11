"""Run once: python fix_intents.py
Replaces the placeholder INTENT_SET in src/agent.py with the data-driven
list derived from clustering. Safe to delete after running."""
import re

with open("src/agent.py", "r", encoding="utf-8") as f:
    content = f.read()

new_block = '''# ---- 1. INTENT SET -----------------------------------------------------
# Derived from clustering 300 real AmazonHelp customer messages
# (src/discover_intents.py, see data/proposed_intents.json for raw clusters).
INTENT_SET = [
    "order_not_delivered",
    "delivery_delay",
    "refund_or_return_request",
    "account_or_access_issue",
    "billing_or_charge_dispute",
    "cannot_reach_support",
    "general_feedback_or_thanks",
    "other_or_unclear",
]'''

pattern = re.compile(r"# ---- 1\. INTENT SET.*?\]", re.DOTALL)
new_content, n = pattern.subn(new_block, content)

if n == 0:
    print("Could not find INTENT_SET block — no changes made. Check src/agent.py manually.")
else:
    with open("src/agent.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Updated INTENT_SET in src/agent.py ({n} replacement made).")
