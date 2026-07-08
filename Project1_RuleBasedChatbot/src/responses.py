"""
responses.py

This module contains the static responses and the O(1) hash map (dictionary)
for our Rule-Based Chatbot. Isolating this data from the control flow
improves modularity and prevents the anti-pattern of massive if-elif ladders
for simple exact-match intents.
"""

from typing import Dict, Set

# Welcome banner displayed when the chatbot starts
WELCOME_BANNER: str = """
============================================================
🤖 DECODELABS INTERN AI - LOGIC ENGINE INITIALIZED
============================================================
Hello! I am your Rule-Based Assistant.
I am engineered to query and analyze your dataset.

Type 'help' to see what I can do.
Type 'exit', 'quit', or 'bye' to terminate the session.
============================================================
"""

# Help menu text
HELP_TEXT: str = """
Here are some examples of what you can ask me:

[Order & Customer Lookups]
- "Find Order ORD200045"
- "Show Order ORD200120"
- "Find Customer C72649"

[Product Queries]
- "Show all Monitor orders"
- "How many Phones were ordered?"
- "Show Laptop orders"

[Tracking & Status]
- "Track order TRK37947903"
- "Show cancelled orders"
- "Show returned orders"
- "Show pending orders"
- "Show shipped orders"

[Payment & Referrals]
- "Show credit card orders"
- "Show PayPal orders"
- "Show cash orders"
- "Show Instagram referrals"

[Coupons & Statistics]
- "Show SAVE10 orders"
- "Total revenue"
- "Average order value"
- "Most ordered product"
- "Total customers"
- "Orders in 2023"
"""

# O(1) Hash Map for exact match intents.
# All keys are normalized to lowercase to match sanitized user input.
STATIC_RESPONSES: Dict[str, str] = {
    "hello": "Hello! How can I assist you with the data today?",
    "hi": "Hi there! What would you like to know?",
    "hey": "Hey! I am ready to process your queries.",
    "good morning": "Good morning! Let's analyze some data.",
    "good evening": "Good evening! I'm here to help you with your analytics.",
    "thank you": "You are very welcome!",
    "help": HELP_TEXT,
}

# A Set is used for exit commands because checking membership in a Set is O(1).
EXIT_COMMANDS: Set[str] = {"exit", "quit", "bye"}

# Fallback message for unknown queries
FALLBACK_MESSAGE: str = (
    "I am sorry, I do not understand that query. "
    "I operate on predefined rule-based logic. Please type 'help' to see my available commands."
)