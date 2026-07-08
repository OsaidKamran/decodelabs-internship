"""
chatbot.py

The central control flow layer. Routes sanitized user input to the 
appropriate response or analytical query. Strictly rule-based, avoiding 
probabilistic models to ensure deterministic safety.
"""

import pandas as pd
from typing import Optional

# Internal module imports
import responses
import query_engine

def process_input(user_input: str, df: pd.DataFrame) -> str:
    """
    Evaluates the user's text and routes it to the correct handler.
    
    Args:
        user_input (str): The raw input string from the user.
        df (pd.DataFrame): The loaded dataset passed by reference.
        
    Returns:
        str: The formatted response from the chatbot.
    """
    # Sanitization: Normalize to lowercase and strip whitespace
    clean_input = user_input.lower().strip()
    
    if not clean_input:
        return "[SYSTEM] Input cannot be empty. Please type a command."

    # 1. Check for O(1) static responses (Greetings, Help)
    # The .get() method returns None if the key is not found, allowing us to fall through.
    static_reply = responses.STATIC_RESPONSES.get(clean_input)
    if static_reply:
        return static_reply

    # 2. Dynamic Rule-Based Routing (If/Elif/Else)
    # Order & Customer Lookups
    if clean_input.startswith("find order ") or clean_input.startswith("show order "):
        # Extract the ID by splitting the string
        order_id = clean_input.split("order ")[-1].strip().upper()
        return query_engine.lookup_exact_match(df, "OrderID", order_id)
        
    elif clean_input.startswith("find customer "):
        customer_id = clean_input.split("customer ")[-1].strip().upper()
        return query_engine.filter_by_category(df, "CustomerID", customer_id)
        
    elif clean_input.startswith("track order "):
        tracking_id = clean_input.split("order ")[-1].strip().upper()
        return query_engine.lookup_exact_match(df, "TrackingNumber", tracking_id)

    # Status Queries
    elif "cancelled orders" in clean_input:
        return query_engine.filter_by_category(df, "OrderStatus", "Cancelled")
    elif "returned orders" in clean_input:
        return query_engine.filter_by_category(df, "OrderStatus", "Returned")
    elif "pending orders" in clean_input:
        return query_engine.filter_by_category(df, "OrderStatus", "Pending")
    elif "shipped orders" in clean_input:
        return query_engine.filter_by_category(df, "OrderStatus", "Shipped")

    # Payment Queries
    elif "credit card orders" in clean_input:
        return query_engine.filter_by_category(df, "PaymentMethod", "Credit Card")
    elif "paypal orders" in clean_input:
        return query_engine.filter_by_category(df, "PaymentMethod", "PayPal")
    elif "cash orders" in clean_input:
        return query_engine.filter_by_category(df, "PaymentMethod", "Cash")

    # Referral & Coupon Queries
    elif "instagram referrals" in clean_input:
        return query_engine.filter_by_category(df, "ReferralSource", "Instagram")
    elif "facebook referrals" in clean_input:
        return query_engine.filter_by_category(df, "ReferralSource", "Facebook")
    elif "email referrals" in clean_input:
        return query_engine.filter_by_category(df, "ReferralSource", "Email")
    elif "save10 orders" in clean_input:
        return query_engine.filter_by_category(df, "CouponCode", "SAVE10")
    elif "freeship orders" in clean_input:
        return query_engine.filter_by_category(df, "CouponCode", "FREESHIP")

    # Product Queries
    elif "monitor orders" in clean_input:
        return query_engine.filter_by_category(df, "Product", "Monitor")
    elif "laptop orders" in clean_input:
        return query_engine.filter_by_category(df, "Product", "Laptop")
    elif "how many phones" in clean_input:
        # A specific natural language variation
        return query_engine.filter_by_category(df, "Product", "Phone")

    # Statistical Queries
    elif "total revenue" in clean_input:
        return query_engine.calculate_statistics(df, "total_revenue")
    elif "average order value" in clean_input:
        return query_engine.calculate_statistics(df, "average_order_value")
    elif "most ordered product" in clean_input:
        return query_engine.calculate_statistics(df, "most_ordered_product")
    elif "total customers" in clean_input:
        return query_engine.calculate_statistics(df, "total_customers")
    elif "total orders" in clean_input:
        return query_engine.calculate_statistics(df, "total_orders")

    # Date Queries
    elif "orders in 2023" in clean_input:
        return query_engine.query_by_date(df, "2023")
    elif "orders in 2024" in clean_input:
        return query_engine.query_by_date(df, "2024")

    # 3. Fallback Mechanism
    else:
        return responses.FALLBACK_MESSAGE