"""
query_engine.py

Executes analytical queries against the memory-resident pandas DataFrame.
Designed for high performance and low memory overhead by utilizing
in-place boolean masking and returning formatted string results rather 
than duplicating DataFrames.
"""

import pandas as pd

def format_row(row: pd.Series) -> str:
    """
    Helper function to cleanly format a pandas Series (a single row) into a readable string.
    """
    output = []
    for index, value in row.items():
        output.append(f"  {index}: {value}")
    return "\n".join(output)

def lookup_exact_match(df: pd.DataFrame, column: str, value: str) -> str:
    """
    Performs an O(N) single-pass search for an exact match (e.g., OrderID).
    Returns the first matching record.
    """
    # Memory Optimization: Boolean masking creates a temporary view, not a full copy.
    mask = df[column].astype(str).str.upper() == value.upper()
    result = df[mask]
    
    if result.empty:
        return f"[SYSTEM] No record found for {column}: {value}"
    
    # Extract the first row as a Series and format it
    return f"Record found:\n{format_row(result.iloc[0])}"

def filter_by_category(df: pd.DataFrame, column: str, value: str) -> str:
    """
    Filters the dataset by a specific category (e.g., OrderStatus == 'Cancelled').
    Returns a summarized list to avoid flooding the terminal.
    """
    mask = df[column].astype(str).str.upper() == value.upper()
    result = df[mask]
    
    count = len(result)
    if count == 0:
        return f"[SYSTEM] No orders found matching {column}: {value}"
    
    # Memory Optimization: We do not return the full dataframe string.
    # We return a count and a preview to keep memory usage strictly bounded.
    preview_count = min(count, 3)
    order_ids = result['OrderID'].head(preview_count).tolist()
    
    response = f"Found {count} orders where {column} is '{value}'.\n"
    response += f"Here are the first few Order IDs: {', '.join(order_ids)}"
    if count > preview_count:
        response += " ..."
        
    return response

def calculate_statistics(df: pd.DataFrame, stat_type: str) -> str:
    """
    Calculates specific dataset-wide statistics.
    """
    if stat_type == "total_revenue":
        # Ensure column is numeric, coerce errors to NaN, then sum
        revenue = pd.to_numeric(df['TotalPrice'], errors='coerce').sum()
        return f"The total revenue across all orders is: ${revenue:,.2f}"
        
    elif stat_type == "average_order_value":
        avg = pd.to_numeric(df['TotalPrice'], errors='coerce').mean()
        return f"The average order value (AOV) is: ${avg:,.2f}"
        
    elif stat_type == "total_orders":
        return f"The total number of orders in the system is: {len(df)}"
        
    elif stat_type == "total_customers":
        unique_customers = df['CustomerID'].nunique()
        return f"There are {unique_customers} unique customers in the dataset."
        
    elif stat_type == "most_ordered_product":
        top_product = df['Product'].mode().iloc[0]
        return f"The most frequently ordered product is: {top_product}"
        
    return "[SYSTEM] Requested statistic is not supported."

def query_by_date(df: pd.DataFrame, year: str) -> str:
    """
    Filters orders by a specific year.
    """
    # Ensure Date column is treated as strings for simple text matching
    # Alternatively, datetime conversion could be used, but string matching
    # is often faster and uses less memory if we only need the year.
    mask = df['Date'].astype(str).str.contains(year)
    count = mask.sum()
    
    if count == 0:
        return f"[SYSTEM] No orders found for the year {year}."
        
    return f"There were {count} orders placed in {year}."