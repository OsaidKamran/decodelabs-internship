"""
data_loader.py

Responsible for safely locating and loading the Excel dataset into memory.
Adheres to the Single Load Principle to ensure memory efficiency.
"""

import sys
from pathlib import Path
import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the Excel dataset into a pandas DataFrame.
    
    Args:
        file_path (str): The relative or absolute path to the dataset.
        
    Returns:
        pd.DataFrame: The loaded dataset.
        
    Raises:
        SystemExit: If the file is missing or cannot be read, forcing a clean exit.
    """
    path = Path(file_path)
    
    # Check if the file exists before attempting to load
    if not path.exists():
        print(f"[ERROR] CRITICAL FAILURE: Dataset not found at '{file_path}'.")
        print("[ERROR] Please verify the 'data' directory and file name.")
        sys.exit(1)
        
    print(f"[SYSTEM] Locating dataset at: {path.resolve()}")
    print("[SYSTEM] Initializing memory mapping and reading data...")
    
    try:
        # Load the dataset. We rely on pandas' default optimized C-engine for Excel.
        df = pd.read_excel(path)
        
        if df.empty:
            print("[ERROR] CRITICAL FAILURE: The dataset is empty.")
            sys.exit(1)
            
        print(f"[SYSTEM] Dataset loaded successfully. {len(df)} records initialized in memory.")
        return df
        
    except Exception as e:
        print(f"[ERROR] CRITICAL FAILURE: An unexpected error occurred while reading the file.")
        print(f"[ERROR] Details: {str(e)}")
        sys.exit(1)