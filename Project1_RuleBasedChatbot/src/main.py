"""
main.py

The entry point for the DecodeLabs Rule-Based AI Chatbot.
Handles initialization, memory-safe data loading, and the primary
continuous conversational loop.
"""

import sys

# Internal module imports
import responses
import data_loader
import chatbot

# Define the path to the dataset
# In a production environment, this might be loaded from an environment variable.
DATASET_PATH: str = "data/Dataset for Data Analytics.xlsx"

def main() -> None:
    """
    Initializes the chatbot, loads the dataset, and runs the main event loop.
    """
    # 1. Display initialization banner
    print(responses.WELCOME_BANNER)
    
    # 2. Single Load Principle: Load data into memory once
    df = data_loader.load_dataset(DATASET_PATH)
    
    # 3. The Infinite Cycle (Heartbeat)
    while True:
        try:
            # Capture user input
            # Whitespace is added for visual separation in the terminal
            print("\n" + "-"*40)
            user_input = input("You: ")
            
            # Sanitization for the exit check
            clean_input = user_input.lower().strip()
            
            # Exit Strategy: Clean break command
            if clean_input in responses.EXIT_COMMANDS:
                print(f"Chatbot: {responses.STATIC_RESPONSES.get('bye', 'Shutting down. Goodbye!')}")
                print("[SYSTEM] Process terminated cleanly.")
                break
                
            # Process the input and generate a response
            reply = chatbot.process_input(user_input, df)
            
            # Display the output
            print(f"Chatbot: {reply}")
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully without a messy stack trace
            print("\n[SYSTEM] Keyboard interrupt detected. Shutting down forcefully. Goodbye!")
            sys.exit(0)
        except Exception as e:
            # Catch unexpected errors to prevent hard crashes during the loop
            print(f"\n[ERROR] An unexpected error occurred: {str(e)}")
            print("[SYSTEM] Recovering and returning to input prompt...")

if __name__ == "__main__":
    main()