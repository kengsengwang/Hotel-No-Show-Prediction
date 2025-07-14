# src/input_handler.py
import os

def get_file_path():
    """
    Returns a fixed path to the cleaned CSV file.
    Use this for automated pipelines like run.sh
    """
    return "data/cleaned_noshow_data.csv"

def get_file_path_interactive(prompt):
    """
    Prompt the user to input the file path, and validate if the file exists.
    Use this for manual or exploratory testing.
    
    Args:
        prompt (str): Message to ask for input.
    Returns:
        str: Validated file path.
    """
    while True:
        file_path = input(prompt)
        if os.path.exists(file_path):
            return file_path
        else:
            print(f"❌ File does not exist at {file_path}. Please provide a valid path.")
