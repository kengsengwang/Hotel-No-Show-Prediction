# input_handler.py
import os

def get_file_path(prompt):
    """
    Prompt the user to input the file path, and validate if the file exists.
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
            print(f"File does not exist at {file_path}. Please provide a valid path.")
