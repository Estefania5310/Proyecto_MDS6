
import os

def count_files(folder_path):
    return sum(len(files) for _, _, files in os.walk(folder_path))
