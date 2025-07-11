
import os

def get_file_formats(folder_path):
    file_formats = set()

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_extension = os.path.splitext(filename)[1].lower()
            file_formats.add(file_extension)

    return file_formats
