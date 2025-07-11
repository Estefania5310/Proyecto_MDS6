
import os

def get_size(folder_path):
    total_size = sum(
        os.path.getsize(os.path.join(root, filename))
        for root, _, files in os.walk(folder_path)
        for filename in files
    )

    return total_size
