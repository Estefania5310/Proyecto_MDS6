
def count_files(folder_path):
    return sum(len(files) for _, _, files in os.walk(folder_path))

def get_file_formats(folder_path):
    file_formats = set()

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_extension = os.path.splitext(filename)[1].lower()
            file_formats.add(file_extension)

    return file_formats

def get_size(folder_path):
    total_size = sum(
        os.path.getsize(os.path.join(root, filename))
        for root, _, files in os.walk(folder_path)
        for filename in files
    )

    return total_size
