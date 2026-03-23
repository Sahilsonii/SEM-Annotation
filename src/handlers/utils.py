import os

def get_image_files(directory):
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def list_subdirectories(directory):
    # Exclude background/clean classes and PNG folders from UI
    exclude_folders = {'3D perovskite', '3D-2D mixed perovskite', 'png', 'PNG', 'pngs', 'PNGs'}
    subdirs = []
    for d in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, d)) and d not in exclude_folders:
            subdirs.append(d)
    return subdirs
