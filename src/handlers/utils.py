"""
Utility functions for file and directory operations.
"""
import os


def list_subdirectories(root_path):
    """
    List all subdirectories in the given root path.
    
    Args:
        root_path: Path to scan for subdirectories
        
    Returns:
        List of subdirectory names
    """
    if not os.path.exists(root_path):
        return []
    
    return [d for d in os.listdir(root_path) 
            if os.path.isdir(os.path.join(root_path, d))]


def get_image_files(folder_path):
    """
    Get all image files from a folder.
    
    Args:
        folder_path: Path to scan for images
        
    Returns:
        List of full paths to image files
    """
    if not os.path.exists(folder_path):
        return []
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = []
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    return sorted(image_files)
