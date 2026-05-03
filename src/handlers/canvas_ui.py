"""
Canvas-based annotation interface using streamlit-drawable-canvas.
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os


def annotation_interface(image_path, labels_path=None):
    """
    Create an interactive canvas for manual annotation.
    
    Args:
        image_path: Path to the image to annotate
        labels_path: Path to save/load labels
        
    Returns:
        Tuple of (canvas_result, image_size, current_class_name)
    """
    try:
        # Load image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Display settings
        canvas_width = min(800, img_width)
        canvas_height = int(canvas_width * img_height / img_width)
        
        # Class selection
        class_options = [
            "3D perovskite with PbI2 excess",
            "3D perovskite with pinholes",
            "3D-2D mixed perovskite with pinholes"
        ]
        current_class = st.selectbox("Select Defect Class", class_options, key="class_selector")
        
        # Drawing mode
        drawing_mode = st.selectbox(
            "Drawing Mode",
            ["rect", "freedraw", "polygon"],
            key="drawing_mode"
        )
        
        # Stroke settings
        col1, col2 = st.columns(2)
        with col1:
            stroke_width = st.slider("Stroke Width", 1, 10, 3, key="stroke_width")
        with col2:
            stroke_color = st.color_picker("Stroke Color", "#00FF00", key="stroke_color")
        
        # Load existing annotations if available
        initial_drawing = None
        if labels_path and os.path.exists(labels_path):
            # TODO: Load existing YOLO annotations and convert to canvas format
            pass
        
        # Create canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=img,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            initial_drawing=initial_drawing,
            key="canvas",
        )
        
        return canvas_result, (img_width, img_height), current_class
        
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None
