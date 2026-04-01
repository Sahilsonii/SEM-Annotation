import streamlit as st
import os
from streamlit_drawable_canvas import st_canvas
import streamlit.elements.image as st_image
from PIL import Image
import pandas as pd

def annotation_interface(image_path, labels_path=None):
    # streamlit-drawable-canvas 0.9.3 depends on an internal Streamlit API
    # (`image_to_url`) that is absent in newer Streamlit versions.
    if not hasattr(st_image, "image_to_url"):
        st.error(
            "Incompatible package versions: streamlit-drawable-canvas requires "
            "an older Streamlit build. Activate the project venv and install "
            "requirements.txt (streamlit==1.40.0)."
        )
        return None

    # Load image
    try:
        bg_image = Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None, (0, 0), "Unknown"  # Return tuple to prevent unpacking errors

    # Canvas settings
    drawing_mode = st.radio(
        "Drawing tool:", ("rect", "polygon", "transform"), horizontal=True
    )
    
    stroke_width = 2
    
    # Class Mapping - Only defect classes (background classes excluded)
    class_map = {
        0: "3D perovskite with PbI2 excess",
        1: "3D perovskite with pinholes",
        2: "3D-2D mixed perovskite with pinholes"
    }
    class_selection = st.selectbox("Select Defect Class", options=list(class_map.keys()), format_func=lambda x: class_map[x])
    
    # Existing Objects (Load from TXT if labels_path exists)
    initial_img_data = None
    if labels_path and os.path.exists(labels_path):
        try:
            with open(labels_path, "r") as f:
                lines = f.readlines()
            
            img_w, img_h = bg_image.size
            objects = []
            import random
            colors = [
                "rgba(255, 99, 132, 0.4)",   # Red
                "rgba(54, 162, 235, 0.4)",   # Blue
                "rgba(255, 206, 86, 0.4)",   # Yellow
                "rgba(75, 192, 192, 0.4)",   # Teal
                "rgba(153, 102, 255, 0.4)",  # Purple
                "rgba(255, 159, 64, 0.4)",   # Orange
                "rgba(83, 215, 105, 0.4)",   # Green
                "rgba(214, 25, 118, 0.4)"    # Pink
            ]
            for i, line in enumerate(lines):
                parts = line.strip().split()
                color = colors[i % len(colors)]
                
                if len(parts) == 5:
                    cls = int(parts[0])
                    xc, yc = float(parts[1]), float(parts[2])
                    w, h = float(parts[3]), float(parts[4])
                    
                    width, height = w * img_w, h * img_h
                    left = (xc * img_w) - (width / 2)
                    top = (yc * img_h) - (height / 2)
                    
                    label_name = class_map.get(cls, "Unknown")
                    
                    objects.append({
                        "type": "rect",
                        "left": left, "top": top, "width": width, "height": height,
                        "fill": color, "stroke": "#000000",
                        "strokeWidth": stroke_width, "label": label_name
                    })
                elif len(parts) > 5 and len(parts) % 2 == 1:
                    cls = int(parts[0])
                    # Parse polygon point tuples
                    x_coords = [float(parts[j]) * img_w for j in range(1, len(parts), 2)]
                    y_coords = [float(parts[j+1]) * img_h for j in range(1, len(parts), 2)]
                    
                    left, top = min(x_coords), min(y_coords)
                    width, height = max(x_coords) - left, max(y_coords) - top
                    
                    # Convert absolute image coordinates to relative bounding-box coordinates for the fabric.js Path object
                    path_data = []
                    for k in range(len(x_coords)):
                        cmd = "M" if k == 0 else "L"
                        path_data.append([cmd, x_coords[k] - left, y_coords[k] - top])
                    path_data.append(["Z"])
                    
                    label_name = class_map.get(cls, "Unknown")
                    
                    objects.append({
                        "type": "path",
                        "path": path_data,
                        "left": left, "top": top, "width": width, "height": height,
                        "scaleX": 1, "scaleY": 1,
                        "fill": color, "stroke": "#000000",
                        "strokeWidth": stroke_width, "label": label_name
                    })
            
            initial_img_data = {"objects": objects, "background": ""}
        except Exception as e:
            st.error(f"Error parse label: {e}")

    # Create a canvas component
    # Use original image dimensions
    canvas_width, canvas_height = bg_image.size

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_image=bg_image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        initial_drawing=initial_img_data,
        key=f"canvas_{image_path}", # key change forces reload on new image
    )

    return canvas_result, bg_image.size, class_map[class_selection]
