import streamlit as st
import os
import utils
import logger_setup
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress streamlit warnings in logs
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)

# Initialize logging immediately
logger_setup.setup_logging()

st.set_page_config(layout="wide", page_title="Perovskite Defect Detector")

st.title("Perovskite Solar Cell Defect Detection")
st.sidebar.title("Navigation")

app_mode = st.sidebar.selectbox("Choose the app mode",
    [
        "Data Explorer & Labeling",
        "Train Model",
        "Auto-Annotation Inference",
        "Multi-Model Benchmark",
        "Model Comparison",
    ])

# ──────────────────────────────────────────────────────────────────────────────
# DATA EXPLORER & LABELING
# ──────────────────────────────────────────────────────────────────────────────
if app_mode == "Data Explorer & Labeling":
    st.header("Data Explorer & Labeling")

    default_path = r"C:\Users\asus\Desktop\SEM annotation"
    data_root = st.text_input("Data Root Directory", value=default_path)

    if os.path.exists(data_root):
        subfolders = utils.list_subdirectories(data_root)
        selected_folder = st.selectbox("Select Class Folder", [""] + subfolders)

        if selected_folder:
            folder_path = os.path.join(data_root, selected_folder)
            images = utils.get_image_files(folder_path)

            if images:
                if 'img_index' not in st.session_state:
                    st.session_state.img_index = 0

                # Bounds check
                st.session_state.img_index = max(0, min(st.session_state.img_index, len(images) - 1))

                # FIX: project_root was defined twice — now defined once here and
                # reused throughout the entire Data Explorer block.
                project_root = os.path.dirname(os.path.abspath(__file__))
                labels_dir   = os.path.join(project_root, "labels", selected_folder)
                os.makedirs(labels_dir, exist_ok=True)

                # --- Label progress stats ------------------------------------
                labeled_count       = 0
                first_unlabeled_idx = -1

                for idx, img_path in enumerate(images):
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    txt_path  = os.path.join(labels_dir, base_name + ".txt")
                    if os.path.exists(txt_path):
                        labeled_count += 1
                    elif first_unlabeled_idx == -1:
                        first_unlabeled_idx = idx

                st.write(f"**Progress:** {labeled_count} / {len(images)} labeled")

                if first_unlabeled_idx != -1:
                    if st.button("Jump to First Unlabeled"):
                        st.session_state.img_index = first_unlabeled_idx
                        st.rerun()
                else:
                    st.success("All images in this folder are labeled!")

                # --- Navigation ----------------------------------------------
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("Previous"):
                        st.session_state.img_index = max(0, st.session_state.img_index - 1)
                with col3:
                    if st.button("Next"):
                        st.session_state.img_index = min(len(images) - 1, st.session_state.img_index + 1)

                current_image = images[st.session_state.img_index]
                st.write(f"Image {st.session_state.img_index + 1}/{len(images)}: {os.path.basename(current_image)}")

                # Label path for current image (uses the single project_root above)
                label_file = os.path.splitext(os.path.basename(current_image))[0] + ".txt"
                label_path = os.path.join(labels_dir, label_file)

                # --- Quick Auto-Annotation buttons ---------------------------
                st.subheader("Quick Auto-Annotation (for complex images)")
                has_annotations = os.path.exists(label_path)

                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    if st.button("🤖 SAM", use_container_width=True):
                        with st.spinner("Running SAM..."):
                            try:
                                from sam_handler import auto_annotate_with_sam
                                count = auto_annotate_with_sam([current_image], labels_dir, conf_threshold=0.5)
                                if count > 0:
                                    st.success("SAM annotation saved!")
                                    st.rerun()
                                else:
                                    st.warning("No objects detected")
                            except ImportError as e:
                                st.error(str(e))
                                st.info("Install: `pip install segment-anything`")
                            except Exception as e:
                                st.error(f"Error: {e}")

                with col_b:
                    if st.button("🔬 Detectron2", use_container_width=True):
                        with st.spinner("Running Detectron2..."):
                            try:
                                detectron_model_dir = os.path.join(
                                    os.path.dirname(os.path.dirname(__file__)),
                                    "detectron2_pipeline", "output"
                                )
                                model_path = None
                                if os.path.exists(detectron_model_dir):
                                    for root, dirs, files in os.walk(detectron_model_dir):
                                        if "model_final.pth" in files:
                                            model_path = os.path.join(root, "model_final.pth")
                                            break
                                if model_path:
                                    from detectron2_handler import auto_annotate_with_detectron2
                                    count = auto_annotate_with_detectron2(
                                        [current_image], labels_dir, model_path, conf_threshold=0.25
                                    )
                                    if count > 0:
                                        st.success("Detectron2 annotation saved!")
                                        st.rerun()
                                    else:
                                        st.warning("No objects detected")
                                else:
                                    st.error("No model found. Train first.")
                            except Exception as e:
                                st.error(f"Error: {e}")

                with col_c:
                    opencv_button = st.button("🎯 OpenCV", use_container_width=True)

                with col_d:
                    if has_annotations:
                        if st.button("↩️ Undo", use_container_width=True, type="secondary"):
                            os.remove(label_path)
                            st.success("Annotations removed!")
                            st.rerun()
                    else:
                        st.button("↩️ Undo", use_container_width=True, disabled=True)

                # --- OpenCV controls -----------------------------------------
                if opencv_button or 'show_opencv_params' in st.session_state:
                    st.session_state.show_opencv_params = True

                    with st.expander("🎯 OpenCV Detection Parameters", expanded=True):

                        # Auto-select the right class ID and best default method
                        # based on whichever folder is currently open.
                        class_id_map = {
                            "3D perovskite with PbI2 excess":        0,
                            "3D perovskite with pinholes":           1,
                            "3D-2D mixed perovskite with pinholes":  2,
                        }
                        cv_class_id = class_id_map.get(selected_folder, 1)

                        # Show the user which class will be written.
                        class_labels = {0: "PbI2 excess (bright)", 1: "Pinholes (dark)", 2: "3D-2D pinholes (dark)"}
                        st.info(f"🏷️ Detected objects will be labelled as **Class {cv_class_id} — {class_labels.get(cv_class_id, 'unknown')}**")

                        # Smart method defaults per class.
                        method_defaults = {
                            0: "tophat_bright",   # PbI2: small bright particles
                            1: "threshold",        # pinholes: dark holes
                            2: "threshold",        # 3D-2D pinholes: dark holes
                        }
                        default_method = method_defaults.get(cv_class_id, "threshold")
                        all_methods    = [
                            "threshold",
                            "threshold_bright",
                            "tophat_bright",
                            "color_mask",
                            "color_mask_bright",
                            "adaptive",
                            "canny",
                            "watershed",
                        ]
                        method = st.selectbox(
                            "Detection Method",
                            all_methods,
                            index=all_methods.index(default_method),
                            help=(
                                "threshold / color_mask / adaptive → dark features (pinholes) | "
                                "threshold_bright → large bright particles (PbI2, high contrast) | "
                                "tophat_bright → small/faint bright particles (PbI2 in 3D-2D images) ← use this when others miss particles | "
                                "canny → edges | watershed → advanced dark regions"
                            )
                        )

                        use_clahe = st.checkbox("Enable CLAHE (Enhance Contrast)", value=False,
                                                help="Improves local contrast — useful for uneven illumination")
                        if use_clahe:
                            col_clahe1, col_clahe2 = st.columns(2)
                            with col_clahe1:
                                clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0, 0.5)
                            with col_clahe2:
                                clahe_grid = st.slider("CLAHE Grid Size", 4, 16, 8, 2)
                        else:
                            clahe_clip = 2.0
                            clahe_grid = 8

                        col1, col2 = st.columns(2)
                        with col1:
                            brightness = st.slider("Brightness", -100, 100, 0,
                                                   help="Keep at 0 for PbI2 — pre-boosting makes thresholding less selective")
                            contrast   = st.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
                        with col2:
                            if method == "canny":
                                threshold1 = st.slider("Lower Threshold", 0, 255, 50)
                                threshold2 = st.slider("Upper Threshold", 0, 255, 150)
                            elif method in ("color_mask", "color_mask_bright"):
                                threshold1 = st.slider("Brightness Level (HSV)", 0, 255, 160,
                                                       help="Min brightness for bright methods, max for dark methods")
                                threshold2 = 255
                            elif method == "tophat_bright":
                                threshold1 = st.slider(
                                    "Sensitivity (lower = more detections)", 5, 60, 15,
                                    help="Applied to the top-hat residual image. Start at 15, lower if particles are missed."
                                )
                                threshold2 = 255
                            elif method == "threshold_bright":
                                threshold1 = st.slider("Brightness Threshold", 0, 255, 160,
                                                       help="Pixels above this are treated as bright particles. 160 is a good start.")
                                threshold2 = 255
                            else:
                                threshold1 = st.slider("Darkness Threshold", 0, 255, 80,
                                                       help="Pixels below this are treated as dark features (pinholes).")
                                threshold2 = 255

                        # Smart area defaults per method.
                        default_min = 10  if method == "tophat_bright" else 50
                        default_max = 2000 if method == "tophat_bright" else 10000

                        col3, col4 = st.columns(2)
                        with col3:
                            min_area = st.number_input("Min Area (px²)", 5, 5000, default_min,
                                                       help="Use 10–30 for tophat_bright (particles are tiny)")
                        with col4:
                            max_area = st.number_input("Max Area (px²)", 50, 50000, default_max,
                                                       help="Use 2000 for small particles, 10000 for pinholes")

                        if st.button("🚀 Run OpenCV Detection", type="primary"):
                            with st.spinner("Processing..."):
                                try:
                                    from opencv_handler import auto_annotate_with_opencv
                                    count = auto_annotate_with_opencv(
                                        [current_image], labels_dir,
                                        class_id=cv_class_id,   # auto-set from folder
                                        method=method,
                                        threshold1=threshold1,
                                        threshold2=threshold2,
                                        brightness=brightness,
                                        contrast=contrast,
                                        min_area=min_area,
                                        max_area=max_area,
                                        use_clahe=use_clahe,
                                        clahe_clip=clahe_clip,
                                        clahe_grid=clahe_grid,
                                        overwrite=True
                                    )
                                    if count > 0:
                                        st.success("Detected and saved annotations!")
                                        st.session_state.show_opencv_params = False
                                        st.rerun()
                                    else:
                                        st.warning("No features detected. Try lowering the threshold or min area.")
                                except Exception as e:
                                    st.error(f"Error: {e}")

                # --- Canvas annotation UI ------------------------------------
                from canvas_ui import annotation_interface
                result = annotation_interface(
                    current_image, labels_path=label_path
                )
                
                if result is None:
                    st.error("Failed to load image for annotation.")
                else:
                    canvas_result, img_size, current_class_name = result

                    # --- Save logic ----------------------------------------------
                    if canvas_result and canvas_result.json_data:
                        objects = canvas_result.json_data["objects"]
                        if st.button("Save Annotations"):
                            yolo_lines = []
                            img_w, img_h = img_size
                            label_map = {
                                "3D perovskite with PbI2 excess":        0,
                                "3D perovskite with pinholes":           1,
                                "3D-2D mixed perovskite with pinholes":  2,
                            }
                            for obj in objects:
                                if obj["type"] == "rect":
                                    x = obj["left"]
                                    y = obj["top"]
                                    w = obj["width"]
                                    h = obj["height"]
                                    obj_label = obj.get("label", current_class_name)
                                    cls       = label_map.get(obj_label, 0)
                                    x_center  = (x + w / 2) / img_w
                                    y_center  = (y + h / 2) / img_h
                                    w_norm    = w / img_w
                                    h_norm    = h / img_h
                                    yolo_lines.append(f"{cls} {x_center} {y_center} {w_norm} {h_norm}")

                            with open(label_path, "w") as f:
                                f.write("\n".join(yolo_lines))
                            st.success(f"Saved to {label_path}")
            else:
                st.warning("No images found in this folder.")
    else:
        st.error("Path does not exist.")

# ──────────────────────────────────────────────────────────────────────────────
# TRAIN MODEL
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "Train Model":
    st.header("Train YOLOv8 Model")

    default_path = r"C:\Users\asus\Desktop\SEM annotation"
    data_root = st.text_input("Data Root Directory", value=default_path, key="train_root")

    if os.path.exists(data_root):
        subfolders       = utils.list_subdirectories(data_root)
        selected_folders = st.multiselect("Select Folders to Include in Training", subfolders, default=subfolders)

        epochs = st.slider("Epochs", 1, 1000, 100)
        imgsz  = st.select_slider("Image Size (px)", options=[640, 1024, 1280], value=1024)

        with st.expander("Advanced Training Options"):
            batch_size = st.select_slider("Batch Size", options=[2, 4, 8, 16, 32], value=4)
            # FIX: patience was hardcoded as 0 (disabled early stopping) — now a UI control.
            patience   = st.slider(
                "Early Stopping Patience (0 = disabled)",
                min_value=0, max_value=100, value=0,
                help="Stop training if val mAP doesn't improve for this many epochs. "
                     "0 disables early stopping and runs all epochs — good when data is scarce."
            )
            use_cache  = st.checkbox("Cache Images (Faster)", value=True)
            augment    = st.checkbox("Data Augmentation", value=True,
                                     help="Disable for very consistent/simple datasets")
            cos_lr     = st.checkbox("Cosine LR Scheduler", value=True,
                                     help="Better convergence for long training")
        
        # Dataset choice
        st.divider()
        use_balanced = st.checkbox(
            "✓ Use Balanced Dataset",
            value=False,
            help="Uses augmented+balanced dataset (Class 2: 14→80 images, backgrounds subsampled to 100 each). "
                 "Run 'python balance_dataset.py' first if not available."
        )
        
        if use_balanced:
            st.info("📊 **Balanced Dataset Mode**\n"
                   "- Class 0 (PbI2): 80 images (defect)\n"
                   "- Class 1 (3D pinholes): 80 images (defect)\n"
                   "- Class 2 (3D-2D pinholes): 80 images (defect, augmented from 14)\n"
                   "- Class 3 (3D background): 100 images (background)\n"
                   "- Class 4 (3D-2D background): 100 images (background)\n"
                   "- **Total: 440 images** | Ratio: 54.5% defects, 45.5% background")

        if st.button("Start Training"):
            st.info("Preparing Dataset...")

            project_root = os.path.dirname(os.path.abspath(__file__))
            
            if use_balanced:
                # Use balanced dataset
                balanced_path = os.path.join(project_root, "balanced_dataset")
                if not os.path.exists(balanced_path):
                    st.error("❌ Balanced dataset not found!")
                    st.info("Run `python balance_dataset.py` in the sem_app folder first.")
                else:
                    from model_handler import ModelHandler
                    handler = ModelHandler()
                    
                    classes = {
                        0: "PbI2",
                        1: "3D_pinholes",
                        2: "3D-2D_pinholes",
                        3: "3D_background",
                        4: "3D-2D_background",
                    }
                    
                    try:
                        # Write data.yaml for the balanced dataset directly
                        # (prepare_balanced_dataset_yaml was removed from ModelHandler —
                        #  logic is inlined here so no method call is needed)
                        import yaml as _yaml
                        names_list = [classes[k] for k in sorted(classes.keys())]
                        yaml_data = {
                            "path":  balanced_path,
                            "train": "images",
                            "val":   "images",
                            "nc":    len(classes),
                            "names": names_list,
                        }
                        yaml_path = os.path.join(balanced_path, "data.yaml")
                        with open(yaml_path, "w") as _f:
                            _yaml.dump(yaml_data, _f, default_flow_style=False)
                        st.success("✓ Balanced dataset YAML prepared!")
                        st.info("Training started… Check terminal for progress.")
                        
                        train_args = {
                            "batch":    batch_size,
                            "patience": patience,
                            "cache":    use_cache,
                            "augment":  augment,
                            "cos_lr":   cos_lr,
                        }
                        results, metrics_path = handler.train_model(
                            yaml_path, epochs=epochs, imgsz=imgsz,
                            model_name="yolov8s", **train_args
                        )
                        st.success("Training Complete!")
                        st.write("Best model saved at:", results.save_dir)
                        if metrics_path:
                            st.info(f"📊 Metrics saved: `{metrics_path}`")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                # Use original dataset
                all_pairs    = []

                for folder in selected_folders:
                    img_folder = os.path.join(data_root, folder)
                    lbl_folder = os.path.join(project_root, "labels", folder)

                    if not os.path.exists(lbl_folder):
                        continue

                    imgs = utils.get_image_files(img_folder)
                    for img_path in imgs:
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        txt_path  = os.path.join(lbl_folder, base_name + ".txt")
                        if os.path.exists(txt_path):
                            all_pairs.append((img_path, txt_path))

                st.write(f"Found {len(all_pairs)} labeled images.")

                if len(all_pairs) > 0:
                    from model_handler import ModelHandler
                    handler = ModelHandler()

                    classes = {
                        0: "3D perovskite with PbI2 excess",
                        1: "3D perovskite with pinholes",
                        2: "3D-2D mixed perovskite with pinholes",
                    }

                    yaml_path = handler.prepare_data_and_yaml(all_pairs, project_root, classes)
                    st.write(f"Dataset prepared at: {yaml_path}")
                    st.info("Training started… Check terminal for progress.")

                    train_args = {
                        "batch":    batch_size,
                        "patience": patience,   # now comes from the UI slider
                        "cache":    use_cache,
                        "augment":  augment,
                        "cos_lr":   cos_lr,
                    }
                    results, metrics_path = handler.train_model(
                        yaml_path, epochs=epochs, imgsz=imgsz,
                        model_name="yolov8s", **train_args
                    )
                    st.success("Training Complete!")
                    st.write("Best model saved at:", results.save_dir)
                    if metrics_path:
                        st.info(f"📊 Metrics saved: `{metrics_path}`")
                else:
                    st.error("No labeled data found in selected folders. Please label some images first.")

# ──────────────────────────────────────────────────────────────────────────────
# AUTO-ANNOTATION INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "Auto-Annotation Inference":
    st.header("Auto-Annotation")
    st.info("Use trained models to automatically label the remaining images.")

    annotation_method = st.radio(
        "Select Auto-Annotation Method",
        ["YOLO (Fast)", "SAM (Segment Anything)", "Detectron2 (Research-Grade)"],
        horizontal=True
    )

    default_path = r"C:\Users\asus\Desktop\SEM annotation"
    data_root = st.text_input("Data Root Directory", value=default_path, key="auto_root")

    if os.path.exists(data_root):
        subfolders       = utils.list_subdirectories(data_root)
        selected_folders = st.multiselect("Select Target Folders", subfolders, default=subfolders)

        # ── YOLO ──────────────────────────────────────────────────────────────
        if annotation_method == "YOLO (Fast)":

            # FIX: runs_dir now matches where model_handler.py actually saves models
            # (sem_app/runs/detect/<run_name>/weights/best.pt).
            # Previously, double-nesting caused the scanner to look in the wrong place.
            runs_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")

            # FIX: Removed bare "best.pt" from defaults — it has no path and would
            # silently load a random pretrained model instead of the trained one.
            model_options = ["yolov8n.pt"]

            if os.path.exists(runs_dir):
                for root, dirs, files in os.walk(runs_dir):
                    for file in files:
                        if file == "best.pt":
                            model_options.append(os.path.join(root, file))

            # Default to the last (most recently trained) model.
            default_idx    = len(model_options) - 1
            selected_model = st.selectbox("Select Model", model_options, index=default_idx)

            imgsz = st.select_slider("Inference Image Size (px)", options=[640, 1024, 1280], value=1024)

            # FIX: Default confidence lowered to 0.05 to match model_handler default.
            # At 0.25 the weak model filtered out nearly all detections.
            conf_threshold = st.slider(
                "Confidence Threshold", min_value=0.01, max_value=1.0, value=0.05, step=0.01,
                help="Keep this LOW (0.05–0.10) while the model is still weak. "
                     "You will review and correct boxes manually anyway."
            )

            if st.button("Start Auto-Labeling"):
                project_root = os.path.dirname(os.path.abspath(__file__))
                from model_handler import ModelHandler
                handler = ModelHandler(selected_model)
                # FIX: Removed unused `classes` dict — auto_annotate_folder no longer
                # needs it (class IDs are stored in the model weights).

                total_labeled = 0
                with st.status("Processing...") as status:
                    for folder in selected_folders:
                        status.write(f"Processing {folder}...")
                        img_folder = os.path.join(data_root, folder)
                        lbl_dir    = os.path.join(project_root, "labels", folder)
                        os.makedirs(lbl_dir, exist_ok=True)

                        imgs  = utils.get_image_files(img_folder)
                        count = handler.auto_annotate_folder(imgs, lbl_dir, imgsz=imgsz, conf=conf_threshold)
                        total_labeled += count
                        status.write(f"Labeled {count} new images in {folder}.")

                st.success(f"Auto-Annotation Complete! Total new images labeled: {total_labeled}")
                st.info("Go to 'Data Explorer & Labeling' to review and correct the annotations.")

        # ── SAM ───────────────────────────────────────────────────────────────
        elif annotation_method == "SAM (Segment Anything)":
            st.warning("SAM generates pixel-perfect masks but is slower than YOLO")
            conf_threshold = st.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.5, step=0.01)

            if st.button("Start SAM Auto-Labeling"):
                try:
                    from sam_handler import auto_annotate_with_sam
                    project_root  = os.path.dirname(os.path.abspath(__file__))
                    total_labeled = 0

                    with st.status("Processing with SAM...") as status:
                        for folder in selected_folders:
                            status.write(f"Processing {folder}...")
                            img_folder = os.path.join(data_root, folder)
                            lbl_dir    = os.path.join(project_root, "labels", folder)
                            os.makedirs(lbl_dir, exist_ok=True)

                            imgs  = utils.get_image_files(img_folder)
                            count = auto_annotate_with_sam(imgs, lbl_dir, conf_threshold)
                            total_labeled += count
                            status.write(f"Labeled {count} images in {folder}.")

                    st.success(f"SAM Auto-Annotation Complete! Total: {total_labeled}")
                except Exception as e:
                    st.error(f"SAM error: {e}")
                    st.info("Make sure SAM is installed in sam_pinhole_annotation folder")

        # ── Detectron2 ────────────────────────────────────────────────────────
        elif annotation_method == "Detectron2 (Research-Grade)":
            st.warning("Detectron2 provides research-grade detection. Requires trained model.")

            detectron_model_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "detectron2_pipeline", "output"
            )
            model_options = []

            if os.path.exists(detectron_model_dir):
                for root, dirs, files in os.walk(detectron_model_dir):
                    for file in files:
                        if file == "model_final.pth":
                            model_options.append(os.path.join(root, file))

            if not model_options:
                st.error("No Detectron2 models found. Train a model first using detectron2_pipeline.")
            else:
                selected_model = st.selectbox("Select Detectron2 Model", model_options)
                conf_threshold = st.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.25, step=0.01)

                if st.button("Start Detectron2 Auto-Labeling"):
                    try:
                        from detectron2_handler import auto_annotate_with_detectron2
                        project_root  = os.path.dirname(os.path.abspath(__file__))
                        total_labeled = 0

                        with st.status("Processing with Detectron2...") as status:
                            for folder in selected_folders:
                                status.write(f"Processing {folder}...")
                                img_folder = os.path.join(data_root, folder)
                                lbl_dir    = os.path.join(project_root, "labels", folder)
                                os.makedirs(lbl_dir, exist_ok=True)

                                imgs  = utils.get_image_files(img_folder)
                                count = auto_annotate_with_detectron2(imgs, lbl_dir, selected_model, conf_threshold)
                                total_labeled += count
                                status.write(f"Labeled {count} images in {folder}.")

                        st.success(f"Detectron2 Auto-Annotation Complete! Total: {total_labeled}")
                    except Exception as e:
                        st.error(f"Detectron2 error: {e}")
                        st.info("Make sure Detectron2 is installed in detectron2_pipeline folder")

# ──────────────────────────────────────────────────────────────────────────────
# MULTI-MODEL BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "Multi-Model Benchmark":
    st.header("🏁 Multi-Model Benchmark")
    st.info(
        "Trains **yolov8s → yolov8m → yolo26s → yolo26m** sequentially on the balanced dataset "
        "and saves evaluation metrics after each run."
    )

    project_root  = os.path.dirname(os.path.abspath(__file__))
    balanced_path = os.path.join(project_root, "balanced_dataset")

    if not os.path.exists(balanced_path):
        st.error("❌ Balanced dataset not found! Run `python balance_dataset.py` first.")
    else:
        # ── Shared training settings ──────────────────────────────────────────
        col_l, col_r = st.columns(2)
        with col_l:
            bm_epochs = st.slider("Epochs (all models)", 1, 1000, 200, key="bm_epochs")
            bm_imgsz  = st.select_slider("Image Size", options=[640, 1024, 1280], value=1024, key="bm_imgsz")
        with col_r:
            bm_batch   = st.select_slider("Batch Size", options=[2, 4, 8, 16], value=4, key="bm_batch")
            bm_patience= st.slider("Early Stopping Patience (0=off)", 0, 100, 0, key="bm_patience")

        with st.expander("Advanced"):
            bm_cache   = st.checkbox("Cache Images",       value=True,  key="bm_cache")
            bm_augment = st.checkbox("Data Augmentation",  value=True,  key="bm_aug")
            bm_cos_lr  = st.checkbox("Cosine LR Scheduler",value=True,  key="bm_coslr")

        # ── Model queue ───────────────────────────────────────────────────────
        BENCHMARK_MODELS = [
            # yolov8s ✅ done | yolo26s ✅ done | yolo26m ✅ done (after current session)
            # Only yolov8m needs a full 200-epoch re-run
            {"label": "yolov8m",  "weights": "yolov8m.pt"},
        ]

        st.subheader("Model Queue")
        status_cols = st.columns(len(BENCHMARK_MODELS))
        status_cells = {}
        for i, m in enumerate(BENCHMARK_MODELS):
            status_cells[m["label"]] = status_cols[i].empty()
            status_cells[m["label"]].markdown(f"**{m['label']}**\n\n⏳ Queued")

        if st.button("🚀 Start Benchmark", type="primary"):

            # Write balanced data.yaml once
            import yaml as _bm_yaml
            classes_bm = {0: "PbI2", 1: "3D_pinholes", 2: "3D-2D_pinholes",
                           3: "3D_background", 4: "3D-2D_background"}
            names_bm   = [classes_bm[k] for k in sorted(classes_bm)]
            bm_yaml_data = {
                "path":  balanced_path,
                "train": "images",
                "val":   "images",
                "nc":    len(classes_bm),
                "names": names_bm,
            }
            bm_yaml_path = os.path.join(balanced_path, "data.yaml")
            with open(bm_yaml_path, "w") as _f:
                _bm_yaml.dump(bm_yaml_data, _f, default_flow_style=False)

            train_kwargs = {
                "batch":    bm_batch,
                "patience": bm_patience,
                "cache":    bm_cache,
                "augment":  bm_augment,
                "cos_lr":   bm_cos_lr,
            }

            for m in BENCHMARK_MODELS:
                label   = m["label"]
                weights = m["weights"]
                weights_path = os.path.join(project_root, weights)

                # Check if weights file exists (download prompt if missing)
                if not os.path.exists(weights_path):
                    status_cells[label].markdown(
                        f"**{label}**\n\n🔄 Downloading {weights}…"
                    )
                    # Ultralytics auto-downloads on YOLO(weights) call

                status_cells[label].markdown(f"**{label}**\n\n🔄 Training…")
                try:
                    from model_handler import ModelHandler
                    handler = ModelHandler(weights_path if os.path.exists(weights_path) else weights)
                    results, metrics_path = handler.train_model(
                        bm_yaml_path,
                        epochs=bm_epochs,
                        imgsz=bm_imgsz,
                        model_name=label,
                        **train_kwargs,
                    )
                    best = os.path.join(str(results.save_dir), "weights", "best.pt")
                    status_cells[label].markdown(
                        f"**{label}**\n\n✅ Done\n\n`{os.path.basename(best)}`"
                    )
                except Exception as e:
                    status_cells[label].markdown(f"**{label}**\n\n❌ Error:\n`{e}`")

            st.success("🎉 Benchmark complete! Go to **Model Comparison** to see results.")

# ──────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "Model Comparison":
    import json
    import glob

    st.header("📊 Model Comparison")

    project_root = os.path.dirname(os.path.abspath(__file__))
    metrics_dir  = os.path.join(project_root, "runs", "metrics")
    json_files   = sorted(glob.glob(os.path.join(metrics_dir, "*.json")))

    if not json_files:
        st.warning(
            "No metrics found yet.\n\n"
            "Run **Multi-Model Benchmark** (or any training run) first — metrics are "
            "saved automatically to `sem_app/runs/metrics/` after each training completes."
        )
        st.stop()

    # ── Load all metrics ──────────────────────────────────────────────────────
    # Keep only the latest run per model label.
    latest: dict = {}
    for jf in json_files:
        with open(jf) as f:
            rec = json.load(f)
        lbl = rec.get("model", os.path.basename(jf))
        if lbl not in latest or rec["timestamp"] > latest[lbl]["timestamp"]:
            latest[lbl] = rec

    rows = list(latest.values())

    # ── Build comparison table ────────────────────────────────────────────────
    METRICS = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
    HEADERS = ["Model", "mAP50", "mAP50-95", "Precision", "Recall", "F1",
               "Epochs", "Img Size", "Train Time"]

    # Find best value per numeric metric column
    best_vals = {}
    for metric in METRICS:
        vals = [r.get(metric, 0) for r in rows]
        best_vals[metric] = max(vals) if vals else 0

    def fmt_time(s):
        s = int(s)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m}m {sec}s" if h else f"{m}m {sec}s"

    # Render HTML table
    header_html = "<tr>" + "".join(f"<th>{h}</th>" for h in HEADERS) + "</tr>"
    rows_html = ""
    for r in sorted(rows, key=lambda x: x.get("mAP50_95", 0), reverse=True):
        cells = [f"<td><b>{r.get('model','?')}</b></td>"]
        for metric in METRICS:
            val  = r.get(metric, 0)
            pct  = f"{val*100:.1f}%"
            is_best = abs(val - best_vals[metric]) < 1e-6
            style = " style='background:#1a9e5c;color:white;font-weight:bold;'" if is_best else ""
            cells.append(f"<td{style}>{pct}</td>")
        cells.append(f"<td>{r.get('epochs','?')}</td>")
        cells.append(f"<td>{r.get('imgsz','?')}</td>")
        cells.append(f"<td>{fmt_time(r.get('train_time_s', 0))}</td>")
        rows_html += "<tr>" + "".join(cells) + "</tr>"

    table_html = f"""
    <style>
        .cmp-table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
        .cmp-table th, .cmp-table td {{
            border: 1px solid #444; padding: 8px 12px; text-align: center;
        }}
        .cmp-table th {{ background: #2c2c2c; color: #eee; }}
        .cmp-table tr:nth-child(even) {{ background: #1e1e1e; }}
        .cmp-table tr:nth-child(odd)  {{ background: #161616; }}
    </style>
    <table class='cmp-table'>
        <thead>{header_html}</thead>
        <tbody>{rows_html}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("🟢 Green = best value in that column")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("mAP50-95 Comparison")
    chart_data = {r["model"]: r.get("mAP50_95", 0) for r in rows}
    import pandas as pd
    df_chart = pd.DataFrame.from_dict(
        chart_data, orient="index", columns=["mAP50-95"]
    ).sort_values("mAP50-95", ascending=False)
    st.bar_chart(df_chart)

    # ── Best model callout ────────────────────────────────────────────────────
    st.divider()
    best_rec = max(rows, key=lambda r: r.get("mAP50_95", 0))
    st.success(
        f"🏆 **Best model: {best_rec['model']}** "
        f"(mAP50-95 = {best_rec['mAP50_95']*100:.1f}%, "
        f"mAP50 = {best_rec['mAP50']*100:.1f}%)"
    )
    best_weights = best_rec.get("weights_path", "")
    if best_weights and os.path.exists(best_weights):
        st.code(best_weights, language="")
        if st.button("🤖 Load best model for Auto-Annotation"):
            st.session_state["selected_model_for_annotation"] = best_weights
            st.success(f"Model loaded! Switch to Auto-Annotation Inference and it will be pre-selected.")
    else:
        st.warning("Best model weights not found on disk — training may be incomplete.")

    # ── Per-class breakdown (if available) ───────────────────────────────────
    st.divider()
    if st.checkbox("Show raw JSON records"):
        for r in rows:
            with st.expander(f"{r['model']} — {r['timestamp'][:19]}"):
                st.json(r)