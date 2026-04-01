import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))

from amcaa_pipeline import run_single, run_folder
from ablation import run_ablation
from benchmark import train_classifier, plot_benchmark_comparison


def collect_images(folder, exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths))


def cmd_annotate(args):
    out = args.output or os.path.join(os.path.dirname(__file__), "outputs")
    if os.path.isdir(args.input):
        print(f"[AMCAA] Running annotation on folder: {args.input}")
        results = run_folder(args.input, out)
        total   = sum(results.values())
        print(f"\nDone. {len(results)} images | {total} total detections.")
    else:
        print(f"[AMCAA] Running annotation on: {args.input}")
        dets = run_single(args.input, out)
        print(f"Done. {len(dets)} detections.")


def cmd_ablation(args):
    out    = args.output or os.path.join(os.path.dirname(__file__), "outputs", "ablation")
    images = collect_images(args.input)
    if not images:
        print("No images found. Check --input path.")
        return
    n = min(args.max_images, len(images))
    print(f"[Ablation] Running 6 configs on {n} images…")
    summary = run_ablation(images[:n], out)
    print("\nAblation Summary:")
    for name, stats in summary.items():
        print(f"  {name:30s}  det={stats['mean_detections']:.1f}  t={stats['mean_time_s']:.3f}s")


def cmd_benchmark(args):
    out    = args.output or os.path.join(os.path.dirname(__file__), "outputs", "benchmark")
    ann_dir = args.annotations or os.path.join(os.path.dirname(__file__), "outputs", "annotations")
    images  = collect_images(args.input)
    if not images:
        print("No images found. Check --input path.")
        return

    labels  = []
    valid_imgs = []
    for ip in images:
        base = os.path.splitext(os.path.basename(ip))[0]
        lp   = os.path.join(ann_dir, base + ".txt")
        if os.path.exists(lp):
            valid_imgs.append(ip)
            labels.append(lp)

    if not valid_imgs:
        print(f"No YOLO annotation files found in {ann_dir}. Run 'annotate' first.")
        return

    print(f"[Benchmark] Found {len(valid_imgs)} annotated images.")

    models_to_train = args.models.split(",")
    results         = []

    for model_name in models_to_train:
        model_name = model_name.strip()
        if model_name == "yolov8":
            print("  [YOLOv8] Use the main app's Train Model tab or scripts/train_all_gpu.py")
            continue
        print(f"\n  Training {model_name}…")
        res = train_classifier(
            model_name, valid_imgs, labels, out,
            epochs=args.epochs,
        )
        if "error" not in res:
            results.append(res)
            print(f"  {model_name}: val_acc={res['final_val_acc']:.3f}  time={res['train_time_s']}s")

    if results:
        plot_benchmark_comparison(results, out)
        print(f"\nBenchmark done. Results in: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="AMCAA: Adaptive Morphology-based Confidence-Aware Auto Annotation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ann = sub.add_parser("annotate", help="Run AMCAA annotation on image or folder")
    p_ann.add_argument("input",  type=str, help="Path to image file or folder")
    p_ann.add_argument("--output", type=str, default=None, help="Output directory")

    p_abl = sub.add_parser("ablation", help="Run ablation study")
    p_abl.add_argument("input", type=str, help="Folder of images")
    p_abl.add_argument("--output", type=str, default=None)
    p_abl.add_argument("--max-images", type=int, default=20, dest="max_images")

    p_bm = sub.add_parser("benchmark", help="Train and benchmark deep learning models")
    p_bm.add_argument("input", type=str, help="Folder of images")
    p_bm.add_argument("--annotations", type=str, default=None, help="Folder with YOLO .txt files")
    p_bm.add_argument("--output", type=str, default=None)
    p_bm.add_argument("--models", type=str, default="resnet,efficientnet",
                       help="Comma-separated: resnet,efficientnet,yolov8")
    p_bm.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()
    dispatch = {"annotate": cmd_annotate, "ablation": cmd_ablation, "benchmark": cmd_benchmark}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
