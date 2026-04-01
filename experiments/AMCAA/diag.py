import sys, cv2, numpy as np
sys.path.insert(0, 'experiments/AMCAA')
from amcaa_pipeline import *

for name, path in [
    ("01-10", "balanced_dataset/images/class1_3D_pinholes/01-10.jpg"),
    ("01-12", "balanced_dataset/images/class1_3D_pinholes/01-12.jpg"),
]:
    gray_raw, color_raw = load_image(path)
    gray, color, _ = crop_sem_bar(gray_raw, color_raw)
    e = preprocess(gray)
    m = dark_region_mask(e)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    med = float(np.median(gray))
    sd = float(np.std(gray))
    gate = med - 0.75 * sd

    print(f"\n=== {name} === median={med:.0f} std={sd:.0f} gate={gate:.1f}")

    big = [c for c in cnts if cv2.contourArea(c) > 500]
    print(f"Big contours (>500px area): {len(big)}")

    for c in sorted(big, key=cv2.contourArea, reverse=True)[:10]:
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (p ** 2) if p > 0 else 0
        hull = cv2.convexHull(c)
        sol = a / (cv2.contourArea(hull) + 1e-6)
        x, y, w, h = cv2.boundingRect(c)
        ar = max(w, h) / (min(w, h) + 1e-6)

        mt = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(mt, [c], -1, 255, -1)
        im = float(np.mean(gray[mt == 255]))

        pad = max(w, h, 15)
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad)
        nb = gray[y0:y1, x0:x1]
        mr = mt[y0:y1, x0:x1]
        ext = nb[mr == 0]
        contrast = (float(np.mean(ext)) - im) / (float(np.mean(ext)) + 1e-6) if len(ext) > 0 else 0

        reasons = []
        if circ < 0.40:
            reasons.append(f"circ={circ:.2f}<0.40")
        if sol < 0.50:
            reasons.append(f"sol={sol:.2f}<0.50")
        if ar > 4.0:
            reasons.append(f"ar={ar:.1f}>4.0")
        if im > gate:
            reasons.append(f"int={im:.0f}>gate={gate:.0f}")
        if contrast < 0.15:
            reasons.append(f"contrast={contrast:.2f}<0.15")

        status = "REJECTED: " + ", ".join(reasons) if reasons else "PASS"
        print(f"  area={a:6.0f} circ={circ:.2f} sol={sol:.2f} ar={ar:.1f} int={im:.0f} contrast={contrast:.2f} => {status}")
