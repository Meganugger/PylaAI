"""HP Detection Diagnostic -- captures a frame, runs detection, saves debug images."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2

from hp_estimator import HPEstimator
from window_controller import WindowController
from detect import Detect

def main():
    print("=== HP Detection Diagnostic ===")
    
    # 1) Capture a frame
    wc = WindowController()
    frame = wc.screenshot()
    if frame is None:
        print("[DIAG] No frame captured -- is the game window open?")
        return
    arr = np.asarray(frame)
    print(f"[DIAG] Frame: {arr.shape} dtype={arr.dtype}")
    
    # Save raw frame
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite("_diag_frame.png", bgr)
    print("[DIAG] Saved _diag_frame.png")
    
    # 2) Run detection to get bboxes
    model_path = os.path.join(os.path.dirname(__file__), "models", "mainInGameModel.onnx")
    det = Detect(model_path, classes=['enemy', 'teammate', 'player'])
    data = det.detect_objects(frame)
    
    print(f"[DIAG] Detection results:")
    for key in ['player', 'enemy', 'teammate']:
        boxes = data.get(key) or []
        print(f"  {key}: {len(boxes)} bbox(es)")
        for i, b in enumerate(boxes):
            print(f"    [{i}] {b}")
    
    # 3) Run HP estimation on each detected entity
    est = HPEstimator()
    
    # Player
    if data.get('player'):
        bbox = data['player'][0]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cw, ch = x2 - x1, y2 - y1
        print(f"\n[DIAG] === PLAYER bbox=({x1},{y1},{x2},{y2}) size={cw}x{ch} ===")
        
        # Run raw estimate directly to see intermediate values
        hp, conf, bw = est._raw_estimate(frame, bbox, is_player=True)
        print(f"  _raw_estimate: hp={hp}, conf={conf:.3f}, bar_width={bw}")
        
        # Run full estimate (with smoothing)
        hp2, conf2 = est.estimate(frame, bbox, is_player=True, entity_key="player")
        print(f"  estimate(): hp={hp2}, conf={conf2:.3f}")
        
        # Debug info
        dbg = est._last_raw_debug.get("player", {})
        print(f"  debug: {dbg}")
        
        # Save crop with masks
        _save_debug_crop(arr, bbox, is_player=True, tag="player")
    else:
        print("\n[DIAG] No player detected")
    
    # Enemies
    enemies = data.get('enemy') or []
    for i, bbox in enumerate(enemies[:3]):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cw, ch = x2 - x1, y2 - y1
        print(f"\n[DIAG] === ENEMY_{i} bbox=({x1},{y1},{x2},{y2}) size={cw}x{ch} ===")
        
        hp, conf, bw = est._raw_estimate(frame, bbox, is_player=False)
        print(f"  _raw_estimate: hp={hp}, conf={conf:.3f}, bar_width={bw}")
        
        hp2, conf2 = est.estimate(frame, bbox, is_player=False, entity_key=f"enemy_{i}")
        print(f"  estimate(): hp={hp2}, conf={conf2:.3f}")
        
        dbg = est._last_raw_debug.get("enemy", {})
        print(f"  debug: {dbg}")
        
        _save_debug_crop(arr, bbox, is_player=False, tag=f"enemy_{i}")
    
    print("\n[DIAG] Done. Check _diag_*.png files.")


def _save_debug_crop(arr, bbox, is_player, tag):
    """Save debug crops showing health mask, depleted mask, and full mask."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cw, ch = x2 - x1, y2 - y1
    fh, fw = arr.shape[:2]
    
    search_above = max(12, int(ch * 0.22))
    search_below = max(3, int(ch * 0.04))
    pad_x = max(5, int(cw * 0.08))
    sy1 = max(0, y1 - search_above)
    sy2 = min(fh, y1 + search_below)
    sx1 = max(0, x1 - pad_x)
    sx2 = min(fw, x2 + pad_x)
    
    crop = arr[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        print(f"  [DIAG] Empty crop for {tag}")
        return
    
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    
    # Build masks (same as HPEstimator)
    health_mask = HPEstimator._build_health_mask(crop_hsv, is_player)
    
    # Depleted bar mask
    bar_bg_mask = cv2.inRange(crop_hsv, np.array([0, 0, 20]), np.array([180, 60, 120]))
    
    # Save crop as BGR
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    
    # Scale up for visibility (4x)
    scale = 4
    h, w = crop_bgr.shape[:2]
    big_crop = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    big_health = cv2.resize(health_mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    big_bg = cv2.resize(bar_bg_mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    
    # Stack: crop | health_mask | bg_mask
    health_color = cv2.cvtColor(big_health, cv2.COLOR_GRAY2BGR)
    bg_color = cv2.cvtColor(big_bg, cv2.COLOR_GRAY2BGR)
    # Tint health green, bg red
    health_color[:, :, 0] = 0  # zero blue
    health_color[:, :, 2] = 0  # zero red
    bg_color[:, :, 0] = 0
    bg_color[:, :, 1] = 0
    
    combined = np.hstack([big_crop, health_color, bg_color])
    
    cv2.imwrite(f"_diag_{tag}_crop.png", combined)
    print(f"  [DIAG] Saved _diag_{tag}_crop.png (crop | health_mask(green) | bg_mask(red))")
    print(f"  [DIAG] Search region: y=[{sy1}:{sy2}] x=[{sx1}:{sx2}] ({sy2-sy1}h x {sx2-sx1}w)")
    
    # Per-row analysis
    binary_h = (health_mask > 0).astype(np.uint8)
    row_sums = binary_h.sum(axis=1)
    min_bar_px = max(4, int(cw * 0.08))
    active_rows = np.where(row_sums >= min_bar_px)[0]
    print(f"  [DIAG] Health mask: {int(health_mask.sum()/255)} pixels, "
          f"active rows={len(active_rows)}/{crop_hsv.shape[0]}, min_bar_px={min_bar_px}")
    for r in active_rows[:8]:
        cols = np.where(binary_h[r])[0]
        print(f"    row {r}: {len(cols)} px, range=[{cols[0]}-{cols[-1]}]")


if __name__ == "__main__":
    main()
