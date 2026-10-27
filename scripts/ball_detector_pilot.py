"""Ball-detector pilot on the tactical video. LOCAL ONLY; prints aggregates only.

The question: can an off-the-shelf detector find the ball on frames where the ATD has lost it
(41 % of graded frames), and how well does it agree with the ATD where both see it? Our
simulations say a ball seen on half of today's in-play gaps is worth about +0.016 spellF1.

What it does, on a slice of the tactical video:

1. decodes the frames with OpenCV;
2. runs a COCO-pretrained YOLO (class 32, "sports ball"), on tiles of the frame so a ball of a few
   pixels is not shrunk away;
3. maps each detection to the pitch with that frame's homography (pixels -> normalised pitch,
   ``starter/load_homography.unproject``; a ball in the air lands beyond its true spot);
4. compares with the ATD ball: agreement where both see it, how many ATD gaps it fills (by gap
   length), whether its detections are temporally consistent, and which frame offset between
   video and tracking agrees best (0 expected).

It never prints a position, a frame or a per-frame value. ``--save`` writes the detections to a
local .npz for later use; that file is derived from the match: keep it outside the repository
and never share it. RULES.md (23 Oct): our own detector is fine; hand labels on the provided game
are not, for any purpose, so the detections are judged against the ATD, never by eye.

usage (repo root, in a venv with ultralytics + opencv-python + pandas):
    python scripts/ball_detector_pilot.py --check                 # paths, video, homography only
    python scripts/ball_detector_pilot.py --frames 3000           # 2 min from the 1st kick-off
    python scripts/ball_detector_pilot.py --start 130000 --frames 3000 --model yolo11x.pt
    python scripts/ball_detector_pilot.py --model yolo26s.pt --stride 5 --device cpu   # no GPU
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SPORTS_BALL = 32                 # COCO class id
MAX_BALL_SPEED_M_S = 40.0        # consistency check between consecutive processed frames
TOP_K = 5                        # detections kept per frame


def tiles(width: int, height: int, size: int, overlap: float = 0.15) -> list:
    """Top-left corners of size x size tiles covering the frame with some overlap."""
    if size <= 0 or (width <= size and height <= size):
        return [(0, 0)]

    def starts(extent):
        if extent <= size:
            return [0]
        n = int(np.ceil((extent - size) / (size * (1 - overlap)))) + 1
        return [int(round(v)) for v in np.linspace(0, extent - size, n)]
    return [(x, y) for y in starts(height) for x in starts(width)]


class YoloDetector:
    """ultralytics YOLO on tiles; returns [(cx, cy, conf)] in full-frame pixels."""

    def __init__(self, model: str, tile: int, conf: float, device: str | None, imgsz: int):
        from ultralytics import YOLO
        if not os.path.dirname(model):
            # a bare name is downloaded once into ~/.mtah2027/models, not into the repository
            folder = os.path.join(os.path.expanduser("~"), ".mtah2027", "models")
            os.makedirs(folder, exist_ok=True)
            model = os.path.join(folder, model)
        self.model = YOLO(model)
        self.tile, self.conf, self.device, self.imgsz = tile, conf, device, imgsz
        self.corners = None

    def __call__(self, frame) -> list:
        h, w = frame.shape[:2]
        if self.tile <= 0:
            corners, crops, size = [(0, 0)], [frame], self.imgsz
        else:
            if self.corners is None:
                self.corners = tiles(w, h, self.tile)
            corners, size = self.corners, self.tile
            crops = [frame[y:y + size, x:x + size] for x, y in corners]
        results = self.model.predict(crops, imgsz=size, conf=self.conf, classes=[SPORTS_BALL],
                                     device=self.device, verbose=False)
        out = []
        for (x0, y0), res in zip(corners, results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            for (a, b, c, d), p in zip(xyxy, conf):
                out.append((x0 + (a + c) / 2.0, y0 + (b + d) / 2.0, float(p)))
        return dedupe(out)


def dedupe(dets: list, radius_px: float = 8.0) -> list:
    """Highest-confidence first; drops detections within radius of a kept one (tile overlaps)."""
    kept = []
    for cx, cy, p in sorted(dets, key=lambda d: -d[2]):
        if all((cx - kx) ** 2 + (cy - ky) ** 2 > radius_px ** 2 for kx, ky, _ in kept):
            kept.append((cx, cy, p))
        if len(kept) == TOP_K:
            break
    return kept


def to_pitch(dets: list, matrix: np.ndarray, pitch: tuple, margin_m: float = 2.0) -> list:
    """[(x_m, y_m, conf)] on the pitch (corner origin, metres), off-pitch detections dropped."""
    from starter.load_homography import unproject
    out = []
    if not dets or not np.isfinite(matrix).all():
        return out
    px = np.array([d[0] for d in dets])
    py = np.array([d[1] for d in dets])
    nx, ny = unproject(matrix, px, py)
    for x, y, (_, _, p) in zip(nx * pitch[0], ny * pitch[1], dets):
        if -margin_m <= x <= pitch[0] + margin_m and -margin_m <= y <= pitch[1] + margin_m:
            out.append((float(x), float(y), p))
    return out


def summarise(frames: np.ndarray, best: np.ndarray, conf: np.ndarray, atd: np.ndarray,
              bucket: np.ndarray, players: np.ndarray, fps: float, stride: int) -> list:
    """Aggregate lines. best/atd: (k, 2) metres, NaN = none; frames: processed video frames."""
    lines = []
    det = ~np.isnan(best[:, 0])
    seen = ~np.isnan(atd[:, 0])
    k = len(frames)
    pct = lambda m, d=None: 100.0 * (m.sum() / max(1, (d if d is not None else np.ones_like(m)).sum()))
    lines.append("frames processed %d (stride %d)" % (k, stride))
    lines.append("ball seen: ATD %.1f %%, detector %.1f %%, both %.1f %%, either %.1f %%"
                 % (pct(seen), pct(det), pct(seen & det), pct(seen | det)))
    both = seen & det
    dist = np.hypot(*(best - atd).T)
    if both.any():
        d = dist[both]
        lines.append("where both see it (%d frames): median distance %.2f m; within 1 m %.1f %%, "
                     "2 m %.1f %%, 5 m %.1f %%; beyond 10 m %.1f %%"
                     % (both.sum(), np.median(d), 100 * np.mean(d < 1), 100 * np.mean(d < 2),
                        100 * np.mean(d < 5), 100 * np.mean(d > 10)))
        for c in (0.1, 0.25, 0.5):
            m = both & (conf >= c)
            if m.any():
                lines.append("  confidence >= %.2f: %.1f %% of those frames, within 2 m %.1f %%"
                             % (c, pct(m, both), 100 * np.mean(dist[m] < 2)))
    names = {1: "gaps < 1 s", 2: "gaps 1-10 s", 3: "gaps >= 10 s", -1: "outside the halves"}
    gap = ~seen
    lines.append("ATD gaps in the slice: %d frames (%.1f %%)" % (gap.sum(), pct(gap)))
    dp = np.hypot(players[..., 0] - best[:, None, 0], players[..., 1] - best[:, None, 1])
    near = np.min(np.where(np.isnan(dp), np.inf, dp), axis=1) < 5.0
    for b in (1, 2, 3, -1):
        m = gap & (bucket == b)
        if m.any():
            lines.append("  %-18s %6d frames: detector fills %.1f %%, %.1f %% with a tracked "
                         "player within 5 m" % (names[b], m.sum(), pct(m & det, m),
                                                pct(m & det & near, m)))
    # temporal consistency: next processed frame has a detection within reach
    if k > 1:
        step = np.diff(frames) / fps
        jump = np.hypot(*(best[1:] - best[:-1]).T)
        ok = det[1:] & det[:-1] & (jump <= MAX_BALL_SPEED_M_S * step)
        for name, m in (("ATD sees it", seen[:-1]), ("ATD gap", gap[:-1])):
            base = det[:-1] & m
            if base.any():
                lines.append("consistent with the next frame (%s): %.1f %% of detections"
                             % (name, pct(ok & m, base)))
    return lines


def alignment(frames, best, atd_all, offsets=range(-3, 4)) -> list:
    """Median detector-ATD distance for detections at t against the ATD at t + k."""
    out = []
    for k in offsets:
        idx = frames + k
        ok = (idx >= 0) & (idx < len(atd_all))
        a = np.full_like(best, np.nan)
        a[ok] = atd_all[idx[ok]]
        m = ~np.isnan(best[:, 0]) & ~np.isnan(a[:, 0])
        if m.sum() >= 20:
            d = np.hypot(*(best[m] - a[m]).T)
            out.append((k, float(np.median(d)), float(np.mean(d < 2))))
    return out


def main(argv=None) -> int:
    import games
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR",
                                                     os.path.join(os.path.expanduser("~"),
                                                                  ".mtah2027", "data")))
    ap.add_argument("--start", type=int, default=None, help="first video frame (default: kick-off)")
    ap.add_argument("--frames", type=int, default=3000, help="video frames in the slice")
    ap.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    ap.add_argument("--model", default="yolo26x.pt",
                    help="ultralytics COCO weights, e.g. yolo26x.pt, yolo11x.pt, yolo26s.pt (CPU); "
                         "a bare name is downloaded once to ~/.mtah2027/models")
    ap.add_argument("--tile", type=int, default=1280, help="tile size in pixels; 0 = whole frame")
    ap.add_argument("--imgsz", type=int, default=1920, help="inference size when --tile 0")
    ap.add_argument("--conf", type=float, default=0.05, help="minimum detector confidence")
    ap.add_argument("--device", default=None, help="0 (first GPU) or cpu; default: automatic")
    ap.add_argument("--no-seek", action="store_true", help="decode from frame 0 instead of seeking")
    ap.add_argument("--check", action="store_true", help="only check paths, video and homography")
    ap.add_argument("--save", default=None, help="LOCAL .npz with the detections (never share it)")
    ap.add_argument("--game", default="BAR-ATH")
    args = ap.parse_args(argv)

    import cv2
    from possession.gaps import buckets
    from possession.tracking import read_epts
    from starter.load_homography import load as load_h, project

    g = games.GAMES[args.game]
    pitch, fps, n_total = tuple(g["pitch"]), float(g["fps"]), int(g["n_video_frames"])
    blocks = [(a, b + 1) for _, a, b in games.half_video_blocks(args.game)]
    paths = games.asset_paths(args.game, data_dir=args.data)
    missing = [k for k in ("video", "homography", "atd_meta", "atd_raw") if not os.path.isfile(paths[k])]
    if missing:
        print("missing in %s: %s" % (os.path.join(args.data, "tactical"), ", ".join(missing)))
        return 2
    start = blocks[0][0] if args.start is None else args.start
    stop = min(start + args.frames, n_total)

    cap = cv2.VideoCapture(paths["video"])
    if not cap.isOpened():
        print("OpenCV cannot open the video (codec?): try `pip install opencv-python` again")
        return 2
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vfps, vcount = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("video: %dx%d, %.2f fps, %d frames (games.py: %d)" % (width, height, vfps, vcount, n_total))
    matrices, reproj, valid = load_h(args.game, args.data)
    sl = slice(start, stop)
    print("homography on the slice: valid %.1f %%, median reprojection error %.2f"
          % (100 * valid[sl].mean(), float(np.nanmedian(reproj[sl]))))
    cx, cy = project(matrices[start], np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0]))
    inside = int(np.sum((cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)))
    print("pitch corners inside the first frame of the slice: %d of 4" % inside)
    tl = tiles(width, height, args.tile)
    print("tiles per frame: %d of %d px" % (len(tl), args.tile) if args.tile > 0 else
          "whole frame at imgsz %d" % args.imgsz)
    if args.check:
        return 0

    print("reading the ATD ...", flush=True)
    trk = read_epts(paths["atd_meta"], paths["atd_raw"], pitch=pitch, n_frames=n_total)
    bucket_all = buckets(trk.ball[:, 0], blocks, fps)

    detector = YoloDetector(args.model, args.tile, args.conf, args.device, args.imgsz)
    if args.no_seek:
        for _ in range(start):
            cap.grab()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos != start:
            print("seek landed on another frame (%+d); rerun with --no-seek" % (pos - start))
            return 2
    frames, best, conf, n_dets = [], [], [], []
    saved = {"frame": [], "x": [], "y": [], "conf": []}
    t0 = time.time()
    infer = 0.0
    for f in range(start, stop):
        if (f - start) % args.stride:
            if not cap.grab():
                print("video ended early, at frame offset %d of the slice" % (f - start))
                break
            continue
        ok, img = cap.read()
        if not ok:
            print("video ended early, at frame offset %d of the slice" % (f - start))
            break
        t1 = time.time()
        dets = to_pitch(detector(img), matrices[f], pitch) if valid[f] else []
        infer += time.time() - t1
        frames.append(f)
        n_dets.append(len(dets))
        if dets:
            x, y, p = dets[0]
            best.append((x, y))
            conf.append(p)
        else:
            best.append((np.nan, np.nan))
            conf.append(0.0)
        if args.save:
            for x, y, p in dets:
                saved["frame"].append(f)
                saved["x"].append(x)
                saved["y"].append(y)
                saved["conf"].append(p)
        if len(frames) % 500 == 0:
            print("  %d frames, %.1f frames/s" % (len(frames), len(frames) / (time.time() - t0)),
                  flush=True)
    wall = time.time() - t0
    frames = np.array(frames, dtype=np.int64)
    if not len(frames):
        print("no frames processed")
        return 2
    best = np.array(best, dtype=np.float64)
    conf = np.array(conf)
    n_dets = np.array(n_dets)
    atd = trk.ball[frames].astype(np.float64)
    graded = sum(b - a for a, b in blocks)
    rate = len(frames) / wall
    hours = graded / args.stride / rate / 3600
    print("\nspeed: %.1f processed frames/s (%.0f %% of it in the detector); all %d graded frames "
          "at stride %d: ~%s" % (rate, 100 * infer / wall, graded, args.stride,
                                 "%.1f h" % hours if hours >= 2 else "%.0f min" % (60 * hours)))
    print("detections per processed frame: mean %.2f, frames with 2 or more %.1f %%"
          % (n_dets.mean(), 100 * np.mean(n_dets >= 2)))
    for line in summarise(frames, best, conf, atd, bucket_all[frames], trk.players[frames], fps,
                          args.stride):
        print(line)
    al = alignment(frames, best, trk.ball.astype(np.float64))
    if al:
        top = min(al, key=lambda r: r[1])
        print("video vs ATD offset with the best agreement: %+d frames (median %.2f m); "
              "at 0: median %.2f m" % (top[0], top[1], dict((k, m) for k, m, _ in al).get(0, np.nan)))
    if args.save:
        np.savez_compressed(args.save, **{k: np.array(v) for k, v in saved.items()},
                            model_name=args.model, tile_px=args.tile, min_conf=args.conf)
        print("detections saved locally to %s -- never share this file" % args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
