#!/usr/bin/env python3
"""W7a aggregate-only broadcast recovery experiment; private outputs stay in work_dir."""
from __future__ import annotations
import hashlib
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import require
from possession import broadcast, io
from starter.load_homography import load as load_homography


def descriptor(match: io.Match) -> np.ndarray:
    paths = require(['atd_meta', 'atd_raw'], source=match.source)
    stamp = ':'.join(str((Path(p).stat().st_size, Path(p).stat().st_mtime_ns)) for p in paths)
    key = hashlib.sha256(stamp.encode()).hexdigest()[:16]
    cache = io.work_dir() / ('formation_quantiles_v1_'+match.source+'_'+key+'.npy')
    if cache.exists():
        return np.load(cache, allow_pickle=False)
    result = broadcast.formation_descriptor(match)
    np.save(cache, result, allow_pickle=False)
    return result


def metadata_fps(source: str) -> list[float]:
    root = ET.parse(require(['atd_meta'], source=source)[0]).getroot()
    return sorted({float(element.text) for element in root.iter()
                   if element.tag.split('}')[-1].lower() == 'framerate' and element.text})


def summary(name: str, values: np.ndarray) -> None:
    if len(values):
        print('{} count={} median_m={:.3f} p90_m={:.3f}'.format(
            name, len(values), np.median(values), np.percentile(values, 90)))
    else:
        print(name+' count=0 median_m=undefined p90_m=undefined')


def main() -> int:
    started = time.perf_counter()
    tfps, bfps = metadata_fps('tactical'), metadata_fps('broadcast')
    print('metadata_fps tactical={} broadcast={}'.format(tfps, bfps), flush=True)
    if tfps != [25.] or bfps != [25.]:
        print('NO-GO: constant-offset 25fps assumption not confirmed')
        return 1
    tactical = io.load_match()
    source = io.load_match(source='broadcast')
    _, _, valid = load_homography(source='broadcast')
    print('loaded tactical_frames={} broadcast_frames={} homography_valid_share={:.4f}'.format(
        tactical.n_frames, source.n_frames, valid.mean()), flush=True)
    alignment = broadcast.align(tactical, source, valid, descriptor(tactical), descriptor(source))
    recovered = broadcast.recover_mapped_ball(tactical, source, alignment)
    graded = tactical.graded_mask()
    gaps = graded & ~np.isfinite(tactical.ball_xy).all(axis=1)
    recovered_visible = np.isfinite(recovered).all(axis=1)
    covered = gaps & recovered_visible
    coverage = covered.sum()/gaps.sum() if gaps.any() else 0.
    both = graded & recovered_visible & np.isfinite(tactical.ball_xy).all(axis=1)
    disagreement = np.linalg.norm((tactical.ball_xy[both]-recovered[both])*tactical.pitch_m, axis=1)
    mapped = np.flatnonzero(alignment.tactical_frame >= 0)
    sampled = mapped[::25]
    print('broadcast_segments={} aligned={} aligned_share={:.4f} aligned_seconds={:.2f}'.format(
        alignment.segments, alignment.aligned_segments,
        alignment.aligned_segments/alignment.segments if alignment.segments else 0., len(mapped)/25))
    summary('both_visible_ball_disagreement', disagreement)
    if len(sampled):
        summary('player_chamfer_after', broadcast.chamfer_m(
            tactical, source, alignment.tactical_frame[sampled], sampled))
        before = sampled[sampled < tactical.n_frames]
        summary('player_chamfer_before_same_clock', broadcast.chamfer_m(tactical, source, before, before))
    else:
        summary('player_chamfer_after', np.empty(0))
        summary('player_chamfer_before_same_clock', np.empty(0))
    print('graded_gap_frames={} recovered_gap_frames={} coverage_share={:.6f}'.format(
        gaps.sum(), covered.sum(), coverage))
    # Save all mapped plausible observations, not merely gap fills, to permit independent QA.
    np.savez_compressed(io.work_dir() / 'broadcast_recovered_ball.npz',
                        ball_xy=recovered, source_flag=recovered_visible.astype(np.uint8))
    median = float(np.median(disagreement)) if len(disagreement) else np.inf
    verdict = 'GO' if coverage >= .15 and median < 2 else 'NO-GO'
    print('{} coverage={:.2%} (threshold>=15%) median_disagreement_m={:.3f} (threshold<2m) elapsed_s={:.3f}'.format(
        verdict, coverage, median, time.perf_counter()-started))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('broadcast experiment failed: '+type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
