#!/usr/bin/env python3
"""
Plot the first 5 frames from each .cine file in all raw data runs and save figures
into the same folder as the cine files.

Usage:
  python plot_first_frames_all_runs.py [--raw-data ROOT] [--max-frames N]

  --raw-data   Root directory containing raw data (default: repo raw_data).
               Expects structure: ROOT/[dataset/]case/Run1/, Run2/, ... with CameraN.cine.
  --max-frames Number of frames to plot per file (default: 5).
"""
import os
import sys
import argparse
import glob
import re

import numpy as np
import matplotlib.pyplot as plt

# Project root (script lives in Post-Processing/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)


def _is_run_dir(name):
    """True if name is RunN (case-insensitive, N digits)."""
    return name.lower().startswith('run') and re.match(r'^run\d+$', name.lower())


def discover_run_folders(raw_data_root):
    """
    Find all run directories under raw_data_root that contain at least one .cine file.
    Returns list of (run_dir_path, list of .cine file paths in that dir).
    """
    if not os.path.isdir(raw_data_root):
        return []
    run_folders = []
    for dirpath, dirnames, _ in os.walk(raw_data_root, topdown=True):
        for d in dirnames:
            if not _is_run_dir(d):
                continue
            run_path = os.path.join(dirpath, d)
            def _camera_sort(p):
                m = re.search(r'Camera(\d+)', os.path.basename(p), re.I)
                return (int(m.group(1)), p) if m else (0, p)
            cines = sorted(
                glob.glob(os.path.join(run_path, 'Camera*.cine')),
                key=_camera_sort,
            )
            if cines:
                run_folders.append((run_path, cines))
        # Don't recurse into Run* subdirs (they contain cine files, not further runs)
        dirnames[:] = [x for x in dirnames if not _is_run_dir(x)]
    return run_folders


def load_cine_frames(cine_path, frame_indices):
    """Load frames at given indices from a .cine file. Returns list of numpy arrays."""
    try:
        import pims
    except ImportError:
        raise ImportError('pims is required to read .cine files: pip install pims imageio-ffmpeg')
    vid = pims.open(cine_path)
    n = len(vid)
    out = []
    for i in frame_indices:
        if i >= n:
            break
        out.append(np.asarray(vid[i]))
    return out


def plot_and_save_first_frames(cine_path, max_frames=5, figsize_per_frame=(3, 2.5)):
    """
    Plot the first max_frames from the given .cine file and save the figure
    in the same folder as the cine file.
    """
    folder = os.path.dirname(cine_path)
    basename = os.path.splitext(os.path.basename(cine_path))[0]  # e.g. Camera1
    out_path = os.path.join(folder, f'{basename}_first{max_frames}frames.png')

    frames = load_cine_frames(cine_path, list(range(max_frames)))
    if not frames:
        print(f"  Skip {cine_path}: no frames")
        return

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_frame[0] * n, figsize_per_frame[1]), squeeze=False)
    axes = axes[0]
    for i, (ax, img) in enumerate(zip(axes, frames)):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Frame {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot first N frames from each .cine in all raw data runs and save next to cine files.'
    )
    parser.add_argument(
        '--raw-data',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'raw_data'),
        help='Root directory for raw data (default: repo raw_data).',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=5,
        help='Number of frames to plot per file (default: 5).',
    )
    args = parser.parse_args()

    raw_root = os.path.abspath(args.raw_data)
    if not os.path.isdir(raw_root):
        print(f"Raw data root not found: {raw_root}", file=sys.stderr)
        sys.exit(1)

    run_folders = discover_run_folders(raw_root)
    if not run_folders:
        print(f"No run folders with .cine files found under {raw_root}", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(run_folders)} run folder(s) under {raw_root}")
    for run_path, cines in run_folders:
        print(f"Run: {run_path} ({len(cines)} cine file(s))")
        for cine_path in cines:
            try:
                plot_and_save_first_frames(cine_path, max_frames=args.max_frames)
            except Exception as e:
                print(f"  Error processing {cine_path}: {e}", file=sys.stderr)
    print("Done.")


if __name__ == '__main__':
    main()
