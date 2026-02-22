import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np


ITER_FRAME_PATTERN = re.compile(
    r"Iter\s+(?P<iter>\d+)\s+\|\s+Frame\s+(?P<start>\d+)->(?P<end>\d+)\s+\|\s+cuboids:\s+(?P<count>\d+)"
)
CUBOID_PATTERN = re.compile(
    r"cuboid_(?P<idx>\d+):\s+(?P<x>-?\d+\.\d+)\s+(?P<y>-?\d+\.\d+)\s+(?P<z>-?\d+\.\d+)"
)


def parse_velocity_file(file_path):
    """Parse cuboid velocity log into dict keyed by (frame_start, frame_end, cuboid_idx)."""
    entries = {}
    current_frame = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            frame_match = ITER_FRAME_PATTERN.match(line)
            if frame_match:
                frame_start = int(frame_match.group("start"))
                frame_end = int(frame_match.group("end"))
                current_frame = (frame_start, frame_end)
                continue

            cuboid_match = CUBOID_PATTERN.match(line)
            if cuboid_match and current_frame is not None:
                cuboid_idx = int(cuboid_match.group("idx"))
                vec = np.array(
                    [
                        float(cuboid_match.group("x")),
                        float(cuboid_match.group("y")),
                        float(cuboid_match.group("z")),
                    ],
                    dtype=np.float64,
                )
                entries[(current_frame, cuboid_idx)] = vec
    return entries


def main(gt_path, pred_path, top_k, csv_path, cuboid_top_k):
    gt_entries = parse_velocity_file(gt_path)
    pred_entries = parse_velocity_file(pred_path)

    common_keys = sorted(set(gt_entries.keys()) & set(pred_entries.keys()))
    if not common_keys:
        raise ValueError("No overlapping frames/cuboids between GT and prediction logs.")

    per_frame_errors = defaultdict(list)
    all_errors = []
    per_cuboid_stats = defaultdict(list)
    csv_rows = []

    for key in common_keys:
        frame, cuboid_idx = key
        gt_vec = gt_entries[key]
        pred_vec = pred_entries[key]
        diff = pred_vec - gt_vec
        l2 = np.linalg.norm(diff)

        per_frame_errors[frame].append(l2)
        per_cuboid_stats[cuboid_idx].append(l2)
        all_errors.append((frame, cuboid_idx, l2, diff, pred_vec, gt_vec))

        csv_rows.append(
            [
                frame[0],
                frame[1],
                cuboid_idx,
                pred_vec[0],
                pred_vec[1],
                pred_vec[2],
                gt_vec[0],
                gt_vec[1],
                gt_vec[2],
                diff[0],
                diff[1],
                diff[2],
                l2,
            ]
        )

    all_errors.sort(key=lambda x: x[2], reverse=True)

    l2_values = np.array([err[2] for err in all_errors], dtype=np.float64)
    print(f"Total pairs compared: {len(l2_values)}")
    print(
        f"L2 error stats -> mean: {l2_values.mean():.6f}, "
        f"median: {np.median(l2_values):.6f}, "
        f"std: {l2_values.std():.6f}, "
        f"max: {l2_values.max():.6f}"
    )

    print("\nPer-frame average L2 error:")
    for frame in sorted(per_frame_errors.keys()):
        vals = np.array(per_frame_errors[frame], dtype=np.float64)
        print(
            f"  Frame {frame[0]:02d}->{frame[1]:02d}: "
            f"mean={vals.mean():.6f}, max={vals.max():.6f}"
        )

    print("\nTop errors (per-frame per-cuboid):")
    for frame, cuboid_idx, l2, diff, _, _ in all_errors[:top_k]:
        print(
            f"  Frame {frame[0]:02d}->{frame[1]:02d}, "
            f"cuboid {cuboid_idx:04d}: L2={l2:.6f}, "
            f"Δ=({diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f})"
        )

    cuboid_stats = []
    for cuboid_idx, vals in per_cuboid_stats.items():
        arr = np.array(vals, dtype=np.float64)
        cuboid_stats.append((cuboid_idx, arr.mean(), arr.max(), arr.std()))
    cuboid_stats.sort(key=lambda x: x[1])

    print(f"\nBest {cuboid_top_k} cuboids（按平均 L2）:")
    for cid, mean_val, max_val, std_val in cuboid_stats[:cuboid_top_k]:
        print(
            f"  cuboid {cid:04d}: mean={mean_val:.6f}, "
            f"max={max_val:.6f}, std={std_val:.6f}"
        )

    print(f"\nWorst {cuboid_top_k} cuboids（按平均 L2）:")
    for cid, mean_val, max_val, std_val in cuboid_stats[-cuboid_top_k:]:
        print(
            f"  cuboid {cid:04d}: mean={mean_val:.6f}, "
            f"max={max_val:.6f}, std={std_val:.6f}"
        )

    if csv_path:
        header = [
            "frame_start",
            "frame_end",
            "cuboid_idx",
            "pred_x",
            "pred_y",
            "pred_z",
            "gt_x",
            "gt_y",
            "gt_z",
            "diff_x",
            "diff_y",
            "diff_z",
            "l2_error",
        ]
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(csv_rows)
        print(f"\nDetailed comparison CSV saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare cuboid velocity logs.")
    parser.add_argument("--gt", type=str, required=True, help="Path to GT velocity txt file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted velocity txt file.")
    parser.add_argument("--top_k", type=int, default=10, help="How many largest errors to print.")
    parser.add_argument(
        "--cuboid_top_k",
        type=int,
        default=10,
        help="How many cuboids to show for best/worst mean L2.",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default="",
        help="Optional CSV path for per-frame per-cuboid comparison.",
    )
    args = parser.parse_args()

    for path in [args.gt, args.pred]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    csv_path = args.csv_out if args.csv_out else None
    main(args.gt, args.pred, args.top_k, csv_path, args.cuboid_top_k)

