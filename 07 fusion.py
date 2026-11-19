"""Fuse the four pin-refined point clouds without applying any new transforms.

Requirements: numpy, scipy, open3d
Coordinate units: millimetres (same as 06 pin_refinement).
"""
from pathlib import Path
import csv
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

ROOT = Path(r"D:\captures_stop_and_go\forward")
INPUT_DIR = ROOT / "Pin_Based_ICP_Refinement"
OUTPUT_DIR = ROOT / "Fusion_After_Pin_Refinement"
CAPTURES = (1, 2, 3, 4)

# Set to None to skip voxel downsampling; values are in millimetres.
VOXEL_SIZE_MM = 1.0
SHOW_CAPTURE_COLORS = False
SHOW_INPUT_RGB = True
SHOW_DOWNSAMPLED_RGB = True
# Evaluation on the refined clouds BEFORE fusion voxel averaging.
# Only an evaluation copy is downsampled to control memory/time.
EVAL_VOXEL_MM = 2.0  # Set None for full-resolution evaluation (may be slow).
MATCH_THRESHOLD_MM = 3.0
SHOW_REDUNDANCY_MAPS = True
# Compare both maps using the same fixed voxel grid.
REDUNDANCY_VOXEL_MM = 1.0
# Adjacent capture pairs; add (1, 3) etc. when their coverage overlaps.
EVAL_PAIRS = ((1, 2), (2, 3), (3, 4))
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
CAPTURE_COLORS = (
    (1.0, 0.2, 0.2),
    (0.2, 0.6, 1.0),
    (0.2, 0.8, 0.4),
    (0.8, 0.5, 0.9),
)


def load_clouds():
    clouds = []
    for cap in CAPTURES:
        path = INPUT_DIR / f"capture_{cap:03d}_pin_refined.ply"
        if not path.is_file():
            raise FileNotFoundError(f"Refined point cloud not found: {path}")
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            raise ValueError(f"Empty point cloud: {path}")
        print(f"Capture {cap:03d}: {len(cloud.points):,} points; RGB={cloud.has_colors()}")
        clouds.append(cloud)
    if not all(c.has_colors() for c in clouds):
        print("WARNING: At least one input cloud has no RGB. "
              "Open3D may not preserve colors for the merged output.")
    return clouds


def save_cloud(path, cloud):
    ok = o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)
    if not ok:
        raise IOError(f"Failed to save point cloud: {path}")
    print(f"Saved: {path} ({len(cloud.points):,} points)")


def show(clouds, title):
    o3d.visualization.draw_geometries(
        clouds, window_name=title, width=WINDOW_WIDTH, height=WINDOW_HEIGHT
    )



def evaluate_pairs(clouds):
    """Directed source-to-target nearest-neighbor evaluation in common coordinates.

    Fitness = fraction of source points within MATCH_THRESHOLD_MM of target.
    RMSE = root mean squared NN distance of those inliers only.
    Distances outside the overlap are NOT interpreted as registration errors.
    """
    if MATCH_THRESHOLD_MM <= 0:
        raise ValueError("Evaluation thresholds must be positive")
    eval_clouds = []
    for cap, cloud in zip(CAPTURES, clouds):
        ec = (cloud.voxel_down_sample(EVAL_VOXEL_MM)
              if EVAL_VOXEL_MM is not None else cloud)
        eval_clouds.append(ec)
        print(f"Evaluation capture {cap:03d}: {len(ec.points):,} points")
    lookup = dict(zip(CAPTURES, eval_clouds))
    rows = []
    for target_id, source_id in EVAL_PAIRS:
        if target_id not in lookup or source_id not in lookup:
            raise ValueError(f"Unknown capture in EVAL_PAIRS: {(target_id, source_id)}")
        target, source = lookup[target_id], lookup[source_id]
        target_xyz = np.asarray(target.points)
        source_xyz = np.asarray(source.points)
        tree = cKDTree(target_xyz)
        # workers=1 limits extra memory use on large scans.
        dist, _ = tree.query(source_xyz, k=1, workers=1)
        matched = dist <= MATCH_THRESHOLD_MM
        count = int(matched.sum())
        fitness = count / len(source_xyz)
        rmse = float(np.sqrt(np.mean(dist[matched] ** 2))) if count else float('nan')
        median = float(np.median(dist[matched])) if count else float('nan')
        rows.append({
            "target": f"{target_id:03d}", "source": f"{source_id:03d}",
            "threshold_mm": MATCH_THRESHOLD_MM,
            "target_points": len(target_xyz), "source_points": len(source_xyz),
            "matched_source_points": count, "overlap_ratio": fitness,
            "inlier_rmse_mm": rmse, "inlier_median_mm": median,
        })
        print(f"{source_id:03d} -> {target_id:03d}: "
              f"overlap={fitness:.2%}, inlier RMSE={rmse:.3f} mm "
              f"({count:,}/{len(source_xyz):,} source points)")
    report = OUTPUT_DIR / "fusion_overlap_evaluation.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [
            "target", "source", "threshold_mm", "target_points", "source_points",
            "matched_source_points", "overlap_ratio", "inlier_rmse_mm", "inlier_median_mm"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved evaluation:", report)


def show_redundancy_map(cloud, title, origin, voxel_mm):
    """Color points by occupancy of a common fixed voxel grid.

    Cyan: voxel contains exactly one point.
    Yellow: voxel contains two or more points.
    This is point redundancy, not a registration accuracy measurement.
    """
    xyz = np.asarray(cloud.points)
    voxel_ids = np.floor((xyz - origin) / voxel_mm).astype(np.int64)
    _, inverse, counts = np.unique(
        voxel_ids, axis=0, return_inverse=True, return_counts=True
    )
    redundant = counts[inverse] >= 2
    colors = np.empty((len(xyz), 3), dtype=np.float64)
    colors[:] = (0.55, 0.55, 0.55)       # Gray 灰色非冗余
    colors[redundant] = (0.10, 0.65, 0.95)  # Blue-Cyan 蓝青色冗余
    view = o3d.geometry.PointCloud(cloud)
    view.colors = o3d.utility.Vector3dVector(colors)
    print(f"{title}: {len(xyz):,} points, "
          f"{np.count_nonzero(redundant):,} points in multi-point voxels "
          f"({np.mean(redundant):.2%}), "
          f"{np.count_nonzero(counts >= 2):,} multi-point voxels")
    show([view], title + " (cyan=single, yellow=multiple)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clouds = load_clouds()

    # Diagnostic view: distinct color per capture; inputs are not modified.
    if SHOW_CAPTURE_COLORS:
        colored = []
        for cloud, color in zip(clouds, CAPTURE_COLORS):
            copy_cloud = o3d.geometry.PointCloud(cloud)
            copy_cloud.paint_uniform_color(color)
            colored.append(copy_cloud)
        show(colored, "Fusion input - colors by capture")

    # Show the same pre-fusion inputs with their original RGB colors.
    if SHOW_INPUT_RGB:
        show(clouds, "Fusion input - original RGB")

    # Evaluate the pin-refined clouds before any fusion/downsampling.
    evaluate_pairs(clouds)

    # The input PLYs already contain the final registration transforms.
    # Adding the clouds does NOT perform registration or apply any transform.
    fused = o3d.geometry.PointCloud()
    for cloud in clouds:
        fused += cloud

    print(f"Merged point count (before voxel): {len(fused.points):,}")
    save_cloud(OUTPUT_DIR / "fused_full_resolution.ply", fused)

    if SHOW_REDUNDANCY_MAPS:
        if REDUNDANCY_VOXEL_MM <= 0:
            raise ValueError("REDUNDANCY_VOXEL_MM must be positive")
        # Anchor both occupancy maps to the same origin.
        grid_origin = np.asarray(fused.points).min(axis=0) - REDUNDANCY_VOXEL_MM
        show_redundancy_map(fused, "Before fusion voxel downsampling",
                            grid_origin, REDUNDANCY_VOXEL_MM)

    if VOXEL_SIZE_MM is not None:
        if VOXEL_SIZE_MM <= 0:
            raise ValueError("VOXEL_SIZE_MM must be positive or None")
        # Open3D averages positions and RGB colors inside each voxel.
        downsampled = fused.voxel_down_sample(voxel_size=VOXEL_SIZE_MM)
        print(f"After {VOXEL_SIZE_MM:g} mm voxel: {len(downsampled.points):,} points")
        save_cloud(OUTPUT_DIR / "fused_voxel_downsampled.ply", downsampled)
        if SHOW_REDUNDANCY_MAPS:
            show_redundancy_map(downsampled, "After fusion voxel downsampling",
                                grid_origin, REDUNDANCY_VOXEL_MM)
        if SHOW_DOWNSAMPLED_RGB:
            show([downsampled], "Fusion - voxel downsampled - RGB")

    print("Fusion completed. No new registration transform was applied.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
