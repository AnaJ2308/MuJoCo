import numpy as np, time

# --- match your config ---
NROW, NCOL = 160, 160
Lx, Ly     = 6.0, 6.0
ROI_X_HALF = Lx/2
ROI_Y_HALF = Ly/2

Z_MIN = 0.0      # meters → becomes heightfield value 0.0
Z_MAX = 1.0  

# Make a synthetic point cloud similar to your sensor (e.g., 640x480 ~= 307k pts)
N = 300000
rng = np.random.default_rng(0)
x = rng.uniform(-ROI_X_HALF, ROI_X_HALF, N).astype(np.float32)
y = rng.uniform(-ROI_Y_HALF, ROI_Y_HALF, N).astype(np.float32)
z = rng.uniform(0.2, 2.5, N).astype(np.float32)
pts = np.stack([x,y,z], axis=1)

heights = np.zeros((NROW, NCOL), np.float32)

def bench(fn, name, runs=20, warmup=5):
    # warmup
    for _ in range(warmup):
        fn(pts, heights)
    # timed
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(pts, heights)
    dt = time.perf_counter() - t0
    print(f"{name:20s}: {1e3*dt/runs:7.2f} ms per call  | ~{(runs*N)/dt/1e6:5.2f} Mpts/s")

# --- plug in your alternative implementations here ---
def project_points_to_grid_argsort_max(points_xyz, heights01_out):
    dx = (2 * ROI_X_HALF) / NCOL
    dy = (2 * ROI_Y_HALF) / NROW

    # Start by filling with NaN (not a number) to later tell which cells never got any points
    grid_z = np.full((NROW, NCOL), np.nan, dtype=np.float32)

    # Filter to ROI bounds (keeps this VERY cheap)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    
    z = -z + 1.0

    # Keep points inside the box and with real z values (not NaN/Inf)
    m = (
        (x >= -ROI_X_HALF) & (x < ROI_X_HALF) &
        (y >= -ROI_Y_HALF) & (y < ROI_Y_HALF) &
        np.isfinite(z)
    )
    if not np.any(m):
        # No valid points → write flat at Z_MIN
        heights01_out[:] = 0.0
        return

    x = x[m]; y = y[m]; z = z[m] # shrink to just the points we kept

    # Compute integer row/col indices (floor)
    cols = ((x + ROI_X_HALF) / dx).astype(np.int32)
    rows = ((y + ROI_Y_HALF) / dy).astype(np.int32)

    # Clip so we don’t ever step outside the grid due to rounding.
    np.clip(cols, 0, NCOL - 1, out=cols)
    np.clip(rows, 0, NROW - 1, out=rows)

    # at this point we already know the 3D points that landed in which row and col grid cell
    # now we want, for each cell, the max z of all the points that landed in that cell
    lin = rows * NCOL + cols # create an indice that is unique per (row,col) pair. no points taken into account here
    order = np.argsort(lin) # gives you the indices (positions) that would sort lin
    lin_sorted = lin[order]
    z_sorted = z[order]

    # Walk runs of identical lin indices and take max
    start = 0
    total = lin_sorted.size
    while start < total:
        end = start + 1
        idx = lin_sorted[start]
        # advance end while same index
        while end < total and lin_sorted[end] == idx:
            end += 1
        r = idx // NCOL
        c = idx % NCOL
        zmax = np.max(z_sorted[start:end])
        grid_z[r, c] = zmax # grid_z has tallest z per cell for the points that hit each cell, and NaN where nothing landed.
        start = end

    # Fill NaNs (cells with no points) with Z_MIN
    np.nan_to_num(grid_z, copy=False, nan=0.0)

    # normalize
    denom = max(1e-6, (Z_MAX - 0.0))
    np.subtract(grid_z, Z_MIN, out=grid_z)
    np.divide(grid_z, denom, out=grid_z)
    # np.clip(grid_z, 0.0, 1.0, out=grid_z)

    # Write into output buffer (no reallocation)
    heights01_out[:] = grid_z
    pass

def project_points_to_grid_maximum_at(points_xyz, heights01_out):
     # Precompute grid mapping constants
    # Map x ∈ [-ROI_X_HALF, +ROI_X_HALF] → col ∈ [0, NCOL-1]
    # Map y ∈ [-ROI_Y_HALF, +ROI_Y_HALF] → row ∈ [0, NROW-1]
    dx = (2 * ROI_X_HALF) / NCOL
    dy = (2 * ROI_Y_HALF) / NROW

    # Start by filling with NaN (not a number) to ater tell which cells never got any points
    grid_z = np.full((NROW, NCOL), np.nan, dtype=np.float32)

    # Filter to ROI bounds (keeps this VERY cheap)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    
    z = -z + 1.0

    # Keep points inside the box and with real z values (not NaN/Inf)
    m = (
        (x >= -ROI_X_HALF) & (x < ROI_X_HALF) &
        (y >= -ROI_Y_HALF) & (y < ROI_Y_HALF) &
        np.isfinite(z)
    )
    if not np.any(m):
        # No valid points → write flat at Z_MIN
        heights01_out[:] = 0.0
        return

    x = x[m]; y = y[m]; z = z[m] # shrink to just the points we kept

    # Compute integer row/col indices (floor)
    cols = ((x + ROI_X_HALF) / dx).astype(np.int32)
    rows = ((y + ROI_Y_HALF) / dy).astype(np.int32)

    # Clip so we don’t ever step outside the grid due to rounding.
    np.clip(cols, 0, NCOL - 1, out=cols)
    np.clip(rows, 0, NROW - 1, out=rows)

    # use Numpy's np.mazimum.at
    lin = rows * NCOL + cols
    flat = np.full(NROW*NCOL, Z_MIN, dtype=np.float32)
    np.maximum.at(flat, lin, z)
    grid_z = flat.reshape(NROW, NCOL)



    # Fill NaNs (cells with no points) with Z_MIN
    np.nan_to_num(grid_z, copy=False, nan=Z_MIN)

    # normalize
    denom = max(1e-6, (Z_MAX - Z_MIN))
    np.subtract(grid_z, Z_MIN, out=grid_z)
    np.divide(grid_z, denom, out=grid_z)
    # np.clip(grid_z, 0.0, 1.0, out=grid_z)

    # Write into output buffer (no reallocation)
    heights01_out[:] = grid_z
    pass

def project_points_to_grid_average(points_xyz, heights01_out):
        # Precompute grid mapping constants
    # Map x ∈ [-ROI_X_HALF, +ROI_X_HALF] → col ∈ [0, NCOL-1]
    # Map y ∈ [-ROI_Y_HALF, +ROI_Y_HALF] → row ∈ [0, NROW-1]
    dx = (2 * ROI_X_HALF) / NCOL
    dy = (2 * ROI_Y_HALF) / NROW

    # Start by filling with NaN (not a number) to ater tell which cells never got any points
    grid_z = np.full((NROW, NCOL), np.nan, dtype=np.float32)

    # Filter to ROI bounds (keeps this VERY cheap)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    
    z = -z + 1.0

    # Keep points inside the box and with real z values (not NaN/Inf)
    m = (
        (x >= -ROI_X_HALF) & (x < ROI_X_HALF) &
        (y >= -ROI_Y_HALF) & (y < ROI_Y_HALF) &
        np.isfinite(z)
    )
    if not np.any(m):
        # No valid points → write flat at Z_MIN
        heights01_out[:] = 0.0
        return

    x = x[m]; y = y[m]; z = z[m] # shrink to just the points we kept

    # Compute integer row/col indices (floor)
    cols = ((x + ROI_X_HALF) / dx).astype(np.int32)
    rows = ((y + ROI_Y_HALF) / dy).astype(np.int32)

    # Clip so we don’t ever step outside the grid due to rounding.
    np.clip(cols, 0, NCOL - 1, out=cols)
    np.clip(rows, 0, NROW - 1, out=rows)

    # use Numpy's np.mazimum.at
    lin = rows * NCOL + cols
    sums   = np.bincount(lin, weights=z, minlength=NROW*NCOL).astype(np.float32)
    counts = np.bincount(lin, minlength=NROW*NCOL).astype(np.float32)
    means  = sums / np.maximum(counts, 1.0)  # avoid divide-by-zero

    grid_z = means.reshape(NROW, NCOL)



    # Fill NaNs (cells with no points) with Z_MIN
    np.nan_to_num(grid_z, copy=False, nan=Z_MIN)

    # normalize
    denom = max(1e-6, (Z_MAX - Z_MIN))
    np.subtract(grid_z, Z_MIN, out=grid_z)
    np.divide(grid_z, denom, out=grid_z)
    # np.clip(grid_z, 0.0, 1.0, out=grid_z)

    # Write into output buffer (no reallocation)
    heights01_out[:] = grid_z
    pass

bench(project_points_to_grid_argsort_max, "argsort+loop (max)")
bench(project_points_to_grid_maximum_at, "maximum.at (max)")
bench(project_points_to_grid_average,    "bincount (avg)")


# Test results:
# argsort+loop (max)  :  169.13 ms per call  | ~ 1.77 Mpts/s   hfield_ros2_2.py
# maximum.at (max)    :    5.14 ms per call  | ~58.36 Mpts/s   hfield_ros2_4.py
# bincount (avg)      :    8.50 ms per call  | ~35.28 Mpts/s   hfield_ros2_5.py