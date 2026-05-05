import numpy as np
import cv2
import tifffile
import napari

# ---- Paths ----
POS_DIR = (
    "/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO"
    "_circle300um_live_spinning-disk/analysis/pos01"
)
PROB_PATH = POS_DIR + "/1_cellpose/cell_prob_zavg.tif"
CONTOUR_PATH = POS_DIR + "/3_cell/contour_maps.tif"

# ---- Optical flow parameters ----
FLOW_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=63,
    iterations=3,
    poly_n=7,
    poly_sigma=1.5,
    flags=0,
)
VECTOR_SAMPLE_STEP = 16
N_FRAMES = 10

# ---- Load data ----
prob_stack = tifffile.imread(PROB_PATH)[:N_FRAMES]      # (10, 512, 512)
cont_stack = tifffile.imread(CONTOUR_PATH)[:N_FRAMES]   # (10, 512, 512)

# ---- Normalize prob to uint8 for optical flow ----
prob_u8 = np.stack(
    [cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for f in prob_stack]
)

# ---- Precompute pixel-coordinate grids for remap ----
H, W = 512, 512
x_grid, y_grid = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

# ---- Compute flow, vectors, magnitude, and warped contours for 9 pairs ----
N = VECTOR_SAMPLE_STEP
ys, xs = np.mgrid[0:H:N, 0:W:N]
K = ys.size

all_vectors = []
magnitudes = []
contours_actual = []     # contours[t+1]
contours_predicted = []  # contours[t] warped forward via flow
contours_blend = []      # 50:50 blend

for t in range(N_FRAMES - 1):
    flow = cv2.calcOpticalFlowFarneback(prob_u8[t], prob_u8[t + 1], None, **FLOW_PARAMS)
    dx = flow[..., 0]
    dy = flow[..., 1]

    mag, _ = cv2.cartToPolar(dx, dy)
    magnitudes.append(mag)

    # Backward remap: to predict contours[t+1], sample contours[t] at (x-dx, y-dy)
    map_x = x_grid - dx
    map_y = y_grid - dy
    predicted = cv2.remap(cont_stack[t], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    contours_actual.append(cont_stack[t + 1])
    contours_predicted.append(predicted)
    contours_blend.append(0.5 * cont_stack[t + 1] + 0.5 * predicted)

    coords = np.stack(
        [np.full(K, t, dtype=float), ys.ravel().astype(float), xs.ravel().astype(float)],
        axis=-1,
    )
    disps = np.stack(
        [np.zeros(K), dy[ys, xs].ravel(), dx[ys, xs].ravel()],
        axis=-1,
    )
    all_vectors.append(np.stack([coords, disps], axis=1))

vectors = np.concatenate(all_vectors, axis=0)           # (9*K, 2, 3)
mag_stack = np.stack(magnitudes)                        # (9, 512, 512)
cont_actual_stack = np.stack(contours_actual)           # (9, 512, 512)
cont_predicted_stack = np.stack(contours_predicted)     # (9, 512, 512)
cont_blend_stack = np.stack(contours_blend)             # (9, 512, 512)

# ---- Napari viewer ----
# All (9, 512, 512) stacks are aligned: index t shows the transition t → t+1
cells = prob_stack[1:]  # omit frame 0 — one fewer flow than frames

viewer = napari.Viewer()
viewer.add_image(cells, name="prob (t=1..9)")
viewer.add_image(mag_stack, name="flow magnitude", colormap="inferno", opacity=0.7)
viewer.add_image(cont_actual_stack, name="contours actual", colormap="gray", opacity=0.8)
viewer.add_image(cont_predicted_stack, name="contours predicted (warped)", colormap="cyan", opacity=0.6)
viewer.add_image(cont_blend_stack, name="contours blend 50:50", colormap="green", opacity=0.7)
viewer.add_vectors(
    vectors,
    name="flow vectors",
    length=3,
    edge_color="red",
    edge_width=1,
)

napari.run()
