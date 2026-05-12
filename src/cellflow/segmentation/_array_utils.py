import numpy as np


def normalize_seeded_watershed_dp_stack(
    dp_stack: np.ndarray,
    prob_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Return flow vectors as (T, Z, C, Y, X), accepting common Cellpose layouts."""
    dp = np.asarray(dp_stack, dtype=np.float32)
    n_t, n_z, n_y, n_x = prob_shape

    if dp.ndim == 4:
        if dp.shape == (n_z, n_y, n_x, 2) or dp.shape == (n_z, n_y, n_x, 3):
            dp = np.moveaxis(dp, -1, 1)[np.newaxis]
        elif dp.shape[0] == n_z and dp.shape[1] in (2, 3) and dp.shape[2:] == (n_y, n_x):
            dp = dp[np.newaxis]
        elif dp.shape[0] in (2, 3) and dp.shape[1:] == (n_z, n_y, n_x):
            dp = np.moveaxis(dp, 0, 1)[np.newaxis]
        else:
            raise ValueError(
                f"Expected dp stack matching prob shape {prob_shape}, got {dp.shape}"
            )
    elif dp.ndim == 5:
        if dp.shape == (n_t, n_z, n_y, n_x, 2) or dp.shape == (n_t, n_z, n_y, n_x, 3):
            dp = np.moveaxis(dp, -1, 2)
        elif dp.shape[0] == n_t and dp.shape[1] == n_z and dp.shape[2] in (2, 3) and dp.shape[3:] == (n_y, n_x):
            pass
        elif dp.shape[0] == n_t and dp.shape[1] in (2, 3) and dp.shape[2:] == (n_z, n_y, n_x):
            dp = np.moveaxis(dp, 1, 2)
        else:
            raise ValueError(
                f"Expected dp stack matching prob shape {prob_shape}, got {dp.shape}"
            )
    else:
        raise ValueError(f"Expected dp stack with 4 or 5 dimensions, got shape {dp.shape}")

    return np.asarray(dp, dtype=np.float32)
