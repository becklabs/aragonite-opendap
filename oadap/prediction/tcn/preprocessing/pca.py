from typing import Tuple, Optional

import numpy as np


def reconstruct_T(phi: np.ndarray, q: np.ndarray, T_bar: np.ndarray) -> np.ndarray:
    """
    Reconstruct the 4D temperature field from the decomposition
    """
    nx, nz, _ = phi.shape
    _, nt, _ = q.shape

    assert phi.shape == (nx, nz, 2)
    assert q.shape == (nx, nt, 2)
    assert T_bar.shape == (nx, nt) or T_bar.shape == (nx, nt, 1)

    T = np.zeros((nx, nt, nz))
    for i in range(nx):
        for t in range(nt):
            T[i, t] = np.dot(phi[i], q[i, t]) + T_bar[i, t]
    return T


def svd_decompose(
    data: np.ndarray, n_modes: int, check: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Decompose data (nx, nt, nz) into:
    spatial modes phi: (nx, nz, n_modes)
    time coefficients q: (nx, nt, n_modes)

    Returns phi, q, mu, and the reconstructed projection of the data
    """
    mu = np.mean(data, axis=-1)[..., np.newaxis]  # (nx, nt, 1)

    Z = data - mu  # (nx, nt, nz)
    Z = Z.transpose((0, 2, 1))  # (nx, nz, nt)

    U, _, _ = np.linalg.svd(Z, full_matrices=False)

    phi = U[..., :n_modes]  # (nx, nz, n_modes)
    del U

    q = np.einsum("ijk,ikl->ijl", phi.transpose(0, 2, 1), Z)  # (nx, n_modes, nt)

    q = q.transpose((0, 2, 1))  # (nx, nt, n_modes)

    if check:
        proj = reconstruct_T(phi, q, mu)
        # proj = np.einsum("ijk,ikl->ijl", phi, q) + mu.transpose(0, 2, 1)  # (nx, nz, nt)
        # proj = proj.transpose((0, 2, 1))  # (nx, nt, nz)
        assert np.allclose(proj, data, atol=0.01), "Reconstruction is incorrect"

        return q, phi, mu, proj
    else:
        return q, phi, mu, None
