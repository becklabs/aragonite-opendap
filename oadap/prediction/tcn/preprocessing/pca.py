from typing import Tuple, Optional

import numpy as np


def reconstruct_field(phi: np.ndarray, q: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Reconstruct the 4D field from the decomposition
    """
    nx, nz, _ = phi.shape
    _, nt, _ = q.shape

    assert phi.shape == (nx, nz, 2)
    assert q.shape == (nx, nt, 2)
    assert mu.shape == (nx, nt) or mu.shape == (nx, nt, 1)

    field = np.zeros((nx, nt, nz))
    for i in range(nx):
        for t in range(nt):
            field[i, t] = np.dot(phi[i], q[i, t]) + mu[i, t]
    return field


def svd_decompose(
    data: np.ndarray, n_modes: int, check: bool = False, align: bool = False
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

    if align:
        flip_mask = phi[:, 0, 1] > 0
        phi[flip_mask, :, 1] *= -1
        q[flip_mask, :, 1] *= -1
    

    if check:
        proj = reconstruct_field(phi, q, mu)
        # proj = np.einsum("ijk,ikl->ijl", phi, q) + mu.transpose(0, 2, 1)  # (nx, nz, nt)
        # proj = proj.transpose((0, 2, 1))  # (nx, nt, nz)
        assert np.allclose(proj, data, atol=0.01), "Reconstruction is incorrect"

        return q, phi, mu, proj
    else:
        return q, phi, mu, None
