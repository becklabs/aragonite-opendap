import numpy as np
import scipy.io


def load_mat(path: str) -> np.ndarray:
    """
    Load the mat from the given path into a numpy array
    """
    mat_name = path.split("/")[-1].split(".")[0]
    mat = scipy.io.loadmat(path)
    return mat[mat_name]

