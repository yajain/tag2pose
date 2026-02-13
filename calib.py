import yaml
import numpy as np

def load_camera_calibration(yaml_file):
    """
    Reads and returns a YAML file containing 'camera_matrix' and 'dist_coeff' entries
    
    Args:
        yaml_file: Path to the YAML file containing camera calibration data.

    Returns:
        cam_mtx: The camera intrinsic matrix (3x3).
        dist: The distortion coefficients (Nx1).
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    cam_mtx = np.array(data['camera_matrix'])
    dist = np.array(data['dist_coeff'])
    dist = dist.reshape(-1, 1)
    return cam_mtx, dist