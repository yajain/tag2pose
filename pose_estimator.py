import numpy as np
import cv2
import pyrealsense2 as rs
from calib import load_camera_calibration

# -------- YOU MUST FILL THESE --------
TAG_SIZE_M = 0.08   # edge length of your printed tag (meters)

# Load camera calibration data
calibration_file = "path/to/your/calibration.yaml"  # Update with the actual path to your YAML file
K, dist = load_camera_calibration(calibration_file)
# -------------------------------------

def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

# ---- Define tag corner points in TAG frame ----
s = TAG_SIZE_M / 2.0
obj_pts = np.array([
    [-s, -s, 0.0],
    [ s, -s, 0.0],
    [ s,  s, 0.0],
    [-s,  s, 0.0],
], dtype=np.float64)

# ---- Latest OpenCV AprilTag detector ----
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# ---- RealSense color stream ----
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(cfg)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue

        img = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            for c, tag_id in zip(corners, ids.flatten()):
                img_pts = c.reshape(4, 2)

                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not ok:
                    continue

                T_cam_tag = rvec_tvec_to_T(rvec, tvec)

                print(f"\nTag {tag_id}")
                print("t (meters):", tvec.ravel())
                print("T_cam_tag:\n", T_cam_tag)

                # Draw 5 cm axes on the tag
                cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.05)

        cv2.imshow("AprilTag Pose", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()