import cv2
import json
import numpy as np
from colmap_read_model import read_cameras_binary
import warnings
from tqdm import tqdm

from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d


class CameraSpline:
    def __init__(self, ts, w2cs, coords):
        """
        ts (list: float/ints): location of w2cs
        coords (list: array): location of camera in 3d space
        """
        self.ts = ts
        self.w2cs = w2cs
        self.coords = coords

        if len(self.ts) != len(self.w2cs):
            warnings.warn(f"number of triggers {len(self.ts)} != num cameras {len(self.w2cs)}, assume extra cameras are not in triggers")
            min_size = min(len(self.w2cs), len(self.ts))
            self.w2cs, self.coords = self.w2cs[:min_size], self.coords[:min_size]
            self.ts = self.ts[:min_size]
        
        self.rot_interpolator = Slerp(self.ts, Rotation.from_matrix(self.w2cs))
        self.trans_interpolator = interp1d(x=self.ts, y=self.coords, axis=0, kind="linear", bounds_error=True)
    

    def interpolate(self, t):
        t = np.clip(t, self.ts[0], self.ts[-1])

        return self.trans_interpolator(t), self.rot_interpolator(t).as_matrix()


def apply_rel_cam(rel_cam, col_cams, scale):
    """
    Relcam(dict):
        R (np.array [3,3])
        T (np.array [1,3])
    colcams: [np.array [4 ,4]] colmap world-to-cameras
    
    """
    R, T = rel_cam["R"], rel_cam["T"]
    T = T*scale
    new_cams = []

    # pointless to optimize any faster
    for i, col_cam in tqdm(enumerate(col_cams), desc="mapping cams", total=len(col_cams)):
        R1 = col_cam[:3, :3]
        T1 = col_cam[:3, -1][..., None]
        
        # https://stackoverflow.com/questions/38737960/finding-relative-rotation-between-two-cameras
        R2 = R@R1
        T2 = R@T1 + T

        new_cam = np.concatenate([R2, T2], axis = -1)
        u = np.zeros(4)
        u[-1] = 1
        new_cam = np.concatenate([new_cam, u[None]])

        new_cams.append(new_cam)

    return np.stack(new_cams)


# reference:
# https://github.com/colmap/colmap/blob/8e7093d22b324ce71c70d368d852f0ad91743808/src/colmap/sensor/models.h#L268C34-L268C34
def read_colmap_cam(cam_path, scale=1):
    # fx, fy, cx, cy, k1, k2, k3, k4
    cam = read_cameras_binary(cam_path)
    cam = cam[1]

    fx, fy, cx, cy = cam.params[:4]*scale
    foc = (fx + fy)*0.5
    fx = fy = foc
    int_mtx = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]]) 
    
    # cv2 model fx, fy, cx, cy, k1, k2, p1, p2
    return {"M1": int_mtx, "d1":cam.params[4:]}

def read_colmap_cam_param(cam_path):
    out = read_colmap_cam(cam_path, scale=1)

    return out["M1"], out["d1"]

def read_prosphesee_ecam_param(cam_path):
    
    with open(cam_path,"r") as f:
        data = json.load(f)
    
    return np.array(data["camera_matrix"]['data']).reshape(3,3), \
           np.array(data['distortion_coefficients']['data'][:4])


def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Undistorts an image using the given camera matrix and distortion coefficients.

    Parameters:
    - image: The distorted input image.
    - camera_matrix: The camera intrinsic matrix.
    - dist_coeffs: The distortion coefficients (k1, k2, p1, p2).

    Returns:
    - undistorted_image: The undistorted output image.
    """

    # Get the image size
    h, w = image.shape[:2]

    # Calculate the undistortion and rectification transformation map
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)

    # Remap the original image to the new undistorted image
    undist_img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # Crop the image to the ROI
    x, y, w, h = roi
    undist_img = undist_img[y:y+h, x:x+w]

    return undist_img


def poses_to_w2cs_hwf(poses):
    """
    takes LLFF poses and return to colmap c2w
    poses (3x5xn)
    """

    inv = np.concatenate([poses[:,1:2,:], poses[:,0:1,:], -poses[:,2:3,:], poses[:, 3:]],1)
    c2ws = inv[:3,:4,:]
    dummy = np.zeros((1,4,1))
    dummy[0,-1,0] = 1
    c2ws = np.concatenate([c2ws, np.tile(dummy, (1, 1, c2ws.shape[-1]))]).transpose(2,0,1)
    w2cs = np.linalg.inv(c2ws)

    ## w2cs, hwf
    return w2cs, poses[:,4:,:]



def w2cs_hwf_to_poses(w2c_mats, hwf):
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, hwf], 1)

    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    return poses


def to_homogenous(mtxs):
    """
    inputs:
        mtxs (np.ndarray): shape is (n, 3, 4)
    outputs:
        hom_mtx (np.ndarray): shape is (n, 4, 4)
    """
    if len(mtxs.shape) == 2:
        mtxs = mtxs[np.newaxis, ...]
    
    hom_mtx = np.concatenate([mtxs, np.tile(np.array([0,0,0,1])[np.newaxis, np.newaxis, :], (mtxs.shape[0], 1, 1))], 1)
    return hom_mtx.squeeze()

def inv_mtxs(mtxs):
    if mtxs.shape[-1] != 4 or mtxs.shape[-2] != 4:
        mtxs = to_homogenous(mtxs)
    
    return np.linalg.inv(mtxs)


def make_camera_json(ext_mtx, intr_mtx, dist, img_size):
    """
    input:
        ext_mtx (np.array): World to cam matrix - shape = 4x4
        intr_mtx (np.array): intrinsic matrix of camera - shape = 3x3

    return:
        nerfies.camera.Camera of the given mtx
    """
    R = ext_mtx[:3,:3]
    t = ext_mtx[:3,3]
    k1, k2, p1, p2 = dist[:4]
    coord = -t.T@R  

    cx, cy = intr_mtx[:2,2]
    cx, cy = float(cx), float(cy)

    new_camera = {
        "orientation":R.tolist(),
        "position":coord.tolist(),
        "focal_length":float(intr_mtx[0,0]),
        "pixel_aspect_ratio":1,
        "principal_point":[cx, cy],
        "radial_distortion":[k1, k2, 0],
        "tangential_distortion":[p1, p2],
        "skew":0,
        "image_size":img_size  ## (width, height) of camera
    }

    return new_camera


if __name__ == "__main__":
    colcam_path = "/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/calib_checker_recons/sparse/0/cameras.bin"
    img_f = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/black_seoul_b3_v3/colcam_set/rgb/1x/00000.png"
    K, D = read_colmap_cam_param(colcam_path)
    img = cv2.imread(img_f)
    undist = undistort_image(img, K, D)
    
