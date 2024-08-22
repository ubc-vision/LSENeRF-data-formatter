import numpy as np
from scipy.spatial.transform import Rotation
import os
import os.path as osp

def load_extrinsics_data(dataset_info_f, key="full_trajectory"):
    """
    data_info_f (str): a path to dataset_info.npz
    data extrinsics are cam2world

    output world2cam matrix
    """
    meta = np.load(dataset_info_f, allow_pickle=True)["meta"].item()  # cameras are cam2world
    traj_data = meta[key]

    cameras = []
    cam_ts = []
    read_quat_fn = lambda x : np.array([x["x"], x["y"], x["z"], x["w"]])
    read_trans_fn = lambda x : np.array([x["x"], x["y"], x["z"]])
    for frame_data in traj_data:
        pose = frame_data['cam']["pos"]
        quat, T = read_quat_fn(pose["q"]), read_trans_fn(pose["t"])
        R = Rotation.from_quat(quat).as_matrix()
        w2c_mtx = np.eye(4)
        w2c_mtx[:3,:3] = R
        w2c_mtx[:3, 3] = T
        w2c_mtx = np.linalg.inv(w2c_mtx)
        cameras.append(w2c_mtx)
        cam_ts.append(frame_data["ts"])
    
    return np.stack(cameras), np.array(cam_ts)


def load_ts(dataset_info_f, key="full_trajectory"):
    meta = np.load(dataset_info_f, allow_pickle=True)["meta"].item()  # cameras are cam2world
    traj_data = meta[key]

    cam_ts = [frame_data["ts"] for frame_data in traj_data]
    return np.array(cam_ts)



def load_intrinsics_data(dataset_info_f, as_mtx=True):
    """
    dataset_info_f (str): dataset_info.npz

    """
    
    meta = np.load(dataset_info_f, allow_pickle=True)["meta"].item()["meta"]

    cx, cy, fx, fy, k1, k2, k3, k4, p1, p2 = [meta[k] for k in ['cx', 'cy', 'fx', 'fy', 'k1', 'k2', 'k3', 'k4', 'p1', 'p2']]
    res_x, res_y = meta["res_x"], meta["res_y"]

    if as_mtx:
        intrxs = np.array([[fx, 0 , cx],
                        [0, fy, cy],
                        [0, 0, 1]])
    else:
        intrxs = np.array([fx, fy, cx, cy])
    dist = np.array([k1, k2, p1, p2, k3])    

    return intrxs.astype(np.float32), dist.astype(np.float32)

def load_camera_data(dataset_info_f, key="full_trajectory"):
    cameras, cam_ts = load_extrinsics_data(dataset_info_f, key=key)
    intrxs, dist = load_intrinsics_data(dataset_info_f)

    return cameras, cam_ts, intrxs, dist


def load_rig_extrnxs(file):
    """
    file (str): path to dataset_extrinsics.npz; camera to rig
    """
    ext = np.load(osp.join(file), allow_pickle=True)
    q = ext['q_rigcamera'].item()
    q_rc = np.array([q['x'], q['y'], q['z'], q['w']])
    t = ext['t_rigcamera'].item()
    t_rc = np.array([t['x'], t['y'], t['z']])
    T_rc = np.array([*t_rc, *q_rc])
    return T_rc


def warp_camera_frame(w2cs, Tre, Trc):
    """
    w2cs: world to camera of c
    Tre: camera, e, to rig
    Trc: camera, c, to rig

    return:
        w2es: world to event cameras
    """

    Tcw = w2cs
    Ter = inv_transform(Tre)
    Tec = apply_transform(Ter, Trc)
    Tew = np.stack(list(map(lambda x : apply_transform(Tec, x), Tcw)))

    return Tew



def to_evimo_fmt(inp):
    if len(inp.shape) == 2:
        quat = Rotation.from_matrix(inp[:3,:3]).as_quat()
        trans = inp[:3,3]
        inp = np.array([*trans, *quat])
    
    assert len(inp) == 7, "should be [coord, quat]"
    return inp



def apply_transform(T_cb, T_ba):
    T_cb, T_ba = to_evimo_fmt(T_cb), to_evimo_fmt(T_ba)
    R_ba = Rotation.from_quat(T_ba[3:7])
    t_ba = T_ba[0:3]

    R_cb = Rotation.from_quat(T_cb[3:7])
    t_cb = T_cb[0:3]

    R_ca = R_cb * R_ba
    t_ca = R_cb.as_matrix() @ t_ba.reshape(3,1) + t_cb.reshape(3,1)
    return np.concatenate([R_ca.as_matrix(), t_ca.reshape(3,1)], axis=1)
    

# Invert an SE(3) transform
# def inv_transform(T_ba):
#     T_ba = to_evimo_fmt(T_ba)
#     R_ba = Rotation.from_quat(T_ba[3:7])
#     t_ba = T_ba[0:3]

#     R_ab = R_ba.inv()
#     t_ab = -R_ba.inv().as_matrix() @ t_ba

#     return np.concatenate([R_ab.as_matrix(), t_ab.reshape(3,1)], axis=1)

def inv_transform(T_ba):
    T_ba = to_evimo_fmt(T_ba)
    mtx = np.eye(4)
    mtx[:3,:3] = Rotation.from_quat(T_ba[3:7]).as_matrix()
    mtx[:3,3] = T_ba[:3]

    return np.linalg.inv(mtx)[:4]


def scale_intrxs(intrxs:np.ndarray, scale:float):
    u = intrxs.copy()
    u[:2] = u[:2] * scale
    return u


def to_hom(mtxs):
    u = np.zeros((len(mtxs), 4, 4))
    u[:, 3, 3] = 1
    u[:,:3, :4] = mtxs[:,:3, :4]
    return u

def inv_cam_extrinsics(cam):
    """
    cam: (n, 3 or 4, 4) - camera extrinsics
    """

    if cam.shape[1] == 3:
        cam = np.concatenate([cam, np.array([[[0,0,0,1]]*cam.shape[0]]).reshape(cam.shape[0],1,4)], axis=1)

    inv_cam = np.linalg.inv(cam)
    return inv_cam


def w2cs_hwf_to_poses(w2c_mats, hwf):
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, hwf], 1)

    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    return poses

def build_poses_bounds(w2c, h, w, fx, bd_min=None, bd_max=None):
    """
    M: np.array (n, 4, 4) - w2c camera extrinsics
    h, w: int - height and width of image
    fx: float - focal length
    """

    # hwfs = np.stack([np.array([h, w, fx])]*len(M))[..., None]
    hwf = np.array([h, w, fx]).reshape(3, 1)
    hwfs = np.tile(hwf[:, np.newaxis], [1, 1, len(w2c)])
    poses = w2cs_hwf_to_poses(to_hom(w2c), hwfs).transpose(2, 0, 1).reshape(-1, 15)
    if (bd_min is not None) and (bd_max is not None):
        poses_bds = np.concatenate([poses, np.array([[bd_min, bd_max]]*len(poses))], axis=1)
    else:
        poses_bds = np.concatenate([poses, np.array([[0.01, 1]]*len(poses))], axis=1)
    return poses_bds