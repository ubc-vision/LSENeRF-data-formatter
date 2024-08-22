import numpy as np
import glob
import os.path as osp
import json
import cv2

from camera_utils import poses_to_w2cs_hwf
from utils import parallel_map
import colmap_read_model as read_model
from utils import load_json_intr, load_json_extr

def load_poses_bounds(path):
    poses_arr = np.load(path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    hwf = poses[:, 4:, :].squeeze()

    # shapes = (3,5,n), (2,n), (3,n)
    return poses, bds, hwf


class LLFFRGBManager:

    def __init__(self, data_dir, single_cam=True):
        self.data_dir = data_dir
        self.single_cam = single_cam
        self.img_fs = sorted(glob.glob(osp.join(data_dir, "images", "*")))
        
        self.mid_rgb_poses_bounds_f = osp.join(data_dir, "mid_rgb_poses_bounds.npy")
        rgb_poses_bounds_f = osp.join(data_dir, "rgb_poses_bounds.npy") if not osp.exists(self.mid_rgb_poses_bounds_f) else self.mid_rgb_poses_bounds_f
        if osp.exists(rgb_poses_bounds_f):
            self.poses, self.bds, self.hwf = load_poses_bounds(rgb_poses_bounds_f)
        else:
            self.poses, self.bds, self.hwf = load_poses_bounds(osp.join(data_dir, "poses_bounds.npy"))
        self.w2cs, _ = poses_to_w2cs_hwf(self.poses)
        self.w2cs = self.w2cs[:, :3, :4]
        self.hwf = self.hwf[..., 0]

        self.n_bins = 5
        meta_f = osp.join(data_dir, "metadata.json")
        self.meta = None
        if osp.exists(meta_f):
            with open(meta_f, "r") as f:
                self.meta = json.load(f)
            
            self.n_bins = self.meta["n_bins"]
        
        self.ori_w2cs = self.w2cs
        self.w2cs = self.w2cs.reshape(-1, self.n_bins, 3, 4) if not osp.exists(self.mid_rgb_poses_bounds_f) else self.w2cs
        if single_cam and (not osp.exists(self.mid_rgb_poses_bounds_f)):
            self.w2cs = self.w2cs[:, self.n_bins//2, :, :]
        
        self.img_size = self.get_img(0).shape[:2]

    def __len__(self):
        if hasattr(self, "meta") and self.meta.get("mid_cam_ts") is not None:
            return len(self.meta.get("mid_cam_ts"))

        return len(self.img_fs)
        # return min(len(self.imgs), len(self.w2cs))
    
    # REQUIRED
    def get_img_size(self):
        return self.img_size # (h, w)
    
    # REQUIRED
    def get_img_f(self, idx):
        return self.img_fs[idx]

    # REQUIRED
    def get_img(self, idx):
        return cv2.imread(self.img_fs[idx])
    
    # REQUIRED
    def get_extrnxs(self, idx):
        return self.w2cs[idx]

    def get_camera_t(self, idx):
        return self.meta["mid_cam_ts"][idx]
    
    def get_colmap_scale(self):
        return self.meta.get("colmap_scale")

    # REQUIRED
    def get_intrnxs(self):
        if self.meta is None or self.meta.get("rgb_K") is None:
            return np.array([[self.hwf[2], 0, self.hwf[1]/2],
                            [0, self.hwf[2], self.hwf[0]/2],
                            [0,           0,           1]]), np.zeros(4)
        else:
            return np.array([[self.hwf[2], 0, self.meta["rgb_K"][2]],
                             [0, self.hwf[2], self.meta["rgb_K"][3]],
                             [0,           0,           1          ]]), np.zeros(4)


class LLFFEVSManager(LLFFRGBManager):
    def __init__(self, data_dir, single_cam=True):
        """
        data_dir (str): path to scene
        single_cam (bool): cams are saved as (n, bin_size, *), if true, will select a single eimg and camera with the shape (n, *)
        """
        self.data_dir = data_dir
        self.single_cam = single_cam
        # self.imgs = torch.load(osp.join(self.data_dir, "events.pt")).numpy()
        self.imgs = np.load(osp.join(data_dir, "events.npy"))

        evs_poses_bounds_f = osp.join(data_dir, "evs_poses_bounds.npy")
        if osp.exists(evs_poses_bounds_f):
            self.poses, self.bds, self.hwf = load_poses_bounds(evs_poses_bounds_f)
        else:
            self.poses, self.bds, self.hwf = load_poses_bounds(osp.join(data_dir, "poses_bounds.npy"))
        self.w2cs, _ = poses_to_w2cs_hwf(self.poses)
        self.w2cs = self.w2cs[:, :3, :4]
        self.hwf = self.hwf[..., 0]


        self.n_bins = 5
        meta_f = osp.join(data_dir, "metadata.json")
        self.meta = None
        if osp.exists(meta_f):
            with open(meta_f, "r") as f:
                self.meta = json.load(f)
            self.n_bins = self.meta["n_bins"]
        
        h, w = self.hwf[:2].astype(int)
        self.img_size = (h, w)

        self.w2cs = self.w2cs.reshape(-1, self.n_bins, 3, 4)
        self.imgs = self.imgs.reshape(-1, self.n_bins - 1, h, w)

        if single_cam:
            self.w2cs = self.w2cs[:, self.n_bins//2, :, :]
            self.imgs = self.imgs[:, self.n_bins//2, :, :]
        else:
            self.imgs = self.imgs.reshape(-1, h, w)
        
        ev_t = self.meta.get("ev_cam_ts")
        self.cam_t = np.array(self.meta["ev_cam_ts"]).reshape(-1, self.n_bins) if not (ev_t is None) else None

    def __len__(self):
        return min(len(self.imgs), len(self.w2cs))

    def get_img_f(self, idx):
        assert 0, "Not implemented"
    
    def get_img(self, idx):
        return np.stack([(self.imgs[idx] != 0).astype(np.uint8) * 255]*3, axis=-1)

    def get_intrnxs(self):
        if self.meta is None or self.meta.get("evs_K") is None:
            return super().get_intrnxs()
        
        return np.array([[self.hwf[2], 0, self.meta["evs_K"][2]],
                         [0, self.hwf[2], self.meta["evs_K"][3]],
                         [0,           0,           1]]), np.zeros(4)


class ColmapCamera:

    def __init__(self, cam_f):
        self.col_cam = read_model.read_cameras_binary(cam_f)[1]
        self.w = self.col_cam.width
        self.h = self.col_cam.height
        
        self.fx, self.fy, self.cx, self.cy = self.col_cam.params[:4]
        self.intrxs = np.array([[self.fx,     0 , self.cx],
                                [0,      self.fy, self.cy],
                                [0,            0,       1]])
        self.k1, self.k2, self.p1, self.p2 = self.col_cam.params[4:]

    def get_dist_coeffs(self):
        return np.array([self.k1, self.k2, self.p1, self.p2])

class ColmapSceneManager:
    """
    Colmap scene manager
    """
    def __init__(self, colmap_dir):
        """
        colmap_dir = xxx_recons/
                         /images
                         /sparse
        """
        self.colmap_dir = colmap_dir
        self.img_dir = osp.join(colmap_dir, "images")

        self.images = read_model.read_images_binary(osp.join(self.colmap_dir, "sparse/0/images.bin"))
        self.camera = ColmapCamera(osp.join(self.colmap_dir, "sparse/0", "cameras.bin"))
        self.img_fs = self.get_all_registered_img_fs()
        self.chosen_points = None
        self.pnts_2d=None

        self.max_images_id = max(list(self.images.keys()))

    def set_sample_method(self, method):
        self.sample_method = method
        self.sample_pnt_fnc = self.sample_pnt_fnc_dic[self.sample_method]
    
    @property
    def image_ids(self):
        return sorted(self.images.keys())
    
    def load_image(self, img_idx):
        img_cam = self.images[img_idx]
        img_name = img_cam.name # get img_name from img_cam
        img_f = osp.join(self.img_dir, img_name)
        return cv2.imread(img_f)



    def __len__(self):
        return len(self.images)


    def get_extrnxs(self, img_idx=None, img_obj=None):
        """
        img_idx: colmap index
        """
        if img_idx < 0:
            img_idx = self.max_images_id + img_idx
            
        if img_idx is not None:
            img_obj = self.images[img_idx]
        
        R = img_obj.qvec2rotmat()
        t = img_obj.tvec[..., None]
        mtx = np.concatenate([R,t], axis=-1)
        dummy = np.zeros((1,4))
        dummy[0,-1] = 1
        mtx = np.concatenate([mtx, dummy], axis=0)
        return mtx

    def get_all_extrnxs(self):
        keys = sorted(list(self.images.keys()))
        return [self.get_extrnxs(e) for e in keys]

    def get_intrnxs(self):
        return self.camera.intrxs, self.camera.get_dist_coeffs()

    def get_img(self, img_idx):
        """
        img_idx = colmap_img_idx
        """
        if img_idx > 0: 
            print("colmap img is 1 indexed! Adding 1 to idx")
            img_idx += 1
        return cv2.imread(self.get_img_f(img_idx))

    def get_all_imgs(self, read_img_fn=cv2.imread):
        return parallel_map(lambda f : read_img_fn(f), self.img_fs)

    def get_img_f(self, img_idx):
        if img_idx == 0:
            print("image_idx needs to be >=1, adding 1 as fix")
            img_idx += 1
        if img_idx < 0:
            img_idx = self.max_images_id + img_idx
        return osp.join(self.img_dir, self.images[img_idx].name)

    
    def get_all_registered_img_fs(self):
        names = sorted([v.name for k, v in self.images.items()])
        return [osp.join(self.img_dir, name) for name in names]

    def get_img_id(self, img_idx):
        img_f = self.get_img_f(img_idx)
        return osp.basename(img_f).split(".")[0]
    

    def get_found_cond(self, n_size):
        """
        return condition showing which image was registered
        """
        keys = np.array(sorted(list(int(k) for k in self.images.keys()))) - 1 # subtract 1 since colmap idx starts at 1
        keys = keys[keys < n_size].astype(np.int32)
        cond = np.zeros(n_size, dtype=bool)
        cond[keys] = True
        return cond


class ColcamSceneManager:
    """
    scene manager for formatted colcam set (rgb)
    """

    def __init__(self, data_dir):
        """
        expect: xxx/colcam_set
        """
        self.data_dir = data_dir
        self.dataset_json_f = osp.join(self.data_dir, "dataset.json")
        self.img_fs = sorted(glob.glob(osp.join(data_dir, "rgb", "1x", "*.png")))

        with open(self.dataset_json_f, "r") as f:
            self.dataset = json.load(f)
            if self.dataset.get("ids") is None:
                self.img_ids = list(range(len(self.img_fs)))
            else:
                self.img_ids = [int(e) for e in self.dataset["ids"]]
            self.train_ids = [int(e) for e in self.dataset["train_ids"]]
            self.val_ids = [int(e) for e in self.dataset["val_ids"]]
    

        self.img_fs = [self.img_fs[e] for e in self.img_ids if e < len(self.img_fs)]

        self.cam_fs = sorted(glob.glob(osp.join(self.data_dir, "camera", "*.json")))
        self.img_shape = self.get_img(0).shape[:2]
        self.ts = self.load_ts(self.cam_fs)
        self.image_ids = list(range(len(self.img_fs)))
    
    def get_img(self, idx):
        return cv2.imread(self.img_fs[idx])

    def load_image(self, idx):
        return self.get_img(idx)

    def get_intrnxs(self):
        return load_json_intr(self.cam_fs[0])
    
    def load_ts(self, cam_fs):
        ts = []
        for i, cam_f in enumerate(cam_fs):
            with open(cam_f, "r") as f:
                data = json.load(f)
                if data.get("t") is None:
                    print("WARNING: NO TIMESTAMP AVAILABLE")
                    return None

                try:
                    ts.append(data["t"])
                except Exception as e:
                    print(e)
                    print(osp.basename(cam_f), f"does not have t. frame: {i + 1}/{len(cam_fs)}. Replacing with last t")
                    ts.append(ts[-1])
        
        return np.array(ts)

    def get_extrnxs(self, idx):
        """
        returns world to cam
        """
        return load_json_extr(self.cam_fs[idx])
    
    def get_all_extrnxs(self):
        return np.stack([load_json_extr(e) for e in self.cam_fs[:self.__len__()]])
    
    def get_val_extrnxs(self):
        return np.stack([load_json_extr(self.cam_fs[e]) for e in self.val_ids])

    def __len__(self):
        return min(len(self.cam_fs), len(self.img_fs))
    
    def get_train_ts(self):
        return self.ts[sorted(list(map(int, self.train_ids)))]


class EcamSceneManager(ColcamSceneManager):
    """
    scene manager for formatted ecam set (evs)
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cam_fs = sorted(glob.glob(osp.join(self.data_dir, "camera", "*.json")))
        self.eimgs = np.load(osp.join(self.data_dir, "eimgs", "eimgs_1x.npy"), "r")

        self.dataset_json_f = osp.join(self.data_dir, "dataset.json")
        prev_cam_dir = osp.join(self.data_dir, "prev_camera")
        if osp.exists(prev_cam_dir):
            self.prev_cam_fs = sorted(glob.glob(osp.join(prev_cam_dir, "*.json")))
            self.next_cam_fs = sorted(glob.glob(osp.join(self.data_dir, "next_camera", "*.json")))
            self.prev_ts = self.load_ts(self.prev_cam_fs)
            self.next_ts = self.load_ts(self.next_cam_fs)
            self.ts = self.prev_ts
        else:
            self.ts = self.load_ts(self.cam_fs)  # this one is slightly meaningless
        
        if not osp.exists(osp.join(self.data_dir, "camera")):
            self.cam_fs = self.prev_cam_fs

    def __len__(self):
        return min(len(self.cam_fs), len(self.eimgs))

    def get_img(self, idx):
        img = np.stack([(self.eimgs[idx] != 0).astype(np.uint8) * 255]*3, axis=-1)
        return img