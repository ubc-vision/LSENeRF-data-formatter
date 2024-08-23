import glob
import os.path as osp
import os
import numpy as np
import cv2
import json
from tqdm import tqdm
import argparse
import shutil


from evimo_camera_utils import (load_camera_data, 
                                scale_intrxs, 
                                load_ts, 
                                load_rig_extrnxs, 
                                warp_camera_frame,
                                inv_transform, apply_transform)
from camera_utils import create_interpolated_cams
from utils import (ev_to_eimg, 
                   parallel_map, 
                   create_and_write_camera_extrinsics, 
                   load_evimo_frame_data)

from evimo_event_buffer import EvimoEventBuffer as EventBuffer

np.set_printoptions(precision=3)

avail_scenes = sorted([osp.basename(p) for p in glob.glob("evimov2_dataset/*")])

class ColsetFormatter:
    def __init__(self, src_dir, targ_dir, n_bin = 4):
        self.scale = 0.5  # image scale to reduce by; 0.5 is downsample by 2; RGB image is 4x event camera resolution

        self.n_bin = n_bin
        self.src_dir = src_dir
        self.dataset_info_f = osp.join(self.src_dir, "dataset_info.npz")
        self.img_npz_f = osp.join(self.src_dir, "dataset_classical.npz")
        self.rig_f = osp.join(self.src_dir, "dataset_extrinsics.npz")

        self.imgs, self.classical_ids = load_evimo_frame_data(self.img_npz_f, ret_id=True)
        self.ori_img_size = self.imgs[0].shape[:2]
        self._init_camera_data()
        
        self.targ_dir = targ_dir
        self.targ_rgb_dir = osp.join(self.targ_dir, "rgb", "1x")
        self.targ_camera_dir = osp.join(self.targ_dir, "camera")
        self.targ_dirs = [self.targ_dir, self.targ_rgb_dir, self.targ_camera_dir]
        self._init_targdirs()
    
    def _init_targdirs(self):
        for d in self.targ_dirs:
            os.makedirs(d, exist_ok=True)
    
    def _init_camera_data(self):
        self.src_extrxs, self.src_ts, self.src_K, self.src_D = load_camera_data(self.dataset_info_f)
        im_h, im_w = self.ori_img_size
        self.undist_K, self.roi = cv2.getOptimalNewCameraMatrix(
            self.src_K, self.src_D, (im_w, im_h), 1, (im_w, im_h)
        )

        self.app_undist_K = np.copy(self.undist_K)  # K used to undistort

        x, y, w, h = self.roi
        self.undist_K[0, 2] = self.undist_K[0, 2] - x
        self.undist_K[1, 2] = self.undist_K[1, 2] - y
        self.save_undist_K = scale_intrxs(self.undist_K, self.scale)

        self.targ_img_size = (round(h*self.scale), round(w*self.scale))

        # assume camera transform to be:
        # 1) undistort
        # 2) scale

        ## rgb camera to rig
        self.Trc = load_rig_extrnxs(self.rig_f)


    def transform_and_save_img(self, imgs=None, scale = 0.5, targ_dir = None):
        # NOTE: ensure transform is consistent with intrnxs in _init_camera_data
        if targ_dir is None:
            targ_dir = self.targ_rgb_dir

        if imgs is None:
            imgs = self.imgs

        im_h, im_w = imgs[0].shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.src_K, self.src_D, None, self.app_undist_K, (im_w, im_h), cv2.CV_32FC1
        )
        x, y, w, h = self.roi

        new_size = (round(w*scale), round(h*scale))  # should be the same as self.targ_img_size
        undist = lambda img : cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        scale_img = lambda img : cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)  

        
        def transform_and_save_fn(inp):
            idx_str, img = inp
            undist_img = undist(img)[y:y+h, x:x+w]
            scaled_img = scale_img(undist_img)
            save_f = osp.join(targ_dir, idx_str + ".png")
            cv2.imwrite(save_f, scaled_img)
        
        self.img_ids = [str(idx).zfill(5) for idx in list(range(len(imgs)))]
        parallel_map(transform_and_save_fn, list(zip(self.img_ids, imgs)), show_pbar=True, desc="saving imgs")
        self.img_ids = self.img_ids[:-1] # remove last image to avoid bugs

    

    def interp_and_save_cameras(self):
        self.rgb_ts = load_ts(self.dataset_info_f, key="frames")
        interp_extrxs = create_interpolated_cams(self.rgb_ts, self.src_ts, self.src_extrxs)

        h, w = self.targ_img_size
        save_img_size = (w, h)
        create_and_write_camera_extrinsics(self.targ_camera_dir, interp_extrxs, 
                                           self.rgb_ts * 1e6, # NOTE: convert to micro seconds
                                           self.save_undist_K, 
                                           (0,0,0,0,0), # NOTE: no distortion
                                           save_img_size)

    def create_and_write_metadata(self):

        meta = {"colmap_scale": 1}
        appearance_ids = self.n_bin//2 + np.arange(len(self.img_ids)) * self.n_bin
        for img_id, app_id in zip(self.img_ids, appearance_ids):
            meta[img_id] = {"warp_id": int(app_id),
                            "appearance_id": int(app_id),
                            "camera_id": 0}
        targ_meta_f = osp.join(self.targ_dir, "metadata.json")
        with open(targ_meta_f, "w") as f:
            json.dump(meta, f, indent=2)


    def copy_dataset_json(self):
        assert osp.basename(self.src_dir) in avail_scenes, f"scene {osp.basename(self.src_D)} not in available scenes"
        src_dataset_json = osp.join("evimov2_dataset", osp.basename(self.src_dir), "dataset.json")
        targ_dataset_json = osp.join(self.targ_dir, "dataset.json")
        
        if not osp.exists(targ_dataset_json):
            shutil.copy(src_dataset_json, targ_dataset_json)


    def format_colcam_set(self):
        self.transform_and_save_img(self.imgs,scale=self.scale)
        self.interp_and_save_cameras()
        self.create_and_write_metadata()
        self.copy_dataset_json()
    
    def warp_to_ecam(self, Tre):
        """
        Tre: event camera to rig
        """
        return warp_camera_frame(self.src_extrxs, Tre, self.Trc)


class EcamsetFormatter:
    def __init__(self, src_dir, targ_dir, n_bin = 4, exp_t=29983*1e-6):  # 30fps
        """
        exp_t (float): exposure time in seconds
        """
        self.exp_t = exp_t
        self.n_bin = n_bin
        self.src_dir = src_dir
        self.dataset_info_f = osp.join(self.src_dir, "dataset_info.npz")
        self.rig_f = osp.join(self.src_dir, "dataset_extrinsics.npz")

        self.ori_eimg_size = (480, 640)  # (h, w)
        self._init_camera_data()

        self.evs = EventBuffer(self.src_dir)

        self.targ_dir = targ_dir
        self.eimgs_dir = osp.join(self.targ_dir, "eimgs")
        self.prev_cam_dir = osp.join(self.targ_dir, "prev_camera")
        self.next_cam_dir = osp.join(self.targ_dir, "next_camera")
        self.targ_dirs = [self.targ_dir, self.eimgs_dir, self.prev_cam_dir, self.next_cam_dir]
        self._init_targdirs()
    
    def _init_targdirs(self):
        for d in self.targ_dirs:
            os.makedirs(d, exist_ok=True)
    
    def _init_camera_data(self):
        # NOTE: undistort only
        _, _, self.src_K, self.src_D = load_camera_data(self.dataset_info_f)
        im_h, im_w = self.ori_eimg_size
        self.undist_K, self.roi = cv2.getOptimalNewCameraMatrix(
            self.src_K, self.src_D, (im_w, im_h), 1, (im_w, im_h)
        )

        self.app_undist_K = np.copy(self.undist_K)  # K used to undistort
        x, y, w, h = self.roi
        self.undist_K[0, 2] = self.undist_K[0, 2] - x
        self.undist_K[1, 2] = self.undist_K[1, 2] - y
        self.targ_img_shape = (h, w)

        self.Tre = load_rig_extrnxs(self.rig_f)

    def set_src_extrxs_ts(self, src_extrxs, src_ts):
        self.src_extrxs = src_extrxs
        self.src_ts = src_ts
    
    def create_interp_ts(self, center_ts, exp_t=None):
        if exp_t is None:
            exp_t = self.exp_t
        
        start_ts = center_ts - exp_t*0.5
        delta_t = exp_t/(self.n_bin - 1)
        t_steps = np.arange(self.n_bin) * delta_t
        cam_ts = start_ts[..., None] + t_steps[None]
        cam_ts = cam_ts.reshape(-1)
        return cam_ts
        
    def create_eimgs(self, interp_ts, do_flatten=True):
        im_h, im_w = self.ori_eimg_size
        interp_ts = interp_ts.reshape(-1, self.n_bin)
        eimgs = np.zeros((len(interp_ts), self.n_bin - 1, *self.targ_img_shape), dtype=np.int8)

        x, y, w, h = self.roi
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.src_K, self.src_D, None, self.app_undist_K, (im_w, im_h), cv2.CV_32FC1
        )

        undist_fn = lambda img : cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST)[y:y+h, x:x+w]
        frame_cnter = 0
        for i in tqdm(range(len(eimgs)), desc="creating eimgs"):
            frame_cnter += 1
            prev_t = 0
            for bi in range(self.n_bin - 1):
                st_t, end_t = interp_ts[i, bi], interp_ts[i, bi + 1]
                is_valid = self.evs.validate_time(st_t)

                if not is_valid:
                    break

                ts, xs, ys, ps = self.evs.retrieve_data(st_t, end_t)
                eimg = ev_to_eimg(xs, ys, ps, img_size=(480, 640))

                pos_eimg, neg_eimg = np.copy(eimg), np.copy(eimg)
                pos_cond = eimg > 0
                pos_eimg[~pos_cond] = 0
                neg_eimg[pos_cond] = 0
                pos_eimg, neg_eimg = pos_eimg.astype(np.uint8), np.abs(neg_eimg).astype(np.uint8)
                pos_re, neg_re = undist_fn(pos_eimg), undist_fn(neg_eimg)
                
                eimgs[i, bi] = pos_re.astype(np.int8) + neg_re.astype(np.int8) * -1
                
                self.evs.drop_cache_by_t(prev_t)
                prev_t = st_t
            
            if not is_valid:
                break
        
        # self.n_eimgs = frame_cnter
        # return eimgs[:frame_cnter].reshape(-1, *self.targ_img_size)
        if do_flatten:
            eimgs = eimgs.reshape(-1, *self.targ_img_shape)
        else:
            print("WARNING, self.n_eimgs is garbage now; ok if formatting to e2nerf")
        self.n_eimgs = len(eimgs)
        return eimgs

    def create_and_save_eimg(self, interp_ts, save_f=None):
        eimgs = self.create_eimgs(interp_ts)
        save_f = osp.join(self.eimgs_dir, "eimgs_1x.npy") if save_f is None else save_f
        np.save(save_f, eimgs)
    
    def create_and_save_cameras(self, interp_ts):
        interp_extrxs = create_interpolated_cams(interp_ts, self.src_ts, self.src_extrxs)
        prev_extrnxs, next_extrnxs = [], []

        interp_extrxs = interp_extrxs.reshape(-1, self.n_bin, *interp_extrxs.shape[-2:])
        for i in range(len(interp_extrxs)):
            for j in range(self.n_bin - 1):
                prev_extrnxs.append(interp_extrxs[i, j])
                next_extrnxs.append(interp_extrxs[i, j + 1])


        h, w = self.targ_img_shape
        save_img_size = (w, h)
        create_and_write_camera_extrinsics(self.prev_cam_dir, prev_extrnxs, 
                                           interp_ts * 1e6, # NOTE: convert to micro seconds
                                           self.undist_K, 
                                           (0,0,0,0,0), # NOTE: no distortion
                                           save_img_size,
                                           n_zeros=6)
        
        create_and_write_camera_extrinsics(self.next_cam_dir, next_extrnxs, 
                                           interp_ts * 1e6, # NOTE: convert to micro seconds
                                           self.undist_K, 
                                           (0,0,0,0,0), # NOTE: no distortion 
                                           save_img_size,
                                           n_zeros=6)

    def create_and_write_metadata(self):
        meta = {"colmap_scale": 1}

        for i in range(self.n_eimgs):
            meta[str(i).zfill(6)] = {"warp_id": i,
                                     "appearance_id": i,
                                     "camera_id": 0}
        
        with open(osp.join(self.targ_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def copy_dataset_json(self):
        assert osp.basename(self.src_dir) in avail_scenes, f"scene {osp.basename(self.src_D)} not in available scenes"
        src_dataset_json = osp.join("evimov2_dataset", osp.basename(self.src_dir), "dataset.json")
        targ_dataset_json = osp.join(self.targ_dir, "dataset.json")
        
        if not osp.exists(targ_dataset_json):
            shutil.copy(src_dataset_json, targ_dataset_json)
    
    def format_ecamset(self, src_extrxs, src_ts, rgb_ts):
        self.set_src_extrxs_ts(src_extrxs, src_ts)
        interp_ts = self.create_interp_ts(rgb_ts)

        self.create_and_save_eimg(interp_ts)
        self.create_and_save_cameras(interp_ts)
        self.create_and_write_metadata()
        # self.copy_dataset_json()


def create_and_save_relcam(Trc, Tre, targ_dir):
    Ter = inv_transform(Tre)
    Tec = apply_transform(Ter, Trc)
    R = Tec[:3, :3]
    T = Tec[:3, 3:]

    relcam_json = {
        "R": R.tolist(),
        "T": T.tolist()
    }

    with open(osp.join(targ_dir, "rel_cam.json"), "w") as f:
        json.dump(relcam_json, f, indent=2)
    


def create_ednerf_v2(src_rgb_dir, src_evs_dir, targ_dir, bin_dt):
    
    n_bin=int(np.ceil(29983*1e-6/bin_dt)) + 1

    col_targ_dir = osp.join(targ_dir, "colcam_set")
    evs_targ_dir = osp.join(targ_dir, "ecam_set")

    col_formatter = ColsetFormatter(src_rgb_dir, col_targ_dir, n_bin=n_bin)
    col_formatter.format_colcam_set()

    evs_formatter = EcamsetFormatter(src_evs_dir, evs_targ_dir, n_bin=n_bin)
    ecam_extrnsics = col_formatter.warp_to_ecam(evs_formatter.Tre)
    evs_formatter.format_ecamset(ecam_extrnsics, 
                                 col_formatter.src_ts,
                                 col_formatter.rgb_ts)

    create_and_save_relcam(col_formatter.Trc, evs_formatter.Tre, targ_dir)


def create_full_camera_traj(src_rgb_dir, targ_dir):
    os.makedirs(targ_dir, exist_ok=True)

    col_formatter = ColsetFormatter(src_rgb_dir, targ_dir)
    src_ts, extrxs = col_formatter.src_ts, col_formatter.src_extrxs
    h, w = col_formatter.ori_img_size
    create_and_write_camera_extrinsics(
        targ_dir, extrxs, 
        src_ts * 1e6,
        col_formatter.src_K, 
        col_formatter.src_D,
        (w, h),
        n_zeros=6
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_src_dir", help="path to npz rgb data; see provided default path for example", 
                                         default="evimo2_v2_data/npz/flea3_7/sfm/train/scene8_01_000000")
    
    parser.add_argument("--evs_src_dir", help="path to npz evs data; see provided default path for example",
                                         default="evimo2_v2_data/npz/samsung_mono/sfm/train/scene8_01_000000")
    
    parser.add_argument("--out_dir", help="path to save formatted data")
    parser.add_argument("--bin_dt", help="BII time in seconds", type=float, default=5000*1e-6)
    args = parser.parse_args()
    
    rgb_src_dir = args.rgb_src_dir
    evs_src_dir = args.evs_src_dir
    targ_dir = args.out_dir
    # targ_dir = "debug"

    create_ednerf_v2(
        rgb_src_dir,
        evs_src_dir,
        targ_dir,
        bin_dt=args.bin_dt
    )

    create_full_camera_traj(
        rgb_src_dir,
        osp.join(targ_dir, "full_camera")
    )