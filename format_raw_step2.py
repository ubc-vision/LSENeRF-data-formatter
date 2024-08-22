import numpy as np
import os.path as osp
import os
import glob
import cv2
import argparse
import glob
from tqdm import tqdm
import shutil
import json

from nerfies_camera import NerfiesCamera
from scene_managers import LLFFRGBManager, LLFFEVSManager



def make_nerfies_camera(ext_mtx, intr_mtx, dist, img_size):
    """
    input:
        ext_mtx (np.array): World to cam matrix - shape = 4x4
        intr_mtx (np.array): intrinsic matrix of camera - shape = 3x3
        img_size [h, w] (list/tupple): size of image

    return:
        nerfies.camera.Camera of the given mtx
    """
    R = ext_mtx[:3,:3]
    t = ext_mtx[:3,3]
    k1, k2, p1, p2 = dist
    coord = -R.T@t  
    h, w = img_size

    cx, cy = intr_mtx[:2,2].astype(int)

    new_camera = NerfiesCamera(
        orientation=R,
        position=coord,
        focal_length=intr_mtx[0,0],
        pixel_aspect_ratio=1,
        principal_point=np.array([cx, cy]),
        radial_distortion=(k1, k2, 0),
        tangential_distortion=(p1, p2),
        skew=0,
        image_size=np.array([w, h])  ## (width, height) of camera
    )

    return new_camera

def format_rgb_cameras(rgbScene:LLFFRGBManager, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(rgbScene)):
        M = rgbScene.get_extrnxs(i)
        K, dist = rgbScene.get_intrnxs()
        camera = make_nerfies_camera(M, K, dist, rgbScene.get_img_size())
        cam_json = camera.to_json()

        if rgbScene.meta.get("mid_cam_ts") is not None:
            cam_json["t"] = rgbScene.get_camera_t(i)

        with open(osp.join(save_dir, f"{i:05d}.json"), "w") as f:
            json.dump(cam_json, f, indent=2)


def format_full_cameras(rgbScene:LLFFRGBManager, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    rgb_cams = rgbScene.ori_w2cs.reshape(-1, 3, 4)
    for i in range(len(rgb_cams)):
        M = rgb_cams[i]
        K, dist = rgbScene.get_intrnxs()
        camera = make_nerfies_camera(M, K, dist, rgbScene.get_img_size())
        cam_json = camera.to_json()
        cam_json["t"] = rgbScene.meta["all_rgb_ts"][i]

        with open(osp.join(save_dir, f"{i:05d}.json"), "w") as f:
            json.dump(cam_json, f, indent=2)



def format_evs_cameras(evsScene:LLFFEVSManager, save_dir):
    """
    save_dir (str): directory to save the cameras, expect PATH/ecam_set
    """
    prev_dir = osp.join(save_dir, "prev_camera")
    next_dir = osp.join(save_dir, "next_camera")
    os.makedirs(prev_dir, exist_ok=True), os.makedirs(next_dir, exist_ok=True) 

    cam_idx = 0
    K, D = evsScene.get_intrnxs()
    n_frames, n_bins = evsScene.w2cs.shape[:2]
    for frame_idx in range(n_frames):
        for bin_idx in range(n_bins - 1):
            prev_cam  = make_nerfies_camera(evsScene.w2cs[frame_idx, bin_idx], K, D, evsScene.get_img_size())
            next_cam  = make_nerfies_camera(evsScene.w2cs[frame_idx, bin_idx+1], K, D, evsScene.get_img_size())
            
            # I don't know why I decided on 6d
            prev_cam_f, next_cam_f = osp.join(prev_dir, f"{cam_idx:06d}.json"), osp.join(next_dir, f"{cam_idx:06d}.json")
            prev_json, next_json = prev_cam.to_json(), next_cam.to_json()

            if evsScene.cam_t is not None:
                prev_t, next_t = evsScene.cam_t[frame_idx, bin_idx], evsScene.cam_t[frame_idx, bin_idx + 1]
                prev_json["t"], next_json["t"] = int(prev_t), int(next_t)


            with open(prev_cam_f, "w") as f:
                json.dump(prev_json, f, indent=2)

            with open(next_cam_f, "w") as f:
                json.dump(next_json, f, indent=2)
            
            cam_idx += 1


def copy_imgs_to_dir(rgbScene, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(len(rgbScene)), desc="copying rgb images"):
        img = rgbScene.get_img(i)
        cv2.imwrite(osp.join(save_dir, f"{i:05d}.png"), img)


def save_eimgs(evsScene, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    if evsScene.imgs.dtype == np.int8:
        eimgs = evsScene.imgs
    else:
        eimgs = evsScene.imgs.astype(np.int16)
        assert (np.abs(eimgs) < 127).all(), "can't format to int8!"
        eimgs = eimgs.astype(np.int8)

    save_f = osp.join(save_dir, "eimgs_1x.npy")
    np.save(save_f, eimgs)


def write_rgb_metadata(rgbScene, save_f):
    metadata = {"colmap_scale": rgbScene.get_colmap_scale()}

    n_bins = rgbScene.n_bins
    for i in range(len(rgbScene)):
        metadata[str(i).zfill(5)] = {"warp_id": i*(n_bins - 1) + n_bins//2,
                                     "appearance_id": i*(n_bins - 1) + n_bins//2,
                                     "camera_id": 0}
    
    with open(save_f, "w") as f:
        json.dump(metadata, f, indent=2)

def write_evs_metadata(evsScene, save_f):
    n_frames = len(evsScene.w2cs)* (evsScene.n_bins - 1)

    metadata = {"colmap_scale": evsScene.get_colmap_scale()}
    for i in range(n_frames):
        metadata[str(i).zfill(6)] = {"warp_id": i,
                                     "appearance_id": i,
                                     "camera_id": 0}
    
    with open(save_f, "w") as f:
        json.dump(metadata, f, indent=2)

def write_dataset(scene, save_f, n_digit, all=False):

    ids = [str(i).zfill(n_digit) for i in range(len(scene))]

    if all:
        dataset_json = {
            "count": len(ids),
            "num_exemplars": len(ids),
            "train_ids": ids,
            "val_ids": ids,
            "test_ids": ids
        }
        
    else:
        dataset_json = {
            "count": len(ids),
            "num_exemplars": len(ids),
            "train_ids": ids,
            "val_ids": [],
            "test_ids": []
        }

    with open(save_f, "w") as f:
        json.dump(dataset_json, f, indent=2)


def main(scene_dir, targ_dir=None, cam_only=False):
    if targ_dir is None:
        targ_dir = osp.join("/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data", osp.basename(scene_dir))
    colcam_dir = osp.join(targ_dir, "colcam_set")
    ecam_dir = osp.join(targ_dir, "ecam_set")

    rgbScene = LLFFRGBManager(scene_dir)

    if not cam_only:
        save_img_dir = osp.join(colcam_dir, "rgb", "1x")
        copy_imgs_to_dir(rgbScene, save_img_dir)

    save_rgb_cam_dir = osp.join(colcam_dir, "camera")
    format_rgb_cameras(rgbScene, save_rgb_cam_dir)

    rgb_metadata_f = osp.join(colcam_dir, "metadata.json")
    write_rgb_metadata(rgbScene, rgb_metadata_f)
    if rgbScene.meta.get("all_rgb_ts") is not None:
        save_rgb_cam_dir = osp.join(colcam_dir, "full_camera")
        format_full_cameras(rgbScene, save_rgb_cam_dir)

    rgb_dataset_f = osp.join(colcam_dir, "dataset.json")
    src_dataset_f = osp.join(scene_dir, "dataset.json")
    
    shutil.copy(src_dataset_f, rgb_dataset_f)


    evsScene = LLFFEVSManager(scene_dir, single_cam=False)

    if not cam_only:
        save_eimgs_dir = osp.join(ecam_dir, "eimgs")
        save_eimgs(evsScene, save_eimgs_dir)

    format_evs_cameras(evsScene, ecam_dir)

    evs_meta_f = osp.join(ecam_dir, "metadata.json")
    write_evs_metadata(evsScene, evs_meta_f)

    evs_dataset_f = osp.join(ecam_dir, "dataset.json")
    write_dataset(evsScene, evs_dataset_f, 6)

    rel_cam_f = osp.join(scene_dir, "rel_cam.json")
    dst_f = osp.join(targ_dir, "rel_cam.json")
    if not osp.exists(dst_f) and osp.exists(rel_cam_f):
        shutil.copy(rel_cam_f, dst_f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="source directory of the scene")
    parser.add_argument("--targ_dir", type=str, help="target directory to save the formatted data")
    parser.add_argument("--cam_only", action="store_true", help="only format the cameras", default=False)
    args = parser.parse_args()


    main(args.src_dir, args.targ_dir, args.cam_only)
