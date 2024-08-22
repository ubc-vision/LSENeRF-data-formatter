from concurrent import futures
from tqdm import tqdm
import numpy as np
import os
import bisect
import h5py
import json

import colmap_read_model as read_model

class EventBuffer:
    def __init__(self, ev_f) -> None:
        self.ev_f = ev_f
        self.x_f, self.y_f, self.p_f, self.t_f = self.load_events(self.ev_f)

        self.fs = [self.x_f, self.y_f, self.p_f, self.t_f]

        self.n_retrieve = 5000000
        self._init_cache(0)


    def _init_cache(self, idx=0):
        self.x_cache = np.array([self.x_f[idx]])
        self.y_cache = np.array([self.y_f[idx]])
        self.t_cache = np.array([self.t_f[idx]])
        self.p_cache = np.array([self.p_f[idx]])

        self.caches = [self.x_cache, self.y_cache, self.t_cache, self.p_cache]

        self.curr_pnter = idx + 1
    
    def clear_cache(self):
        self.x_cache = np.array([])
        self.y_cache = np.array([])
        self.t_cache = np.array([])
        self.p_cache = np.array([])

        self.curr_pnter = np.nan # points at no where
    
    def load_events(self, ev_f):
        self.f = h5py.File(ev_f, "r")
        x_f = self.f["x"]
        y_f = self.f["y"]
        p_f = self.f["p"]
        t_f = self.f["t"]

        return x_f, y_f, p_f, t_f

    
    def update_cache(self):
        
        rx, ry, rp, rt = [e[self.curr_pnter:self.curr_pnter + self.n_retrieve] for e in self.fs]
        self.x_cache = np.concatenate([self.x_cache, rx])
        self.y_cache = np.concatenate([self.y_cache, ry])
        self.p_cache = np.concatenate([self.p_cache, rp])
        self.t_cache = np.concatenate([self.t_cache, rt])
        
        self.curr_pnter = min(len(self.t_f), self.curr_pnter + self.n_retrieve)

    def drop_cache_by_cond(self, cond):
        self.x_cache = self.x_cache[cond]
        self.y_cache = self.y_cache[cond]
        self.p_cache = self.p_cache[cond]
        self.t_cache = self.t_cache[cond]

    def retrieve_data(self, st_t, end_t, is_far=False):
        if (self.t_cache[0] > st_t) or is_far:
            ## if st_t already out of range
            idx = bisect.bisect(self.t_f, st_t)
            idx = idx if ((st_t == self.t_f[idx]) or st_t <= self.t_f[0]) else idx - 1

            assert idx >= 0, f"{st_t} not found!!"

            self._init_cache(idx)


        while (self.curr_pnter < len(self.t_f)) and (self.t_cache[-1] <= end_t):
            self.update_cache()
        
        ret_cond = ( st_t<= self.t_cache) & (self.t_cache <= end_t)
        ret_data = [self.t_cache[ret_cond], self.x_cache[ret_cond], 
                    self.y_cache[ret_cond], self.p_cache[ret_cond]]
        self.drop_cache_by_cond(~ret_cond)

        return ret_data
    
    def drop_cache_by_t(self, t):
        cond = self.t_cache >= t
        self.drop_cache_by_cond(cond)

    def valid_time(self, st_t):
        return st_t < self.t_f[-1]

def read_rel_cam(cam_rel_path):
    with open(cam_rel_path, "r") as f:
        data = json.load(f)
    
    for k, v in data.items():
        data[k] = np.array(v)
    
    return data

def ev_to_eimg(x, y, p, e_thresh=0.15, img_size = None):
    """
    input:
        evs (np.array [type (t, x, y, p)]): events such that t in [t_st, t_st + time_delta]
        img_size (tuple [int, int]): image size in (h,w)
    return:
        event_img (np.array): of shape (h, w)
    """
    
    if img_size is None:
        h, w = 720, 1280
    else:
        h, w = img_size
    

    pos_p = p==1
    neg_p = p==0

    e_img = np.zeros((h,w), dtype=np.int32)
    
    np.add.at(e_img, (y[pos_p], x[pos_p]), 1)
    np.add.at(e_img, (y[neg_p], x[neg_p]), -1)
    
    assert np.abs(e_img).max() < np.iinfo(np.int8).max, "type needs to be bigger"

    return e_img.astype(np.int8)


## modify from llff
def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = sorted([imdata[k].name for k in imdata])
    print( 'Images #', len(names))
    perm = np.argsort(names)
    ks = sorted(list(imdata.keys()))
    for k in ks:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm

## modify from llff
def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if ind >= len(cams):
                continue

            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    close_depths, inf_depths = [], []
    for i in perm:
        if i >= len(cams):
            continue
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        if len(zs) == 0:
            continue
        # close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 98)
        close_depths.append(close_depth), inf_depths.append(inf_depth)
        
        # save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    close_depths, inf_depths = np.array(close_depths), np.array(inf_depths)
    save_arr = [np.concatenate([poses[..., i].ravel(), np.array([max(0.01, np.median(close_depths[close_depths > 0])), 
                                                                 np.median(inf_depths) ])], 0) for i in range(poses.shape[-1])]
    save_arr = np.array(save_arr)
    
    if ".npy" in basedir:
        np.save(basedir, save_arr)
    else:
        np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)


def parallel_map(f, iterable, max_threads=None, show_pbar=False, desc="", **kwargs):
    """Parallel version of map()."""
    with futures.ThreadPoolExecutor(max_threads) as executor:
        if show_pbar:
            results = tqdm(
                executor.map(f, iterable, **kwargs), total=len(iterable), desc=desc)
        else:
            results = executor.map(f, iterable, **kwargs)
        return list(results)


def load_poses_bounds(path):
    poses_arr = np.load(path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    hwf = poses[:, 4:, :].squeeze()

    # shapes = (3,5,n), (2,n), (3,n)
    return poses, bds, hwf