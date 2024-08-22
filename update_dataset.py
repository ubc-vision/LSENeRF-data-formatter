import json
import os.path as osp
import shutil
import numpy as np
import argparse

from scene_managers import ColcamSceneManager, EcamSceneManager



def change_evs_dataset(manager:EcamSceneManager, train_rgb_ts, t_gap=30000):
    """
    t_gap (int): time in mus, should be less than 1/fps or exposure time
    """
    
    all_ids = np.arange(len(manager.eimgs))
    eimg_ts = manager.ts[:len(all_ids)]

    train_ids = []
    for rgb_t in train_rgb_ts:
        st_t = rgb_t - t_gap/2
        en_t = rgb_t + t_gap/2
        cond = (eimg_ts >= st_t) & (eimg_ts <= en_t)
        train_ids.extend(all_ids[cond])

    train_ids = sorted(np.unique(train_ids))
    train_ids = [str(e).zfill(6) for e in train_ids]

    dataset = {
        "count": len(manager.eimgs),
        "num_exemplars": len(train_ids),
        "train_ids" : train_ids,
        "val_ids":[]
    }

    ori_dataset_f = manager.dataset_json_f
    if not osp.join(manager.data_dir, "ori_dataset.json"):
        shutil.copy(ori_dataset_f, osp.join(manager.data_dir, "ori_dataset.json"))

    with open(osp.join(manager.data_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=2)


def main():
    # scene_dir = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data_abl/dragon_max_v9_c2_10000"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="path to formatted scene directory")
    parser.add_argument("--t_gap", type=int, default=30000, help="window of events for each rgb frame in microsec")
    args = parser.parse_args()

    scene_dir = args.dataset_dir
    colcam_dir = osp.join(scene_dir, "colcam_set")
    ecam_dir = osp.join(scene_dir, "ecam_set")

    colcam = ColcamSceneManager(colcam_dir)
    ecam = EcamSceneManager(ecam_dir)

    print("updating ecam dataset")
    train_rgb_ts = colcam.get_train_ts()  # for updating evs dataset.json
    change_evs_dataset(ecam, train_rgb_ts, t_gap=args.t_gap)


if __name__ == "__main__":
    main()