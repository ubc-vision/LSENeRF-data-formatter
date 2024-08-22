import os
import os.path as osp
import glob
import shutil
from tqdm import tqdm

## src_dirs:
WORK_DIR="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir"
RGB_RAW_DIR="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/rgb-evs-cam-drivers/data_rgb"
EVS_DIR="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/rgb-evs-cam-drivers/ev_recordings"
EVS_K_F="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/rgb-evs-cam-drivers/intrinsics/ecam_K_v7.json"
DATASET_DIR = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data"

DST_DIR="raw_data"
os.makedirs(DST_DIR, exist_ok=True)

name_map = {  # ori_name : Paper Name
  "courtyard1_v10_t1": "Courtyard",
  "bag_v16_c2": "Bag",
  "house_v16_c1": "House",
  "eng_v15_c3": "Engineer Building",
  "bc_v13_t2_c2": "Bicycle",
  "mimi1_v13_c1": "Grad Lounge",
  "mimi2_v13_c2": "Presentation Room",
  "teddy_grass_v9_c2": "Teddy Grass",
  "dragon_max_v9_c2": "Dragon Max",
  "jeff_v9": "Lab"
}


def copy_workdir(src_dir, dst_dir):
    ori_name, paper_name = osp.basename(src_dir), osp.basename(dst_dir)

    # copy recon dir
    src_recon_dir = osp.join(src_dir, f"{ori_name}_recon")
    dst_recon_dir = osp.join(dst_dir, f"{paper_name}_recon")
    shutil.copytree(src_recon_dir, dst_recon_dir)

    # copy processed_events.h5
    src_evs_f = osp.join(src_dir, "processed_events.h5")
    dst_evs_f = osp.join(dst_dir, "events.h5")
    assert osp.exists(src_evs_f), f"{src_evs_f} does not exist"
    shutil.copy(src_evs_f, dst_evs_f)

    # copy start triggers
    st_trigger_f = osp.join(src_dir, "st_triggers.txt")
    dst_st_trigger_f = osp.join(dst_dir, "st_triggers.txt")
    assert osp.exists(st_trigger_f), f"{st_trigger_f} does not exist"
    shutil.copy(st_trigger_f, dst_st_trigger_f)

    # copy end triggers
    end_trigger_f = osp.join(src_dir, "end_triggers.txt")
    dst_end_trigger_f = osp.join(dst_dir, "end_triggers.txt")
    assert osp.exists(end_trigger_f), f"{end_trigger_f} does not exist"
    shutil.copy(end_trigger_f, dst_end_trigger_f)

def copy_cam_params(src_dir, dst_dir):
    rel_cam_f = osp.join(src_dir, "rel_cam.json")
    dst_cam_f = osp.join(dst_dir, "rel_extrinsics.json")
    assert osp.exists(rel_cam_f), f"{rel_cam_f} does not exist"
    shutil.copy(rel_cam_f, dst_cam_f)

    dst_evs_k_f = osp.join(dst_dir, "ecam_intrinsics.json")
    shutil.copy(EVS_K_F, dst_evs_k_f)


def main():
    
    src_dir = osp.join(WORK_DIR, list(name_map.keys())[0])
    copy_cam_params(src_dir, DST_DIR)
    
    # recursively copy all files from src_dirs to dst_dir
    for ori_name, paper_name in tqdm(name_map.items()):
        src_work_dir = osp.join(WORK_DIR, ori_name)
        dst_work_dir = osp.join(DST_DIR, paper_name)
        os.makedirs(dst_work_dir, exist_ok=True)
        copy_workdir(src_work_dir, dst_work_dir)

        # copy raw
        src_rgb_raw_dir = osp.join(RGB_RAW_DIR, ori_name)
        dst_rgb_raw_dir = osp.join(DST_DIR, paper_name, "raw_images")
        shutil.copytree(src_rgb_raw_dir, dst_rgb_raw_dir)
        os.remove(osp.join(dst_rgb_raw_dir, "metadata.json"))

        # copy bias file
        src_evs_f = osp.join(EVS_DIR, ori_name, "bias.bias")
        dst_evs_f = osp.join(DST_DIR, paper_name, "bias.bias")
        shutil.copy(src_evs_f, dst_evs_f)

        src_dataset_f = osp.join(DATASET_DIR, ori_name, "colcam_set", "dataset.json")
        dst_dataset_f = osp.join(DST_DIR, paper_name, "dataset.json")
        assert osp.exists(src_dataset_f), f"{src_dataset_f} does not exist"
        shutil.copy(src_dataset_f, dst_dataset_f)


if __name__ == "__main__":
    main()