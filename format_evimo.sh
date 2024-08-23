
######### INPUT #########
# source directory for rgb camera
RGB_SRC_DIR="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/flea3_7/sanity/depth_var/depth_var_1_lr_000000"

# source directory for event camera
EVS_SRC_DIR="/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/evimo2_v2_data/npz/samsung_mono/sanity/depth_var/depth_var_1_lr_000000"

# save directory
OUT_DIR="formatted_data/$(basename $RGB_SRC_DIR)"
######### INPUT END#########


# format evimov2 data
python format_evimo.py --rgb_src_dir "$RGB_SRC_DIR" \
                       --evs_src_dir "$EVS_SRC_DIR" \
                       --out_dir "$OUT_DIR"


# update dataset.json
python update_dataset.py --dataset_dir "$OUT_DIR"
