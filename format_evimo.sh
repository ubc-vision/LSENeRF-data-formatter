
######### INPUT #########
# source directory for rgb camera
RGB_SRC_DIR="evimo2_v2_data/npz/flea3_7/sfm/train/scene7_00_000001"

# source directory for event camera
EVS_SRC_DIR="evimo2_v2_data/npz/samsung_mono/sfm/train/scene7_00_000001"

# save directory
OUT_DIR="formatted_data/$(basename $RGB_SRC_DIR)"
######### INPUT END#########


# format evimov2 data
python format_evimo.py --rgb_src_dir "$RGB_SRC_DIR" \
                       --evs_src_dir "$EVS_SRC_DIR" \
                       --out_dir "$OUT_DIR"


# generate ecam_set dataset.json
python update_dataset.py --dataset_dir "$OUT_DIR"
