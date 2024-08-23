
############################################# INPUT #############################################

# path to scene directory downloaded
WORK_DIR="raw_data/Bag"

# path to output directory
OUT_DIR="formatted_data/$(basename $WORK_DIR)"

# path to event camera intrinsics
PROPHESEE_CAM_F="raw_data/ecam_intrinsics.json"

# path to relative extrinsics between the rgb and event camera
REL_CAM_F="raw_data/rel_extrinsics.json"
############################################# INPUT END ###########################################

TMP_OUT="tmp"


DELTA_T=5000 # in microseconds; None to turn off and use N_BINS instead


# Step 1: Format raw data to llff format (intermediate format)
python format_raw_step1.py --work_dir "$WORK_DIR" \
                           --targ_dir "$TMP_OUT" \
                           --delta_t "$DELTA_T" \
                           --prophesee_cam_f "$PROPHESEE_CAM_F" \
                            --rel_cam_f "$REL_CAM_F"


# Step 2: Format llff data to final format
python format_raw_step2.py --src_dir "$TMP_OUT" \
                           --targ_dir "$OUT_DIR"

# Step 3: generate corresponding ecam dataset.json
python update_dataset.py --dataset_dir "$OUT_DIR"

# step 4: clean up temporary output
rm -rf "$TMP_OUT"