# LSENeRF-data-formatter
This repo contains the code to format the data in the paper LSENeRF

[Paper](placeholder) | [Webpage](https://ubc-vision.github.io/LSENeRF/) | [Method code](https://github.com/ubc-vision/LSENeRF)

## Download the dataset
The dataset can be download [here](docs/download-page.md)

For details of the files downloaded, see [here](docs/scene-details.md)

## preliminaries
Make an environment with the packages in *requirements.txt* installed

## formating LSENeRF dataset
To format a scene in our dataset:
1. Download *ecam_intrinsics.json*, *rel_extrinsics* and a scene from [here](docs/download-page.md)
2. Fill out the fields in format_raw.sh, the fields details are provided in the comments of the shell script
3. run the below
```bash
bash format_raw.sh
```

## formatting EVIMOv2 dataset
1. Download the flea3_7 and samsung_mono NPZ datatset [here](https://better-flow.github.io/evimo/download_evimo_2.html)
2. Download NPZ extrinsics overlay [here](https://better-flow.github.io/evimo/npz_extrinsics.zip). This is available on the same page as 1.
   1. The NPZ extrinsics overlay is camera transform for camera -> rig, the rig is equivalent to world
3. Extract NPZ extrinsics overlay from 2. on top of the NPZ dataset so that *dataset_extrinsics.npz* is available in the dataset scene folders
4. Fill the neccessary fields in `format_evimo.sh`
5. Format a scene in the evimovs dataset with
```bash
bash format_evimo.sh
```
Note that only these EVIMOv2 scenes are supported (see the ```evimov2_dataset``` directory):
- depth_var_1_lr_000000
- scene7_00_000001
- scene8_01_000000

For details of the formatted scenes, see [here](docs/dataset-format.md)