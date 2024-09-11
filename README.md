# LSENeRF-data-formatter
This repo contains the code to format the data in the paper LSENeRF

[Paper](http://arxiv.org/abs/2409.06104) | [Webpage](https://ubc-vision.github.io/LSENeRF/) | [Method code](https://github.com/ubc-vision/LSENeRF)

## Download the dataset
**For details of the downloaded files and formatted data see [here](docs/scene-details.md)**
#### Mandatory files
- [ecam_intrinsics.json](https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/ecam_intrinsics.json)
- [rel_extrinsics.json](https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/rel_extrinsics.json)

#### Scenes
| Outdoors | Indoors |
|----------|----------|
| [Bag](https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/Bag.zip) | [Dragon Max](<https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/Dragon Max.zip>) |
| [Courtyard](https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/Courtyard.zip) | [Grad Lounge](<https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/Grad Lounge.zip>) |
| [Bicycle](https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/Bicycle.zip) | [Lab](https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/Lab.zip) |
| [Engineer Building](<https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/Engineer Building.zip>) | [Presentation Room](<https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/Presentation Room.zip>) |
| [House](https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/House.zip) | [Teddy Grass](<https://www.cs.ubc.ca/research/kmyi_data/files/2024/LSENeRF/Teddy Grass.zip>) |

## Preliminaries
Make an environment with the packages in *requirements.txt* installed

## Formating LSENeRF dataset
To format a scene in our dataset:
1. Download *ecam_intrinsics.json*, *rel_extrinsics* and a scene from [here](docs/download-page.md)
2. Fill out the fields in format_raw.sh, the fields details are provided in the comments of the shell script
3. run the below
```bash
bash format_raw.sh
```

## Formatting EVIMOv2 dataset
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
