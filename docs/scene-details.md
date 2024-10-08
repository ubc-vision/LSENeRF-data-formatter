# File details
We provide the details of files we provided. Each scene is ~20 seconds, the first ~13 seconds are blurry while the rest are clear.

### Scene folder
We first go through the scene folder. An example is just the `Bag' directory obtained after downloading and unzipping. The folder structure follows this:
```
<scene>
- <scene>_recon         # colmap reconstruction
    - images            # images used for colmap reconstruction
    - sparse            # colmap sparse reconstruction, see colmap for details
        - 0
            - camera.bin   # RGB camera intrinsics; readable with colmap_read_model.read_cameras_binary 
            - images.bin   # RGB camera extrinsics
            ...
    - colmap_scale.txt  # colmap scale (colmap unit/cm); see below and sec4 Camera pose. of paper for details
    - database.db       # colmap database; see colmap for details
- raw_images
    - raw_image_<number>.raw # raw image, <number> is timestamp in nano seconds
    ....
- bias.bias             # bias file from prophesee. see prophesee for details
- cam_ts.npy            # camera pose times for evs_cams.npy and rgb_cams.npy.
- dataset.json          # train, val split for RGB AFTER formatting; unregistered colmap images are excluded
- end_triggers.txt      # triggers for when camera shutter closes (micro sec)
- events.h5             # events saved with keys (x,y,t,p); p=1 is positive, p=0 is negative, t in microsec
- evs_cams.npy          # (n, 4, 4) world to camera matrixes for event camera; see below for details
- rgb_cams.npy          # (n, 4, 4) world to camera matrixes for RGB camera
- st_triggers.txt       # triggers for when camera shutter opens (micro sec)
```
Special note:
- For event camera poses, only cameras where cam_ts < 20sec are valid.
- Raw images can be converted to RGB with (see supplemental for values of k and b):
```python
utils.raw_to_rgb("raw_image_<number>.raw", k=2.2, b=65536)
```

### Mandatory json files
We provide two other files for event camera intrinsics and relative extrinsics between the rgb and event camera
- ecam_intrinsics.json

This file contains the intrinsics used for the event camera. We use the OpenCV camera model. We provide the meaning of useful key here:
```
{
    ...
    "image_size": (width, height),
    "camera_matrix":{...
        "data": flattened 3x3 camera matrix;
    },
    "distortion_coefficients":{
        ...,
        "data": the distortion coefficients
    }
    ...
}
```
The relative extrinsics are provided in:
- **rel_extrinsics.json**

The meaning of the keys are:
```
{
    ...
    "M2": (3x3) event camera intrinsics; same as in ecam_intrinsics.json,
    "dist2": event camera distortion; same as in ecam_intrinsics,
    "R": relative rotation from RGB camera-> Event camera,
    "T": relative translation from RGB camera -> Event camera
    ...
}
```
To be specific, considering a rgb world to camera matrix (3x4)

$$[R_{rgb} | T_{rgb}]$$

The event camera extrinsics can be obtained using $R, T$ in rel_extrinsics.json by:

$$R_{evs} = R \cdot R_{rgb}$$

$$T_{evs} = R \cdot T_{rgb} + T \cdot s$$

$R_{xxx}, T_{xxx}$ are the rotation and translation for the respective cameras. $s$ is a scaler in *colmap_scale.txt*

### Camera intrinsics reminder
The event camera intrinsics can be can be found in:
- **ecam_intrinsics.json**

For RGB camera, the intrinsics are inside the `<scene>/<scene>_recon/sparse/0/camera.bin` readable with `colmap_read_model.read_cameras_binary`.

# Formatted dataset details
we provide the details for each of the files in the formatted dataset. Our dataset format is very similar to nerfies

After formatting, the scene structure will look like:
```
<scene>
- colcam_set  # rgb camera data
- ecam_set    # event camera data
```

### colcam_set details
```
colcam_set
- cameras         
    - 000000.json # camera jsons same as the nerfies format with an extra "t". "t" is in microsec
    ...
- rgb             # the images
- dataset.json    # same as in nerfies
- metadata.json   # same as in nerfies

```

### ecam_set details
```
ecam_set
- eimgs
    - eimgs_1x.npy      # (n, h, w), preprocessed event images (BIIs)
- next_camera           # event camera pose at the end of the BII; same as nerfies
    - 000000.json
    ...
- prev_camera           # event camera pose at the start of the BII; same as nerfies
    - 000000.json
    ...
- dataset.json          # same as in nerfies
- metadata.json         # same as in nerfies