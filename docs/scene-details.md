# File details
We provide the details of files we provided. 

# Scene folder
We first go through the scene folder. An example is just the `Bag' directory obtained after downloading and unzipping. The folder structure follows this:
```
<scene>
- <scene>_recon         # colmap reconstruction
    - images            # images used for colmap reconstruction
    - sparse            # colmap sparse reconstruction, see colmap for details
        - 0
            - camera.bin   # RGB camera intrinsics; readable with colmap_read_model.read_cameras_binary 
            ...
    - colmap_scale.txt  # colmap scale (colmap unit/cm); see below and sec4 Camera pose. of paper for details
    - database.db       # colmap database; see colmap for details
- raw_images
    - raw_image_<number>.raw # raw image, <number> is timestamp in nano seconds
    ....
- bias.bias             # bias file from prophesee. see prophesee for details
- cam_ts.npy            # camera pose times for evs_cams.npy and rgb_cams.npy.
- dataset.json          # train, val split for RGB AFTER formatting; unregistered images are excluded
- end_triggers.txt      # triggers for when camera shutter closes (micro sec)
- events.h5             # events saved with keys (x,y,t,p); p=1 is positive, p=0 is negative
- evs_cams.npy          # (n, 4, 4) world to camera matrixes for event camera; see below for details
- rgb_cams.npy          # (n, 4, 4) world to camera matrixes for RGB camera
- st_triggers.txt       # triggers for when camera shutter opens (micro sec)
```
Special note:
- Only len(cam_ts) number of RGB and Event cameras starting from the beginning are inside the event camera filming time. The extra cameras are cameras after the event camera finished filming

# Extra json files
We provide two other files for event camera intrinsics and relative extrinsics between the rgb and event camera
- ecam_intrinsics.json

This file contains the intrinsics used for the event camera. We use the OpenCV camera model. We provide the meaning of useful key here:
```
{
    ...
    "image_size": (width, height),
    "camera_matrix":{...
        "data": 3x3 camera matrix; obtained by reshaping to 3x3
    },
    "distortion_coefficients":{
        ...,
        "data": the distortion coefficients
    }
    ...
}
```
The relative extrinsics are provided in:
- rel_extrinsics.json

The meaning of the keys are:
```
{
    ...
    "M2": (3x3) event camera intrinsics; same as in ecam_intrinsics.json,
    "dist1": event camera distortion; same as in ecam_intrinsics,
    "R": relative rotation from RGB camera-> Event camera,
    "T": relative translation from RGB camera -> Event camera
    ...
}
```
To be specific, considering a rgb world to camera matrix (3x4)

$$[R_{rgb} | T_{rgb}]$$

The event camera extrinsics can be obtained by:

$$R_{evs} = R \cdot R_{rgb}$$

$$T_{evs} = R \cdot T_{rgb} + T \cdot s$$

$R_*, T_*$ are the rotation and translation for the cameras. $s$ is a scaler in *colmap_scale.txt*

# Camera intrinsics reminder
The event camera intrinsics can be can be found in:
- ecam_intrinsics.json

For RGB camera, the intrinsics are inside the `<scene>/<scene>_recon/sparse/0/camera.bin` readable with colmap_read_model.read_cameras_binary.
