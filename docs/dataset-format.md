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
- cameras         # camera jsons same as the nerfies format with an extra "t". "t" is in microsec
- rgb             # the images
- dataset.json    # same as in nerfies
- metadata.json   # same as in nerfies

```

### ecam_set details
```
ecam_set
- eimgs
    - eimgs_1x.npy      # (n, h, w), preprocessed BIIs
- next_camera           # event camera pose at the end of the BII
- prev_camera           # event camera pose at the start of the BII
- dataset.json          # same as in nerfies
- metadata.json         # same as in nerfies
```