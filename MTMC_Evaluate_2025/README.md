# VTX MTMC Evaluation Kit 2025

This is the official tool for evaluating models on the VTX_MTMC_2024. It can also be used for evaluating other datasets by converting the ground-truth and prediction directories into the format below.

## Requirements

`pip install -r requirements.txt`

## Directory Structure

To evaluate the MTMC models, setup the ground truth directory as follows:

```
dataset_path:
    F11_01:
        111101:
            111101.mp4
            111101.txt
        111103:
            111103.mp4
            111103.txt
        111105:.......
        ..............
        grid_view.mp4
        grid_view.txt
    F11_02:...........
    ..................
```

And the prediction directory as follows:

```
predict_path:
    F11_01:
        mot:
            111101.txt
            111103.txt
            ..........
        mtmc:
            111101.txt
            111103.txt
            ..........
    F11_02:...........
    ..................
```

## Evaluation

Currently, this code only evaluate MTMC.

Go into `run_eval.sh`:
- Set the `dataset_path` and `predict_path` to the appropriate directories.
- Choose the metrics to display. Available metrics include: `"HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "CLR_FP", "CLR_FN", "IDSW", "Frag", "IDF1", "IDFP", "IDFN"`.
- `"Dets", "GT_Dets", "IDs", "GT_IDs"` counts are also included in the metrics.

**Note:** For the legacy VTX MTMC code, the MTMC inference result is offset by 2 frames in future frames. Change the `vtx_model_0_offset` to `"yes"` to account for this.

## Evaluation nuances

- This code abolishes the duplicate ID frame check by the default TrackEval. Should there be a duplicate frame, a single False Positive is added to the total count when yielding the metrics. Duplicate IDs in predictions are still warned however.