# Reference-label defects in rPPG evaluation (UBFC-rPPG)

Code and results accompanying the manuscript *Two defects in a standard reference
heart-rate estimator, and what they conceal about remote photoplethysmography
evaluation*, submitted to **Biomedical Signal Processing and Control**.

## What this repository is for

While re-evaluating a Vision Transformer and LSTM framework for remote
photoplethysmography under a strict subject-independent protocol, we found two
defects in the Welch-periodogram estimator used to derive reference heart rates
from the contact photoplethysmogram. Both affect any comparison built on labels
that estimator produces.

**Defect 1, spectral quantisation.** A six-second Welch segment at approximately
30 fps gives a frequency resolution of 0.167 Hz, which is about 10 beats per
minute. For subject 1 the entire 55-second recording resolved to two distinct
reference values. Corrected by extending the segment to the full 10 s analysis
window and zero-padding the transform to 8192 points, giving roughly 0.2 bpm.

**Defect 2, harmonic misidentification.** A 4.0 Hz upper search bound admits the
second harmonic of the pulse waveform. For subject 37 the reported reference rate
was approximately half the true rate. Corrected by narrowing the band to 3.0 Hz
and adding a sub-harmonic guard.

The corrected estimator is `hr_from_bvp` in `code/rppg_baselines.py`.

## Layout

```
code/
  rppg_baselines.py      GREEN, CHROM, POS, ICA, PCA, and the corrected estimator
  evaluation.py          subject-wise splitters, metrics, Bland-Altman, stratified errors
  revised_pipeline.ipynb end-to-end pipeline: feature extraction, training, evaluation
figures/                 all figures in the manuscript, 300 dpi, plus the editable
                         draw.io source for the architecture diagram
results/                 per-subject and per-method result tables (CSV)
```

## Data

UBFC-rPPG Dataset 2 is **not** redistributed here. Obtain it from the authors at
<https://sites.google.com/view/ybenezeth/ubfcrppg>. The pipeline expects each
subject directory to contain `vid.avi` and `ground_truth.txt`.

Reference heart rate is re-derived from row 0 of `ground_truth.txt` (the BVP
waveform) rather than taken from row 1 (the supplied heart-rate trace), because
row 1 carries sentinel values on several subjects.

## Reproducing

```bash
conda create -n heartrate python=3.10
conda activate heartrate
pip install numpy scipy scikit-learn pandas matplotlib torch torchvision transformers
pip install opencv-python-headless==4.10.0.84
jupyter notebook code/revised_pipeline.ipynb
```

`opencv-python-headless` is pinned because `cv2.CascadeClassifier` was removed in
OpenCV 5.0 and the pipeline uses Haar cascade face detection.

Set the dataset root at the top of the notebook before running. Feature
extraction is cached per subject, so a re-run of the evaluation does not require
the videos a second time.

## Two notes on the code as released

1. In the window-dataset cell, a markdown comment states that the target is the
   reference heart rate at the end of the window. The code takes the value at the
   window centre, which matches `windowed_hr` and matches the manuscript. The code
   is correct and the comment is not.

2. The regression head reads the bidirectional LSTM state at the final timestep.
   At that timestep the backward direction has consumed one frame, so roughly half
   the recurrent capacity does not inform the prediction. This is discussed in the
   Limitations section of the manuscript. The code is released as it was run for
   the reported results rather than silently repaired.

## Citation

Citation details will be added here once the manuscript has a DOI.

## Licence

MIT. See `LICENSE`.
