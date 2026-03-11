# M214A Audio Processing Pipeline

This repository contains the final pipeline for the M214A audio classification project. The pipeline processes input speech data, extracts robust features, and trains a Simple LSTM model only on clean data to classify the audio which can have up to 5dB of noise.

---

### How to Run

1. Upload your dataset (`M214_project_data.zip`) to your Google Drive.
2. Open the `final_pipeline.ipynb` notebook in Google Colab.
3. Locate the file path variable in the initial setup cells and update it to point to the exact location of the zip file in your Google Drive.
4. Run all cells in the notebook sequentially.

---

### Key Changes from the Baseline

* **Feature Extraction:** The `extract_feature` function was completely rewritten to include advanced processing like SVD low-rank approximation, soft flooring, and unsharp masking.
* **Model Checkpointing:** The saving mechanism was modified to save the model weights from the epoch that achieves the best accuracy on the 5dB noisy test set, overriding the baseline's default behavior of saving based on the 10dB test set.
* **Extra Imported Library:** We imported 'scipy.ndimage' in addition to all the imported libraries.

---

### Pipeline Overview

The audio processing pipeline is designed to aggressively filter out noise and isolate the vocal features before passing them to the LSTM. Here is a high-level breakdown of the custom processing steps:

* **Audio Loading & Pre-emphasis:** Loads the raw audio and applies a pre-emphasis filter to amplify high frequencies.
* **VAD & Spectral Subtraction:** Computes Gaussian-smoothed RMS energy for Voice Activity Detection (VAD). It calculates absolute deltas—strictly zeroing out padded edges to prevent false boundary detection—to crop trailing silence. Noise is estimated from non-speech frames and removed via spectral subtraction.
* **Mel-Spectrogram & SVD Purification:** Converts the clean audio into a Log-Mel spectrogram. It then applies Singular Value Decomposition (SVD) to keep only the top components (speech) and zeroes out the bottom components (noise).
* **Image Enhancement:** Applies custom soft flooring and unsharp masking (Gaussian blur subtraction) to further sharpen the spectrogram matrix.
* **Feature Engineering:** Applies RASTA filtering to suppress stationary noise and isolate syllable-rate frequencies. 
* **Safe Deltas & Stacking:** Computes 1st and 2nd order dynamic features using a `compute_safe_delta` function, which pads short audio clips to prevent standard delta functions from crashing. Features are normalized and stacked into four tiers: static base, delta 1, delta 2, and robust energy shape.
