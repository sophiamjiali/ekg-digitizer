# ekg-digitizer
Deep learning framework for extracting time-series data from digitized EKG images

## Github Workflow

Please follow the following workflow when working on this project:

1. Create an issue defining the feature type (bugfix, hotfix, feature, experiment, docs)
2. Create a branch labeled `<type>/<issue_number>-<short_description>`
3. Create commits with conventional messages (feat:, fix:, docs:, test:, refactor:, chore:)
4. Create pull requests linking the issue

## Additional Validation Datasets

The following datasets are candidates for additional validation and possible fine-tuning.

Both datasets contain 12-lead ECG waveform recordings sampled at 500 Hz with 5000 samples per lead, giving 10 seconds of signal per lead. This is similar to many records in the current Kaggle dataset, although the Kaggle data contains multiple sampling frequencies.

One important difference is that these datasets provide all 12 leads for the full 10-second recording, while the Kaggle competition format uses Lead II for the full 10 seconds and the other leads for 2.5 seconds. The full 10-second waveforms could therefore be rendered into the same multi-panel ECG layout used by the Kaggle dataset, using shorter 2.5-second segments for the other leads and a full 10-second Lead II rhythm strip.

Neither additional dataset includes ECG images. The waveform signals would therefore need to be rendered into ECG images before being used as model inputs.

### 1. Large-scale 12-lead ECG Arrhythmia Database

**Source:** PhysioNet  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Link:** https://physionet.org/content/ecg-arrhythmia/1.0.0/

This dataset contains 45,152 12-lead ECG recordings and comes from a separate source from the ECG data used in the current Kaggle dataset, which makes it useful for testing whether the model generalizes to new data.

The waveform data is stored using `.mat` files with accompanying `.hea` header files.

#### Download Steps

From the repository root:

```bash
cd data
wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/
```

### 2. Lobachevsky University Electrocardiography Database (LUDB)

**Source:** PhysioNet  
**License:** Open Data Commons Attribution License v1.0  
**Link:** https://physionet.org/content/ludb/1.0.1/

LUDB contains 200 12-lead ECG recordings. Its smaller size makes it useful for quick external validation and testing before working with a much larger dataset.

The waveform data is stored using WFDB files. Each record includes a `.dat` file containing the ECG waveform and a `.hea` file containing information such as the sampling frequency, signal length, lead names, and signal scaling.

LUDB also includes lead-specific annotation files, although these are not required for the ECG digitization task.

#### Download Steps

From the repository root:

```bash
cd data
wget -r -N -c -np https://physionet.org/files/ludb/1.0.1/ 
```