# Deep Learning for Bradycardia Prediction

## Description

Deep Learing based prediction of a cardiac arrhythmia (bradycardia) in infants using ECG signal. The dataset can be downloaded from [here](https://physionet.org/content/picsdb/1.0.0/). 

## File description

- [models](./models/) -> This directory contains code for EncoderAttention, Encoder with BCE Loss, Fully Convolutional Network, Inception Time, and Sequence-to-Sequence models.

- [DataExtraction.ipynb](./DataExtraction.ipynb) -> This notebook is used for extracting ECG values and annotations from the data files, and storing the retrieved values as `.csv` files.

## References

[1] A. H. Gee, R. Barbieri, D. Paydarfar and P. Indic, "Predicting Bradycardia in Preterm Infants Using Point Process Analysis of Heart Rate," in IEEE Transactions on Biomedical Engineering, vol. 64, no. 9, pp. 2300-2308, Sept. 2017, doi: 10.1109/TBME.2016.2632746.

[2] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
