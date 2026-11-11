# Cloud Type Image Classification

Comparison of classical machine learning (**SVM**, **Random Forest**) and a **Convolutional Neural Network (CNN)** for classifying sky images into seven cloud types. Data Modeling course project.

## Results

| Model | Input | Result |
|---|---|---|
| SVM | PCA-reduced pixels | weighted F1 0.58 |
| Random Forest | PCA-reduced pixels | weighted F1 0.55 |
| **CNN (after hyperparameter tuning)** | 128×128 RGB images | **accuracy 70.3%** |

The tuned CNN performed best. Learning spatial features directly from the images worked better than flattening pixels for traditional models.

## Dataset

960 RGB images in 7 classes, split into train and test folders:

| Class | Train | Test |
|---|---|---|
| Cirriform clouds | 66 | 77 |
| Clear sky | 60 | 64 |
| Cumulonimbus clouds | 14 | 11 |
| Cumulus clouds | 91 | 120 |
| High cumuliform clouds | 133 | 103 |
| Stratiform clouds | 49 | 40 |
| Stratocumulus clouds | 61 | 71 |

The images are not included in this repository. To rerun the notebook, place them in `clouds_train/` and `clouds_test/` folders (one subfolder per class) and update the path in the first cells.

## Pipeline

1. **EDA**: sample images and class distribution
2. **Preprocessing**: resize to 128×128, pixel normalization, label encoding, train/test split
3. **Classical ML**: flatten images → PCA → SVM and Random Forest
4. **Deep learning**: CNN with one-hot labels
5. **Hyperparameter tuning** for SVM, Random Forest and CNN
6. **Model comparison**: accuracy, precision, recall, F1, confusion matrices

## Repository structure

```
notebooks/cloud_type_classification.ipynb   # full workflow with outputs
requirements.txt
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/cloud_type_classification.ipynb
```

The notebook was developed in Google Colab; markdown commentary is partly in Indonesian.

## Tech stack

Python · TensorFlow/Keras · scikit-learn · NumPy · pandas · Pillow · Matplotlib · Seaborn
