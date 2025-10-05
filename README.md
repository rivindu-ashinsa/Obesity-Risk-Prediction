# Obesity Risk Prediction

This repository contains code and resources for predicting obesity risk categories using tabular health and lifestyle data. The project includes data preprocessing, feature engineering, model training, and prediction pipelines.

## Repository Structure

```
main.py
README.md
submission_73.csv
submission.csv
data/
    sample_submission.csv
    processed/
    raw/
        test.csv
        train.csv
data_pipelines/
    training_pipeline.py
    __pycache__/
        training_pipeline.cpython-312.pyc
notebooks/
    main.ipynb
    val_predictions.csv
```

## Data

- **data/raw/train.csv**: Training data with features and target (`Weight_Category`).
- **data/raw/test.csv**: Test data for prediction.
- **data/sample_submission.csv**: Example submission format.

## Main Components

### 1. Data Preprocessing

- **Outlier Removal**: Removes samples with extreme values using [`OutlierRemover`](data_pipelines/training_pipeline.py).
- **Category Renaming**: Unifies categorical values using [`CategoryRenamer`](data_pipelines/training_pipeline.py).
- **Missing Value Imputation**: Fills missing values using [`MissingValueImputer`](data_pipelines/training_pipeline.py).

### 2. Feature Engineering

- Adds features such as BMI, Screen-to-Activity ratio, and Age × Activity interaction using [`FeatureEngineer`](data_pipelines/training_pipeline.py).

### 3. Model Pipeline

- Categorical features are one-hot encoded.
- Features are scaled.
- Model used: `RandomForestClassifier` (can be changed to `GradientBoostingClassifier` in [main.py](main.py)).
- Pipeline implemented in [main.py](main.py) and [data_pipelines/training_pipeline.py](data_pipelines/training_pipeline.py).

### 4. Training & Validation

- Data is split into training and validation sets.
- Target labels are encoded.
- Model is trained and evaluated for accuracy.

### 5. Prediction & Submission

- Predictions on test data are saved in `submission.csv`.
- See [`predict_test_file`](notebooks/main.ipynb) and [main.py](main.py) for details.

## Usage

### Train and Predict

Run the main script:

```sh
python main.py
```

This will train the model and generate predictions for the test set.

### Jupyter Notebook

See [notebooks/main.ipynb](notebooks/main.ipynb) for exploratory data analysis, feature engineering, and pipeline experiments.

## Key Files

- [main.py](main.py): Main script for training and prediction.
- [data_pipelines/training_pipeline.py](data_pipelines/training_pipeline.py): Custom transformers and pipeline logic.
- [notebooks/main.ipynb](notebooks/main.ipynb): Data exploration and prototyping.

## Features

- Outlier removal
- Categorical value unification
- Missing value imputation
- Feature engineering (BMI, ratios, interactions)
- One-hot encoding and scaling
- Random forest classification
- Submission file generation

## Requirements

- Python 3.12+
- pandas, scikit-learn, numpy

Install dependencies:

```sh
pip install -r requirements.txt
```

## License

This project is for educational purposes.
