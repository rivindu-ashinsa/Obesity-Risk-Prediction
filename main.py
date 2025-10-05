import pandas as pd
from data_pipelines.training_pipeline import OutlierRemover, CategoryRenamer, MissingValueImputer,categorical_features, FeatureEngineer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier



# 1️⃣ Load data
df = pd.read_csv("data/raw/train.csv")

# 2️⃣ Separate target
y = df['Weight_Category']
X = df.drop(columns=['Weight_Category', 'PersonID'])

# 3️⃣ Remove outliers BEFORE splitting
outlier_remover = OutlierRemover(lower=0.05, upper=0.95)
X_cleaned = outlier_remover.fit_transform(X)
y_cleaned = y.loc[X_cleaned.index]  # align target with filtered X

# 4️⃣ Split train / validation
X_train, X_val, y_train, y_val = train_test_split(
X_cleaned, y_cleaned, test_size=0.2, random_state=42, stratify=y_cleaned
)

# 5️⃣ Encode target
label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train)
y_val_enc = label_enc.transform(y_val)


preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

model = GradientBoostingClassifier(n_estimators=200, random_state=42)

pipeline = Pipeline([
    ('rename', CategoryRenamer()),
    ('impute', MissingValueImputer()),
    ('feature_eng', FeatureEngineer()),
    ('encode', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train_enc)
y_pred = pipeline.predict(X_val)
acc = accuracy_score(y_val_enc, y_pred)
results = {"GradientBoosting": acc}
print(f"GradientBoosting Validation Accuracy: {acc:.4f}")

# Optionally, print all results together
print("Model comparison:", results)



# ----------------------------
# Predict on test.csv
# ----------------------------
def predict_test_file(pipeline, label_enc, test_file="test.csv", submission_file="submission.csv"):
    # Load test data
    df_test = pd.read_csv(test_file)

    # Keep PersonID for submission
    person_ids = df_test['PersonID']

    # Drop target column if exists (usually it doesn't in test.csv)
    X_test = df_test.drop(columns=['PersonID'], errors='ignore')

    # Predict
    y_test_pred = pipeline.predict(X_test)

    # Convert back to original categories
    y_test_labels = label_enc.inverse_transform(y_test_pred)

    # Prepare submission dataframe
    submission_df = pd.DataFrame({
        "PersonID": person_ids,
        "Weight_Category": y_test_labels
    })

    # Save CSV
    submission_df.to_csv(submission_file, index=False)
    print(f"Predictions saved to {submission_file}")

# Train pipeline as before
pipeline.fit(X_train, y_train_enc)

# Predict validation (optional)
y_pred = pipeline.predict(X_val)
acc = accuracy_score(y_val_enc, y_pred)
print(f"Validation Accuracy: {acc:.4f}")

# Predict test.csv
predict_test_file(pipeline, label_enc, test_file="data/raw/test.csv", submission_file="submission.csv")
