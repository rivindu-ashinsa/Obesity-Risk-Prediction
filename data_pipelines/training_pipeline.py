import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# Custom Transformers
# ----------------------------
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.05, upper=0.95):
        self.lower = lower
        self.upper = upper
        self.feature_names_in_ = None
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        numeric_features = X.select_dtypes(include=[np.number]).columns
        self.feature_names_in_ = numeric_features
        self.lower_bounds_ = X[numeric_features].quantile(self.lower)
        self.upper_bounds_ = X[numeric_features].quantile(self.upper)
        return self

    def transform(self, X):
        X_ = X.copy()
        mask = pd.Series(True, index=X_.index)
        for col in self.feature_names_in_:
            mask &= (X_[col] >= self.lower_bounds_[col]) & (X_[col] <= self.upper_bounds_[col])
        return X_.loc[mask]


class CategoryRenamer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rename_maps = {
            "High_Calorie_Food": {"yes": "Yes", "no": "No"},
            "Family_History": {"yes": "Yes", "no": "No", "yess": "Yes"},
            "Smoking_Habit": {"yes": "Yes", "no": "No"},
            "Alcohol_Consumption": {"no": "No"},
            "Commute_Mode": {"Public_transportation": "Public_Transportation"},
            "Leisure Time Activity": {"Sport": "Sports"}
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in X_.select_dtypes(include="object").columns:
            X_[col] = X_[col].astype(str).str.strip()
        for col, mapping in self.rename_maps.items():
            if col in X_.columns:
                X_[col] = X_[col].replace(mapping)
        return X_


class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, use_smart_activity_impute=True):
        self.use_smart_activity_impute = use_smart_activity_impute
        self.gender_mode_ = None
        self.alcohol_mode_ = None

    def fit(self, X, y=None):
        self.gender_mode_ = X['Gender'].mode()[0]
        self.alcohol_mode_ = X['Alcohol_Consumption'].mode()[0]
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['Gender'] = X_['Gender'].fillna(self.gender_mode_)
        X_['Alcohol_Consumption'] = X_['Alcohol_Consumption'].fillna(self.alcohol_mode_)
        if self.use_smart_activity_impute:
            X_['Physical_Activity_Level'] = X_.apply(self._impute_activity_level, axis=1)
        else:
            X_['Physical_Activity_Level'] = X_['Physical_Activity_Level'].fillna('Unknown')
        return X_

    def _impute_activity_level(self, row):
        if pd.notnull(row['Physical_Activity_Level']):
            return row['Physical_Activity_Level']
        score = row['Activity_Level_Score']
        if score < 0.3:
            return 'Low'
        elif score < 0.7:
            return 'Medium'
        else:
            return 'High'


categorical_features = [
        "High_Calorie_Food",
        "Gender",
        "Family_History",
        "Snack_Frequency",
        "Smoking_Habit",
        "Alcohol_Consumption",
        "Commute_Mode",
        "Physical_Activity_Level",
        "Leisure Time Activity"
    ]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds engineered features to improve model accuracy:
    - BMI
    - Screen_to_Activity ratio
    - Age × Activity interaction
    Drops irrelevant or intermediate columns.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        # ----------------------
        # 1️⃣ BMI
        # ----------------------
        if 'Height_cm' in X_.columns and 'Weight_Kg' in X_.columns:
            X_['Height_m'] = X_['Height_cm'] / 100
            X_['BMI'] = X_['Weight_Kg'] / (X_['Height_m'] ** 2)

        # ----------------------
        # 2️⃣ Screen-to-Activity ratio
        # ----------------------
        if 'Screen_Time_Hours' in X_.columns and 'Activity_Level_Score' in X_.columns:
            X_['Screen_to_Activity'] = X_['Screen_Time_Hours'] / (X_['Activity_Level_Score'] + 0.1)

        # ----------------------
        # 3️⃣ Age × Activity interaction
        # ----------------------
        if 'Age_Years' in X_.columns and 'Activity_Level_Score' in X_.columns:
            X_['Age_Activity'] = X_['Age_Years'] * X_['Activity_Level_Score']

        # ----------------------
        # Drop intermediate / original columns
        # ----------------------
        drop_cols = ['Height_cm', 'Weight_Kg', 'Height_m']
        X_ = X_.drop(columns=[c for c in drop_cols if c in X_.columns])

        return X_

# ----------------------------
# Main Training Pipeline
# ----------------------------
def main():
    global pipeline
    global X_train, y_train_enc, y_val_enc
    global X_val, label_enc
    # 1️⃣ Load data
    df = pd.read_csv("../data/raw/train.csv")

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

    # 6️⃣ Categorical features to One-Hot Encode
    categorical_features = [
        "High_Calorie_Food",
        "Gender",
        "Family_History",
        "Snack_Frequency",
        "Smoking_Habit",
        "Alcohol_Consumption",
        "Commute_Mode",
        "Physical_Activity_Level",
        "Leisure Time Activity"
    ]

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    # 7️⃣ Build full pipeline (without OutlierRemover)
    pipeline = Pipeline([
        ('rename', CategoryRenamer()),
        ('impute', MissingValueImputer()),
        ('encode', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # 8️⃣ Train
    pipeline.fit(X_train, y_train_enc)

    # 9️⃣ Predict & evaluate
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val_enc, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # 10️⃣ Save validation predictions
    val_predictions = pd.DataFrame({
        'PersonID': df.loc[X_val.index, 'PersonID'],
        'Weight_Category': label_enc.inverse_transform(y_pred)
    })
    val_predictions.to_csv("val_predictions.csv", index=False)
    print("Sample predictions saved to val_predictions.csv")


if __name__ == "__main__":
    main()
