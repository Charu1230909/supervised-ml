# --------------------------------------------------
# Imports
# --------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(
    page_title="Healthcare Attrition Prediction",
    layout="wide"
)

st.title("🏥 Healthcare Employee Attrition Prediction")
st.markdown("End-to-End **Supervised Machine Learning** Application")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv("watson_healthcare_modified.csv")

st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Target Encoding
# --------------------------------------------------
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Drop non-informative columns
df.drop(
    columns=["EmployeeID", "EmployeeCount", "Over18", "StandardHours"],
    inplace=True
)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# --------------------------------------------------
# Feature Classification
# --------------------------------------------------
categorical_features = X.select_dtypes(include="object").columns
numerical_features = X.select_dtypes(include=np.number).columns

# --------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------
num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_features),
    ("cat", cat_pipeline, categorical_features)
])

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# Models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    )
}

# --------------------------------------------------
# Train & Evaluate Models
# --------------------------------------------------
st.subheader("📊 Model Performance")

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_pred = pipeline.predict(X_test)
        auc = "N/A"

    accuracy = accuracy_score(y_test, y_pred)

    st.markdown(f"### 🔹 {name}")
    st.write("**Accuracy:**", round(accuracy, 4))
    st.write("**ROC-AUC:**", auc)
    st.text(classification_report(y_test, y_pred))

# --------------------------------------------------
# Cross-Validation
# --------------------------------------------------
st.subheader("🔁 Cross-Validation (Random Forest)")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

cv_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=5,
    scoring="roc_auc"
)

st.write("Mean ROC-AUC:", round(cv_scores.mean(), 4))

# --------------------------------------------------
# Hyperparameter Tuning
# --------------------------------------------------
st.subheader("⚙️ Hyperparameter Tuning (Random Forest)")

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(
    Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ]),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

with st.spinner("Running GridSearchCV..."):
    grid.fit(X_train, y_train)

best_model = grid.best_estimator_

st.success("GridSearch Completed!")
st.write("Best Parameters:", grid.best_params_)

# --------------------------------------------------
# Prediction Section (SUPERVISED)
# --------------------------------------------------
st.subheader("🔮 Predict Employee Attrition")

with st.form("prediction_form"):
    user_input = {}

    for col in X.columns:
        if col in categorical_features:
            user_input[col] = st.selectbox(
                col,
                sorted(df[col].unique())
            )
        else:
            user_input[col] = st.number_input(
                col,
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].mean())
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_input])

    probability = best_model.predict_proba(input_df)[0][1]
    prediction = int(probability >= 0.5)

    st.markdown("### 📌 Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Employee likely to **leave** "
            f"(Attrition Probability: {probability:.2%})"
        )
    else:
        st.success(
            f"✅ Employee likely to **stay** "
            f"(Attrition Probability: {probability:.2%})"
        )
