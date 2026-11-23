import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="CareerPredict", page_icon="🎓", layout="wide")
st.title("🎓 CareerPredict")
st.caption("College Placement Prediction & Student Readiness Analytics")
st.write("Explore student data, compare ML models, predict placement outcomes, and export batch predictions.")

def detect_target(columns):
    aliases = {"placement", "placementstatus", "placement_status", "status", "placed"}
    for col in columns:
        key = str(col).strip().lower().replace(" ", "").replace("-", "_")
        if key in aliases:
            return col
    return None

def clean_target(series):
    yes = {"yes", "placed", "1", "true", "selected", "success"}
    no = {"no", "not placed", "not_placed", "0", "false", "unplaced", "not selected"}
    vals = series.astype(str).str.strip()
    out = []
    known = True
    for v in vals:
        k = v.lower()
        if k in yes:
            out.append("Placed")
        elif k in no:
            out.append("Not Placed")
        else:
            known = False
            out.append(v)
    return pd.Series(out, index=series.index), ("Placed" if known else None)

def build_pipeline(model_name, numeric, categorical):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers = []
    if numeric:
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        transformers.append(("cat", cat_pipe, categorical))
    prep = ColumnTransformer(transformers=transformers)
    if model_name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=1500, class_weight="balanced")
    elif model_name == "Decision Tree":
        estimator = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    else:
        estimator = RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                           class_weight="balanced", random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", prep), ("model", estimator)])

def tips_for_student(row, features):
    tips = []
    checks = [
        ("cgpa", "CGPA", 7.0),
        ("communication", "communication skills", 6.0),
        ("extra_curricular", "extra-curricular score", 5.0),
        ("projects_completed", "completed projects", 2),
        ("prev_sem_result", "previous semester result", 6.5),
        ("iq", "aptitude / IQ score", 90)
    ]
    for token, label, threshold in checks:
        col = next((c for c in features if token in str(c).lower()), None)
        if col:
            try:
                if float(row[col]) < threshold:
                    tips.append(f"Consider improving {label} (current value: {row[col]}).")
            except (ValueError, TypeError):
                pass
    internship = next((c for c in features if "internship" in str(c).lower()), None)
    if internship and str(row[internship]).strip().lower() in {"no", "0", "false"}:
        tips.append("Look for practical internship experience.")
    return tips or ["Continue building projects, practical skills, and interview readiness."]

# Data upload / local data
with st.sidebar:
    st.header("Dataset")
    upload = st.file_uploader("Upload Kaggle CSV", type=["csv"])
    st.markdown("[Dataset on Kaggle](https://www.kaggle.com/datasets/sahilislam007/college-student-placement-factors-dataset)")
    st.caption("Alternatively put the CSV in the project's data/ folder.")

df = None
if upload:
    df = pd.read_csv(upload)
else:
    files = glob.glob("data/*.csv") + glob.glob("*.csv")
    if files:
        df = pd.read_csv(files[0])

if df is None:
    st.info("Upload the Kaggle CSV from the sidebar, or run download_data.py to place it in data/.")
    st.code("pip install -r requirements.txt\npython download_data.py\nstreamlit run app.py")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
detected_target = detect_target(df.columns)

overview, analysis, lab, single, batch = st.tabs([
    "Overview", "Data Analysis", "Model Lab", "Single Prediction", "Batch Prediction"
])

with overview:
    st.subheader("Dataset overview")
    a, b, c, d = st.columns(4)
    a.metric("Student records", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(15), use_container_width=True)
    info = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[x].dtype) for x in df.columns],
        "Missing": [int(df[x].isna().sum()) for x in df.columns],
        "Unique": [int(df[x].nunique(dropna=True)) for x in df.columns]
    })
    st.markdown("#### Column summary")
    st.dataframe(info, use_container_width=True)
    if detected_target:
        st.success(f"Detected placement target: {detected_target}")
    st.warning("The Kaggle dataset is described as a simulated dataset. Treat model results as educational estimates, not guarantees about real hiring.")

with analysis:
    st.subheader("Exploratory data analysis")
    if not detected_target:
        st.warning("Could not auto-detect target. Check your target column name.")
    else:
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots()
            df[detected_target].fillna("Missing").astype(str).value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Placement outcome distribution")
            ax.set_xlabel("Placement outcome")
            ax.set_ylabel("Students")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            miss = df.isna().sum().sort_values(ascending=False)
            miss = miss[miss > 0]
            if len(miss):
                fig, ax = plt.subplots()
                miss.plot(kind="bar", ax=ax)
                ax.set_title("Missing values by column")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.success("No missing values found.")
        numeric = df.drop(columns=[detected_target], errors="ignore").select_dtypes(include=np.number).columns.tolist()
        if numeric:
            feature = st.selectbox("Choose a numeric feature", numeric)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=detected_target, y=feature, ax=ax)
            ax.set_title(f"{feature} by placement outcome")
            st.pyplot(fig)
            plt.close(fig)
            if len(numeric) > 1:
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.heatmap(df[numeric].corr(), cmap="coolwarm", center=0, ax=ax)
                ax.set_title("Numeric feature correlation")
                st.pyplot(fig)
                plt.close(fig)

with lab:
    st.subheader("Train and compare machine-learning models")
    target = st.selectbox("Choose target column", list(df.columns),
                          index=list(df.columns).index(detected_target) if detected_target else 0)
    identifiers = [c for c in df.columns if str(c).strip().lower() in
                   {"college_id", "student_id", "id", "sl_no", "serial_no"}]
    default_features = [c for c in df.columns if c != target and c not in identifiers]
    features = st.multiselect("Select input features", [c for c in df.columns if c != target],
                              default=default_features)
    test_size = st.slider("Test data proportion", 0.15, 0.40, 0.20, 0.05)
    if st.button("Train and compare models", type="primary"):
        if not features:
            st.error("Select at least one feature.")
        else:
            work = df[features + [target]].drop_duplicates().dropna(subset=[target]).copy()
            y, positive = clean_target(work[target])
            X = work[features].copy()
            X = X.drop(columns=[c for c in X.columns if X[c].isna().all()], errors="ignore")
            X = X.drop(columns=[c for c in X.columns if str(c).strip().lower() in
                                {"college_id", "student_id", "id", "sl_no", "serial_no"}], errors="ignore")
            if y.nunique() < 2 or y.value_counts().min() < 2 or X.empty:
                st.error("Need at least two target classes (two rows per class) and usable feature columns.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y)
                numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
                categorical = [c for c in X_train.columns if c not in numeric]
                models = {}
                rows = []
                for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
                    pipe = build_pipeline(name, numeric, categorical)
                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_test)
                    models[name] = pipe
                    rows.append({
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, pred),
                        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                        "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                        "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0)
                    })
                scores = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
                best_name = scores.iloc[0]["Model"]
                best_model = models[best_name]
                best_pred = best_model.predict(X_test)
                st.session_state["model"] = best_model
                st.session_state["features"] = X.columns.tolist()
                st.session_state["reference"] = X.copy()
                st.session_state["positive"] = positive
                st.session_state["best_name"] = best_name
                st.success(f"Best model by weighted F1-score: {best_name}")
                st.dataframe(scores.style.format({x: "{:.3f}" for x in ["Accuracy", "Precision", "Recall", "F1-score"]}),
                             use_container_width=True)
                fig, ax = plt.subplots()
                scores.set_index("Model")[["Accuracy", "F1-score"]].plot(kind="bar", ax=ax)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Score")
                ax.set_title("Model comparison")
                st.pyplot(fig)
                plt.close(fig)
                labels = list(pd.unique(y_test))
                cm = confusion_matrix(y_test, best_pred, labels=labels)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion matrix — best model")
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("#### Classification report")
                st.code(classification_report(y_test, best_pred, zero_division=0))
                with st.spinner("Calculating permutation feature importance..."):
                    perm = permutation_importance(best_model, X_test, y_test, n_repeats=5,
                                                  random_state=42, scoring="f1_weighted")
                importance = pd.DataFrame({"Feature": X.columns,
                                           "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)
                fig, ax = plt.subplots()
                sns.barplot(data=importance.head(12), x="Importance", y="Feature", ax=ax)
                ax.set_title("Permutation feature importance")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Importance shows how much this model's score changes when a feature is shuffled; it does not prove causation.")
    if "model" in st.session_state:
        st.info("Trained model ready: " + st.session_state["best_name"])

with single:
    st.subheader("Predict for one student")
    if "model" not in st.session_state:
        st.info("Train a model in the Model Lab tab first.")
    else:
        model = st.session_state["model"]
        features = st.session_state["features"]
        reference = st.session_state["reference"]
        values = {}
        with st.form("student_form"):
            cols = st.columns(2)
            for i, col in enumerate(features):
                series = reference[col]
                with cols[i % 2]:
                    if pd.api.types.is_numeric_dtype(series):
                        valid = pd.to_numeric(series, errors="coerce").dropna()
                        default = float(valid.median()) if len(valid) else 0.0
                        lo = float(valid.min()) if len(valid) else 0.0
                        hi = float(valid.max()) if len(valid) else max(default + 1, 1.0)
                        if lo == hi:
                            hi = lo + 1.0
                        values[col] = st.number_input(str(col), min_value=lo, max_value=hi,
                                                      value=min(max(default, lo), hi))
                    else:
                        opts = sorted(series.dropna().astype(str).unique().tolist()) or ["Unknown"]
                        values[col] = st.selectbox(str(col), opts)
            submit = st.form_submit_button("Predict placement", type="primary")
        if submit:
            input_df = pd.DataFrame([values], columns=features)
            prediction = model.predict(input_df)[0]
            if str(prediction).lower() in {"placed", "yes", "1", "true", "selected"}:
                st.success(f"Predicted outcome: {prediction}")
            else:
                st.warning(f"Predicted outcome: {prediction}")
                st.markdown("#### Suggested areas to improve")
                for tip in tips_for_student(values, features):
                    st.write("• " + tip)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                prob_df = pd.DataFrame({"Outcome": model.classes_, "Model probability": probs})
                st.dataframe(prob_df, use_container_width=True)
                fig, ax = plt.subplots()
                ax.bar(prob_df["Outcome"].astype(str), prob_df["Model probability"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Model probability")
                ax.set_title("Predicted class probabilities")
                st.pyplot(fig)
                plt.close(fig)
            st.caption("A model probability is not a guarantee of employment or an objective measure of a student's potential.")

with batch:
    st.subheader("Predict for multiple students")
    if "model" not in st.session_state:
        st.info("Train a model in the Model Lab tab first.")
    else:
        batch_file = st.file_uploader("Upload students CSV (same feature columns)", type=["csv"], key="batch")
        if batch_file:
            incoming = pd.read_csv(batch_file)
            required = st.session_state["features"]
            missing = [c for c in required if c not in incoming.columns]
            if missing:
                st.error("CSV is missing these columns: " + ", ".join(missing))
            else:
                batch_X = incoming[required].copy()
                model = st.session_state["model"]
                predicted = model.predict(batch_X)
                result = incoming.copy()
                result["Predicted_Placement"] = predicted
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(batch_X)
                    positive = st.session_state.get("positive")
                    classes = list(model.classes_)
                    if positive in classes:
                        result["Placement_Probability"] = probabilities[:, classes.index(positive)]
                    else:
                        result["Max_Class_Probability"] = probabilities.max(axis=1)
                st.dataframe(result, use_container_width=True)
                st.download_button("Download prediction results",
                                   result.to_csv(index=False).encode("utf-8"),
                                   "careerpredict_results.csv", "text/csv")

st.divider()
st.caption("Educational prototype. Before real institutional use, validate with representative, consented data and audit for bias. Do not use this model as the sole basis for student selection.")
