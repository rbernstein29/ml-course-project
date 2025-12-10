"""
Machine Learning Analysis for Customer Segmentation
CSC156 Course Project - Phase 3
Parsa Jafaripour

This script performs comprehensive machine learning analysis including:
- Data preprocessing and exploration  
- Multiple model training
- Hyperparameter tuning with GridSearchCV
- Learning curves and validation curves
- Model evaluation on training and test sets
- Visualization generation for report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, validation_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import warnings
import os
warnings.filterwarnings("ignore")

def run_multiclass():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10

    if not os.path.exists("figures"):
        os.makedirs("figures")

    print("="*80)
    print("MACHINE LEARNING ANALYSIS - CUSTOMER SEGMENTATION")
    print("="*80)

    # 1. LOAD DATA
    print("\n1. LOADING DATA...")
    train = pd.read_csv("multiclass/code/train.csv")
    test = pd.read_csv("multiclass/code/test.csv")
    TARGET_COL = "Segmentation"

    print(f"Training set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print(f"\nTarget distribution:\n{train[TARGET_COL].value_counts().sort_index()}")

    plt.figure(figsize=(10, 6))
    train[TARGET_COL].value_counts().sort_index().plot(kind="bar", color="steelblue")
    plt.title("Class Distribution in Training Set", fontsize=14, fontweight="bold")
    plt.xlabel("Customer Segment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved class distribution plot")

    # 2. PREPARE DATA  
    print("\n2. PREPARING DATA...")
    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    if "ID" in X.columns:
        X = X.drop(columns=["ID"])
    if "ID" in test.columns:
        X_test_full = test.drop(columns=["ID"])
    else:
        X_test_full = test.copy()

    numeric_features = ["Age", "Work_Experience", "Family_Size"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # 3. CREATE PREPROCESSING PIPELINES
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_transformer_scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    numeric_transformer_unscaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor_scaled = ColumnTransformer([
        ("num", numeric_transformer_scaled, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    preprocessor_unscaled = ColumnTransformer([
        ("num", numeric_transformer_unscaled, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # 4. TRAIN/VALIDATION SPLIT
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # 5. DEFINE MODELS
    print("\n5. DEFINING MODELS AND HYPERPARAMETER GRIDS")
    models = {}

    #models["Decision Tree"] = {
    #    "pipeline": Pipeline([("preprocessor", preprocessor_unscaled), ("classifier", DecisionTreeClassifier(random_state=42))]),
    #   "param_grid": {
    #       "classifier__max_depth": [5, 10, 15, None],
    #        "classifier__min_samples_split": [2, 10],
    #        "classifier__min_samples_leaf": [1, 5],
    #        "classifier__criterion": ["gini", "entropy"]
    #    }
    #}

    models["Random Forest"] = {
        "pipeline": Pipeline([("preprocessor", preprocessor_unscaled), ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))]),
        "param_grid": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }
    }

    # KNN removed due to compatibility issues with string labels
    # models["K-Nearest Neighbors"] = {
    #     "pipeline": Pipeline([("preprocessor", preprocessor_scaled), ("classifier", KNeighborsClassifier())]),
    #     "param_grid": {
    #         "classifier__n_neighbors": [3, 5, 7, 9, 11],
    #         "classifier__weights": ["uniform", "distance"],
    #         "classifier__metric": ["euclidean", "manhattan"]
    #     }
    # }

    models["Support Vector Machine"] = {
        "pipeline": Pipeline([("preprocessor", preprocessor_scaled), ("classifier", SVC(random_state=42))]),
        "param_grid": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["rbf", "poly"],
            "classifier__gamma": ["scale", "auto"]
        }
    }

    models["Neural Network"] = {
        "pipeline": Pipeline([("preprocessor", preprocessor_scaled), ("classifier", MLPClassifier(max_iter=300, random_state=42, early_stopping=False, solver='lbfgs'))]),
        "param_grid": {
            "classifier__hidden_layer_sizes": [(50,), (100,)],
            "classifier__alpha": [0.0001, 0.001]
        }
    }

    #models["Gradient Boosting"] = {
    #    "pipeline": Pipeline([("preprocessor", preprocessor_unscaled), ("classifier", GradientBoostingClassifier(random_state=42))]),
    #   "param_grid": {
    #       "classifier__n_estimators": [50, 100],
    #        "classifier__learning_rate": [0.01, 0.1],
    #        "classifier__max_depth": [3, 5],
    #        "classifier__subsample": [0.8, 1.0]
    #    }
    #}

    #models["Logistic Regression"] = {
    #    "pipeline": Pipeline([("preprocessor", preprocessor_scaled), ("classifier", LogisticRegression(random_state=42, max_iter=1000, multi_class="multinomial"))]),
    #    "param_grid": {
    #        "classifier__C": [0.01, 0.1, 1, 10],
    #        "classifier__penalty": ["l2"],
    #        "classifier__solver": ["lbfgs", "saga"]
    #    }
    #}

    print(f"Total models to train: {len(models)}")

    # 6. HYPERPARAMETER TUNING
    print("\n6. PERFORMING HYPERPARAMETER TUNING")
    results = []
    best_models = {}

    for model_name, model_config in models.items():
        print(f"\nTraining: {model_name}")
        
        grid_search = GridSearchCV(
            estimator=model_config["pipeline"],
            param_grid=model_config["param_grid"],
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model
        
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        train_f1 = f1_score(y_train, y_train_pred, average="weighted")
        val_f1 = f1_score(y_val, y_val_pred, average="weighted")
        
        results.append({
            "Model": model_name,
            "Best Params": grid_search.best_params_,
            "Train Accuracy": train_acc,
            "Val Accuracy": val_acc,
            "Train F1": train_f1,
            "Val F1": val_f1,
            "Val Precision": precision_score(y_val, y_val_pred, average="weighted"),
            "Val Recall": recall_score(y_val, y_val_pred, average="weighted"),
            "CV Best Score": grid_search.best_score_
        })
        
        print(f"✓ {model_name} Results:")
        print(f"  Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"  Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    # 7. RESULTS SUMMARY
    print("\n7. RESULTS SUMMARY")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Val F1", ascending=False)
    print(results_df[["Model", "Train Accuracy", "Val Accuracy", "Train F1", "Val F1"]].to_string(index=False))
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("✓ Saved results to model_comparison_results.csv")

    # 8. VISUALIZATIONS
    print("\n8. GENERATING VISUALIZATIONS")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    results_df.set_index("Model")[["Train Accuracy", "Val Accuracy"]].plot(kind="bar", ax=axes[0], color=["#2ecc71", "#3498db"])
    axes[0].set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(["Training", "Validation"])
    axes[0].set_ylim([0.4, 1.0])
    axes[0].tick_params(axis="x", rotation=45)

    results_df.set_index("Model")[["Train F1", "Val F1"]].plot(kind="bar", ax=axes[1], color=["#e74c3c", "#f39c12"])
    axes[1].set_title("Model F1 Score Comparison", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend(["Training", "Validation"])
    axes[1].set_ylim([0.4, 1.0])
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved model comparison plot")

    best_model_name = results_df.iloc[0]["Model"]
    best_model = best_models[best_model_name]
    y_val_pred = best_model.predict(X_val)
    cm = confusion_matrix(y_val, y_val_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/confusion_matrix_best_model.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved confusion matrix for {best_model_name}")

    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_val, y_val_pred))

    # 9. LEARNING CURVES
    print("\n9. GENERATING LEARNING CURVES")
    top_models = results_df.head(3)["Model"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, model_name in enumerate(top_models[:4]):
        print(f"Generating learning curve for {model_name}...")
        model = best_models[model_name]
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="f1_weighted", shuffle=True, random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[idx].plot(train_sizes, train_mean, "o-", color="#2ecc71", label="Training score")
        axes[idx].plot(train_sizes, val_mean, "o-", color="#3498db", label="Validation score")
        axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#2ecc71")
        axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="#3498db")
        axes[idx].set_title(f"Learning Curve - {model_name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Training Set Size")
        axes[idx].set_ylabel("F1 Score")
        axes[idx].legend(loc="best")
        axes[idx].grid(True, alpha=0.3)

    if len(top_models) < 4:
        for idx in range(len(top_models), 4):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved learning curves")

    # 10. VALIDATION CURVES
    print("\n10. GENERATING VALIDATION CURVES")

    # Decision Tree - max_depth
    #print("Generating validation curve for Decision Tree...")
    #dt_model = models["Decision Tree"]["pipeline"]
    #param_range = [3, 5, 7, 10, 15, 20, None]
    #train_scores, val_scores = validation_curve(
    #    dt_model, X_train, y_train, param_name="classifier__max_depth",
    #    param_range=param_range, cv=cv, scoring="f1_weighted", n_jobs=-1
    #)

    #param_range_plot = [str(x) if x is not None else "None" for x in param_range]
    #plt.figure(figsize=(10, 6))
    #train_mean = np.mean(train_scores, axis=1)
    #val_mean = np.mean(val_scores, axis=1)
    #plt.plot(param_range_plot, train_mean, "o-", color="#2ecc71", label="Training score")
    #plt.plot(param_range_plot, val_mean, "o-", color="#3498db", label="Validation score")
    #plt.title("Validation Curve - Decision Tree (max_depth)", fontsize=14, fontweight="bold")
    #plt.xlabel("max_depth")
    #plt.ylabel("F1 Score")
    #plt.legend(loc="best")
    #plt.grid(True, alpha=0.3)
    #plt.tight_layout()
    #plt.savefig("../../plots/multiclass_plots/validation_curve_dt_maxdepth.png", dpi=300, bbox_inches="tight")
    #plt.close()
    #print("✓ Saved validation curve for Decision Tree")

    # Random Forest - n_estimators
    print("Generating validation curve for Random Forest...")
    rf_model = models["Random Forest"]["pipeline"]
    param_range = [10, 50, 100, 150, 200, 300]
    train_scores, val_scores = validation_curve(
        rf_model, X_train, y_train, param_name="classifier__n_estimators",
        param_range=param_range, cv=cv, scoring="f1_weighted", n_jobs=-1
    )

    plt.figure(figsize=(10, 6))
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    plt.plot(param_range, train_mean, "o-", color="#2ecc71", label="Training score")
    plt.plot(param_range, val_mean, "o-", color="#3498db", label="Validation score")
    plt.title("Validation Curve - Random Forest (n_estimators)", fontsize=14, fontweight="bold")
    plt.xlabel("n_estimators")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/validation_curve_rf_nestimators.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved validation curve for Random Forest")

    # KNN validation curve removed since KNN model was removed
    # (sklearn version compatibility issues with string labels)

    # 11. RETRAIN BEST MODEL
    print("\n11. RETRAINING BEST MODEL ON FULL TRAINING SET")
    best_model_name = results_df.iloc[0]["Model"]
    final_model = best_models[best_model_name]

    print(f"Best Model: {best_model_name}")
    final_model.fit(X, y)

    y_train_final_pred = final_model.predict(X)
    final_train_acc = accuracy_score(y, y_train_final_pred)
    final_train_f1 = f1_score(y, y_train_final_pred, average="weighted")

    print(f"✓ Final Training Accuracy: {final_train_acc:.4f}")
    print(f"✓ Final Training F1 Score: {final_train_f1:.4f}")

    # 12. EVALUATE ON TEST SET
    print("\n12. EVALUATING ON TEST SET")
    test_predictions = final_model.predict(X_test_full)

    submission = pd.DataFrame({"ID": test["ID"], "Segmentation": test_predictions})
    submission.to_csv("test_predictions.csv", index=False)
    print("✓ Saved test predictions to test_predictions.csv")

    print("\nTest set prediction distribution:")
    print(pd.Series(test_predictions).value_counts().sort_index())

    plt.figure(figsize=(10, 6))
    pd.Series(test_predictions).value_counts().sort_index().plot(kind="bar", color="coral")
    plt.title("Predicted Class Distribution in Test Set", fontsize=14, fontweight="bold")
    plt.xlabel("Customer Segment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/test_predictions_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved test predictions distribution plot")

    # 13. FINAL SUMMARY
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    print(f"""
    Dataset: Customer Segmentation
    Training samples: {len(train)}
    Test samples: {len(test)}
    Number of features: {X.shape[1]}
    Classes: {sorted(y.unique())}

    BEST MODEL: {best_model_name}
    Validation F1 Score: {results_df.iloc[0]["Val F1"]:.4f}
    Final Training F1 Score: {final_train_f1:.4f}

    GENERATED FILES:
    - class_distribution.png
    - model_comparison.png
    - confusion_matrix_best_model.png
    - learning_curves.png
    - validation_curve_rf_nestimators.png
    - test_predictions_distribution.png
    """)

    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
