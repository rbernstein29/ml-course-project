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

    # 1. LOAD DATA
    print("\n1. LOADING DATA...")
    train = pd.read_csv("multiclass/code/train.csv")
    test = pd.read_csv("multiclass/code/test.csv")
    TARGET_COL = "Segmentation"

    plt.figure(figsize=(10, 6))
    train[TARGET_COL].value_counts().sort_index().plot(kind="bar", color="steelblue")
    plt.title("Class Distribution in Training Set", fontsize=14, fontweight="bold")
    plt.xlabel("Customer Segment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    # 5. DEFINE MODELS
    print("\n5. DEFINING MODELS AND HYPERPARAMETER GRIDS")
    models = {}

    models["Random Forest"] = {
        "pipeline": Pipeline([("preprocessor", preprocessor_unscaled), ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))]),
        "param_grid": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }
    }

    models["Support Vector Machine"] = {
        "pipeline": Pipeline([("preprocessor", preprocessor_scaled), ("classifier", SVC(random_state=42))]),
        "param_grid": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__kernel": ["rbf"],#, "poly"],
            "classifier__gamma": [0.001, 0.01, 0.1, 1]
        }
    }

    models["Neural Network"] = {
        "pipeline": Pipeline([("preprocessor", preprocessor_scaled), ("classifier", MLPClassifier(max_iter=300, random_state=42, early_stopping=False, solver='adam'))]),
        "param_grid": {
            "classifier__hidden_layer_sizes": [(50,), (100,)],
            "classifier__alpha": [0.0001, 0.001]
        }
    }


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
        y_test_pred = best_model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average="weighted")
        test_f1 = f1_score(y_test, y_test_pred, average="weighted")
        
        results.append({
            "Model": model_name,
            "Best Params": grid_search.best_params_,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Train F1": train_f1,
            "Test F1": test_f1,
            "CV Best Score": grid_search.best_score_
        })
        
        print(f"{model_name} Results:")
        print(f"  Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"  Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Optimal Hyperparameters: {grid_search.best_params_}")

    # 7. RESULTS SUMMARY
    print("\n7. RESULTS SUMMARY")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Test F1", ascending=False)
    print(results_df[["Model", "Train Accuracy", "Test Accuracy", "Train F1", "Test F1"]].to_string(index=False))
    #results_df.to_csv("model_comparison_results.csv", index=False)

    # 8. VISUALIZATIONS
    print("\n8. GENERATING VISUALIZATIONS")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    results_df.set_index("Model")[["Train Accuracy", "Test Accuracy"]].plot(kind="bar", ax=axes[0], color=["#2ecc71", "#3498db"])
    axes[0].set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(["Training", "Validation"])
    axes[0].set_ylim([0.4, 1.0])
    axes[0].tick_params(axis="x", rotation=45)

    results_df.set_index("Model")[["Train F1", "Test F1"]].plot(kind="bar", ax=axes[1], color=["#e74c3c", "#f39c12"])
    axes[1].set_title("Model F1 Score Comparison", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend(["Training", "Testing"])
    axes[1].set_ylim([0.4, 1.0])
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


    # 9. LOG LOSS CURVE
    print("\n9. GENERATING LOG LOSS CURVE")

    plt.figure(figsize=(12, 7))

    # Use the hyperparameters from your param_grid
    configs = [
        {"hidden_layer_sizes": (50,), "alpha": 0.0001, "label": "50 neurons, α=0.0001", "color": "#e74c3c"},
        {"hidden_layer_sizes": (50,), "alpha": 0.001, "label": "50 neurons, α=0.001", "color": "#3498db"},
        {"hidden_layer_sizes": (100,), "alpha": 0.0001, "label": "100 neurons, α=0.0001", "color": "#2ecc71"},
        {"hidden_layer_sizes": (100,), "alpha": 0.001, "label": "100 neurons, α=0.001", "color": "#f39c12"},
    ]

    for config in configs:
        nn_pipeline = models["Neural Network"]["pipeline"]
        
        nn_pipeline.fit(X_train, y_train)
        loss_curve = nn_pipeline.named_steps['classifier'].loss_curve_
        
        plt.plot(range(1, len(loss_curve) + 1), loss_curve, 
                "o-", color=config["color"], label=config["label"], 
                linewidth=2, markersize=3, alpha=0.8)

    plt.title("Training Log Loss Curve - Neural Network", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Log Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/log_loss_nn.png", dpi=300, bbox_inches="tight")
    plt.close()


    # 10. VALIDATION CURVES
    print("\n10. GENERATING VALIDATION CURVES")

    # Random Forest - n_estimators
    rf_model = models["Random Forest"]["pipeline"]
    param_range = [10, 50, 100, 150, 200, 300]
    train_scores, val_scores = validation_curve(
        rf_model, X_train, y_train, param_name="classifier__n_estimators",
        param_range=param_range, cv=cv, scoring="f1_weighted", n_jobs=-1
    )

    plt.figure(figsize=(10, 6))
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    plt.plot(param_range, train_mean, "o-", color="#e74c3c", label="Training score")
    plt.plot(param_range, val_mean, "o-", color="#3498db", label="Validation score")
    plt.title("Tuning Curve - Random Forest (n_estimators)", fontsize=14, fontweight="bold")
    plt.xlabel("n_estimators")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/tuning_curve_rf_nestimators.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Random Forest - max tree depth
    rf_model = models["Random Forest"]["pipeline"]
    param_range = [3, 5, 10, 15, 20, 25, 30, 40, 50]
    train_scores, val_scores = validation_curve(
        rf_model, X_train, y_train, param_name="classifier__max_depth",
        param_range=param_range, cv=cv, scoring="f1_weighted", n_jobs=-1
    )

    plt.figure(figsize=(10, 6))
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    plt.plot(param_range, train_mean, "o-", color="#e74c3c", label="Training score", linewidth=2, markersize=8)
    plt.plot(param_range, val_mean, "o-", color="#3498db", label="Validation score", linewidth=2, markersize=8)
    plt.title("Tuning Curve - Random Forest (max_depth)", fontsize=14, fontweight="bold")
    plt.xlabel("max_depth")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/tuning_curve_rf_maxdepth.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Validation Curve for SVM - C parameter
    svm_base = models["Support Vector Machine"]["pipeline"]
    param_range_c = [0.01, 0.1, 1, 10, 100]
    kernels = ["rbf"]
    colors = {"train": "#e74c3c", "val": "#3498db"}

    plt.figure(figsize=(12, 7))

    for kernel in kernels:
        svm_model = svm_base
        
        train_scores, val_scores = validation_curve(
            svm_model, X_train, y_train, param_name="classifier__C",
            param_range=param_range_c, cv=cv, scoring="f1_weighted", n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        
        # Plot training scores (lighter, dashed)
        plt.plot(param_range_c, train_mean, "o-", color=colors["train"], 
                linewidth=2.5, markersize=6, label="Training")
        
        # Plot validation scores (solid, prominent)
        plt.plot(param_range_c, val_mean, "o-", color=colors["val"], 
                linewidth=2.5, markersize=6, label="Validation")

    plt.title("Tuning Curve - SVM (C)", fontsize=14, fontweight="bold")
    plt.xlabel("Regularization (logC)", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/tuning_curve_svm_c.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Validation Curve for SVM - gamma parameter
    param_range_gamma = [0.001, 0.01, 0.1, 1]
    kernels_gamma = ["rbf"] 

    plt.figure(figsize=(12, 7))

    for kernel in kernels_gamma:
        svm_model = svm_base
        
        train_scores, val_scores = validation_curve(
            svm_model, X_train, y_train, param_name="classifier__gamma",
            param_range=param_range_gamma, cv=cv, scoring="f1_weighted", n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        
        plt.plot(param_range_gamma, train_mean, "o-", color=colors["train"], 
                linewidth=2.5, markersize=6, label="Training")
        
        plt.plot(param_range_gamma, val_mean, "o-", color=colors["val"], 
                linewidth=2.5, markersize=6, label="Validation")

    plt.title("Tuning Curve - SVM (gamma)", fontsize=14, fontweight="bold")
    plt.xlabel("Gamma (logGamma)", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/tuning_curve_svm_gamma.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Validation Curve for Neural Network - hidden_layer_sizes
    nn_model = models["Neural Network"]["pipeline"]
    param_range_layers = [(10,), (25,), (50,), (75,), (100,), (150,), (200,)]
    train_scores, val_scores = validation_curve(
        nn_model, X_train, y_train, param_name="classifier__hidden_layer_sizes",
        param_range=param_range_layers, cv=cv, scoring="f1_weighted", n_jobs=-1
    )

    plt.figure(figsize=(10, 6))
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    # Convert tuples to strings for x-axis labels
    param_labels = [str(p[0]) for p in param_range_layers]
    x_pos = np.arange(len(param_labels))
    plt.plot(x_pos, train_mean, "o-", color="#e74c3c", label="Training score")
    plt.plot(x_pos, val_mean, "o-", color="#3498db", label="Validation score")
    plt.title("Tuning Curve - Neural Network (hidden_layer_sizes)", fontsize=14, fontweight="bold")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("F1 Score")
    plt.xticks(x_pos, param_labels)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/tuning_curve_nn_layers.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Validation Curve for Neural Network - alpha
    param_range_alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    train_scores, val_scores = validation_curve(
        nn_model, X_train, y_train, param_name="classifier__alpha",
        param_range=param_range_alpha, cv=cv, scoring="f1_weighted", n_jobs=-1
    )

    plt.figure(figsize=(10, 6))
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    plt.plot(param_range_alpha, train_mean, "o-", color="#e74c3c", label="Training score")
    plt.plot(param_range_alpha, val_mean, "o-", color="#3498db", label="Validation score")
    plt.title("Tuning Curve - Neural Network (alpha)", fontsize=14, fontweight="bold")
    plt.xlabel("Alpha (logAlpha)")
    plt.ylabel("F1 Score")
    plt.xscale('log')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/multiclass_plots/tuning_curve_nn_alpha.png", dpi=300, bbox_inches="tight")
    plt.close()


    # # 11. RETRAIN BEST MODEL
    # print("\n11. RETRAINING BEST MODEL ON FULL TRAINING SET")
    # best_model_name = results_df.iloc[0]["Model"]
    # final_model = best_models[best_model_name]

    # print(f"Best Model: {best_model_name}")
    # final_model.fit(X, y)

    # y_train_final_pred = final_model.predict(X)
    # final_train_acc = accuracy_score(y, y_train_final_pred)
    # final_train_f1 = f1_score(y, y_train_final_pred, average="weighted")

    # print(f"✓ Final Training Accuracy: {final_train_acc:.4f}")
    # print(f"✓ Final Training F1 Score: {final_train_f1:.4f}")

    # # 12. EVALUATE ON TEST SET
    # print("\n12. EVALUATING ON TEST SET")
    # test_predictions = final_model.predict(X_test_full)

    # submission = pd.DataFrame({"ID": test["ID"], "Segmentation": test_predictions})
    # submission.to_csv("test_predictions.csv", index=False)

    # print("\nTest set prediction distribution:")
    # print(pd.Series(test_predictions).value_counts().sort_index())

    # plt.figure(figsize=(10, 6))
    # pd.Series(test_predictions).value_counts().sort_index().plot(kind="bar", color="coral")
    # plt.title("Predicted Class Distribution in Test Set", fontsize=14, fontweight="bold")
    # plt.xlabel("Customer Segment")
    # plt.ylabel("Count")
    # plt.tight_layout()
    # plt.savefig("plots/multiclass_plots/test_predictions_distribution.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # # 13. FINAL SUMMARY
    # print("\n" + "-"*80)
    # print("FINAL SUMMARY REPORT")
    # print("-"*80)
    # print(f"""
    # Dataset: Customer Segmentation
    # Training samples: {len(train)}
    # Test samples: {len(test)}
    # Number of features: {X.shape[1]}
    # Classes: {sorted(y.unique())}

    # BEST MODEL: {best_model_name}
    # Validation F1 Score: {results_df.iloc[0]["Val F1"]:.4f}
    # Final Training F1 Score: {final_train_f1:.4f}

    # """)

    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

#run_multiclass()