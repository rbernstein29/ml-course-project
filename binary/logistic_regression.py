import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score
from .data_handler import load_data
from os.path import join


def train_logistic_regression(C, solver, k_folds=5):
   X_train, y_train, X_test, y_test = load_data()
   
   model = LogisticRegression(random_state=0, C=C, solver=solver)
   cv_scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring="accuracy")

   return model, cv_scores


def run_logistic_regression():
    solvers = ["lbfgs", "liblinear", "saga", "newton-cg"]
    X_train, y_train, X_test, y_test = load_data()
   
    best_score = 0
    optimal_C = 0
    optimal_solver = None

    C = 0.0001
    C_max = 10000
    C_step = 10

    tuning_plot, ax = plt.subplots()
    test_plot, bx = plt.subplots()

    for solver in solvers:
        C = 0.0001
        C_vals = []
        scores = []
        test_scores = []
        while C <= C_max:
            model, cv_scores = train_logistic_regression(C, solver)
            model.fit(X_train, y_train)
            mean_score = cv_scores.mean()
            test_score = model.score(X_test, y_test)

            if mean_score > best_score:
                best_score = mean_score
                optimal_C = C
                optimal_solver = solver
            
            C_vals.append(np.log10(C))
            scores.append(mean_score)
            test_scores.append(test_score)

            C *= C_step

        ax.plot(C_vals, scores, label=f"{solver} solver")
        bx.plot(C_vals, test_scores, label=f"{solver} solver")

    ax.set_xlabel("Regulaization Strength (logC)")
    ax.set_ylabel("Accuracy Score")
    ax.set_title(f"Regulaization Strength Tuning Plot")
    ax.legend()

    bx.set_xlabel("Regulaization Strength (logC)")
    bx.set_ylabel("Accuracy Score")
    bx.set_title(f"Regularization Tuning Test Plot")
    bx.legend()

    tuning_plot.savefig(join("plots/binary_plots", "logistic_regresion_tuning_plot.png"))
    test_plot.savefig(join("plots/binary_plots", "logistic_regression_testing_plot.png"))


    optimal_model = LogisticRegression(random_state=0, C=optimal_C, solver=optimal_solver)
    optimal_model.fit(X_train, y_train)
    train_score = optimal_model.score(X_train, y_train)
    test_score = optimal_model.score(X_test, y_test)

    print("\n--------------Logistic Regression--------------")

    print(f"Training score (optimal model): {train_score:.4f}")
    print(f"Test score (optimal model): {test_score:.4f}")
    print(f"Best cross-validation score: {best_score:.4f}")
    print(f"Optimal C: {optimal_C}")
    print(f"Optimal Solver: {optimal_solver}\n")

    return optimal_model
    

#run_logistic_regression()