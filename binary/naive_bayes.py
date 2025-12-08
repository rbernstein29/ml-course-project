import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from data_handler import load_data
from os.path import join


def train_naive_bayes(smoothing_constant, learn_priors, k_folds=5):
    X_train, y_train, X_test, y_test = load_data()
    
    model = MultinomialNB(alpha=smoothing_constant, fit_prior=learn_priors)
    cv_scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring="accuracy")

    return model, cv_scores


def run_naive_bayes():
    X_train, y_train, X_test, y_test = load_data()
    
    best_score = 0
    optimal_alpha = 0
    optimal_fit_prior = True

    alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    bools = [True, False]

    tuning_plot, ax = plt.subplots()
    test_plot, bx = plt.subplots()
    
    for bool in bools:
        alphas = []
        scores = []
        test_scores = []

        for alpha in alpha_values:
            model, cv_scores = train_naive_bayes(alpha, bool)
            model.fit(X_train, y_train)
            mean_score = cv_scores.mean()
            test_score = model.score(X_test, y_test)
            
            if mean_score > best_score:
                best_score = mean_score
                optimal_alpha = alpha
                optimal_fit_prior = bool

            alphas.append(np.log10(alpha))
            scores.append(mean_score)
            test_scores.append(test_score)

        ax.plot(alphas, scores, label=f"Learn Priors = {bool}")
        bx.plot(alphas, test_scores, label=f"Learn Priors = {bool}")

    ax.set_xlabel("Smoothing Factor (log(alpha))")
    ax.set_ylabel("Accuracy Score")
    ax.set_title(f"Smoothing Tuning Plot")
    ax.legend()

    bx.set_xlabel("Smoothing Factor (log(alpha))")
    bx.set_ylabel("Accuracy Score")
    bx.set_title(f"Smoothing Tuning Test Plot")
    bx.legend()

    tuning_plot.savefig(join("plots/naive_bayes_plots", "naive_bayes_tuning_plot.png"))
    test_plot.savefig(join("plots/naive_bayes_plots", "naive_bayed_testing_plot.png"))

    
    final_model = MultinomialNB(alpha=optimal_alpha, fit_prior=optimal_fit_prior)
    final_model.fit(X_train, y_train)
    train_score = final_model.score(X_train, y_train)
    test_score = final_model.score(X_test, y_test)
    
    print(f"\n--------------Multinomial Naive Bayes--------------")

    print(f"Training score (optimal model): {train_score:.4f}")
    print(f"Test score (optimal model): {test_score:.4f}")
    print(f"Best cross-validation score: {best_score:.4f}")
    print(f"Optiaml smoothing factor: {optimal_alpha}")
    print(f"Fit priors: {optimal_fit_prior}\n")

    return final_model


#run_naive_bayes()