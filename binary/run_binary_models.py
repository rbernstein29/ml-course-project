import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score
from os.path import join
from .logistic_regression import run_logistic_regression
from .naive_bayes import run_naive_bayes
from .random_forest import run_random_forest
from .data_handler import load_data


def run_binary():
    print("\n---Training Logistic Regression---\n")
    logistic_regression_model = run_logistic_regression()
    print("\n---Training Naive Bayes---\n")
    naive_bayes_model = run_naive_bayes()
    print("\n---Training Random Forest---\n")
    random_forest_model = run_random_forest()

    X_train, y_train, X_test, y_test = load_data()

    lr_training_accuracy = logistic_regression_model.score(X_train, y_train)
    lr_testing_accuracy = logistic_regression_model.score(X_test, y_test)
    lr_y_training_pred = logistic_regression_model.predict(X_train)
    lr_training_recall = recall_score(y_train, lr_y_training_pred)
    lr_y_testing_pred = logistic_regression_model.predict(X_test)
    lr_testing_recall = recall_score(y_test, lr_y_testing_pred)

    nb_training_accuracy = naive_bayes_model.score(X_train, y_train)
    nb_testing_accuracy = naive_bayes_model.score(X_test, y_test)
    nb_y_training_pred = naive_bayes_model.predict(X_train)
    nb_training_recall = recall_score(y_train, nb_y_training_pred)
    nb_y_testing_pred = naive_bayes_model.predict(X_test)
    nb_testing_recall = recall_score(y_test, nb_y_testing_pred)

    rf_training_accuracy = random_forest_model.score(X_train, y_train)
    rf_testing_accuracy = random_forest_model.score(X_test, y_test)
    rf_y_training_pred = random_forest_model.predict(X_train)
    rf_training_recall = recall_score(y_train, rf_y_training_pred)
    rf_y_testing_pred = random_forest_model.predict(X_test)
    rf_testing_recall = recall_score(y_test, rf_y_testing_pred)

    print("\n----------Optimal Models----------\n")
    print("Logistic Regression:")
    print(f"\ttraining accuracy: {lr_training_accuracy:.4f}\n\ttesting accuracy: {lr_testing_accuracy:.4f}")
    print(f"\ttraining recall: {lr_training_recall:.4f}\n\ttesting recall: {lr_testing_recall:.4f}")
    print("Multinomial Naive Bayes:")
    print(f"\ttraining accuracy: {nb_training_accuracy:.4f}\n\ttesting accuracy: {nb_testing_accuracy:.4f}")
    print(f"\ttraining recall: {nb_training_recall:.4f}\n\ttesting recall: {nb_testing_recall:.4f}")
    print("Random Forest:")
    print(f"\ttraining accuracy: {rf_training_accuracy:.4f}\n\ttesting accuracy: {rf_testing_accuracy:.4f}")
    print(f"\ttraining recall: {rf_training_recall:.4f}\n\ttesting recall: {rf_testing_recall:.4f}\n")

    models = ['Logistic Regression', 'Naive Bayes', 'Random Forest']
    
    train_accuracy = [lr_training_accuracy, nb_training_accuracy, rf_training_accuracy]
    test_accuracy = [lr_testing_accuracy, nb_testing_accuracy, rf_testing_accuracy]
    train_recall = [lr_training_recall, nb_training_recall, rf_training_recall]
    test_recall = [lr_testing_recall, nb_testing_recall, rf_testing_recall]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - 1.5*width, train_accuracy, width, label='Training Accuracy')
    bars2 = ax.bar(x - 0.5*width, test_accuracy, width, label='Testing Accuracy')
    bars3 = ax.bar(x + 0.5*width, train_recall, width, label='Training Recall')
    bars4 = ax.bar(x + 1.5*width, test_recall, width, label='Testing Recall')
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(join("plots/binary_plots", "model_comparison.png"))

#run_binary()