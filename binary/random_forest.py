import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from .data_handler import load_data
from os.path import join


def train_random_forest(num_trees, max_depth, k_folds=5):
    X_train, y_train, X_test, y_test = load_data()
    
    model = RandomForestClassifier(random_state=0, n_estimators=num_trees, max_depth=max_depth, n_jobs=-1)
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=k_folds, scoring="accuracy")

    return model, cv_accuracy


def run_random_forest():
    X_train, y_train, X_test, y_test = load_data()
    
    best_accuracy = 0
    optimal_num_trees = 0
    optimal_max_depth = 0
    
    num_trees = 10
    num_trees_step = 10
    num_trees_max = 50
    
    max_depth_values = [10, 20, 30, 40, 50, None]

    tuning_plot, ax = plt.subplots()
    testing_plot, bx = plt.subplots()
    
    while num_trees <= num_trees_max:
        depths = []
        accuracies = []
        test_scores = []
        for depth in max_depth_values:
            model, cv_accuracy = train_random_forest(num_trees, depth)
            model.fit(X_train, y_train)
            mean_accuracy = cv_accuracy.mean()
            test_score = model.score(X_test, y_test)

            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                optimal_num_trees = num_trees
                optimal_max_depth = depth

            depths.append(depth)
            accuracies.append(mean_accuracy)
            test_scores.append(test_score)

        ax.plot(depths, accuracies, label=f"{num_trees} trees")
        bx.plot(depths, test_scores, label=f"{num_trees} trees")
        num_trees += num_trees_step

    ax.set_xlabel("Maximum Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Max Tree Depth Tuning Plot")
    ax.legend()

    bx.set_xlabel("Maximum Depth")
    bx.set_ylabel("Accuracy")
    bx.set_title(f"Max Tree Depth Tuning Test Plot")
    bx.legend()

    tuning_plot.savefig(join("plots/binary_plots", "random_forest_tuning_plot.png"))
    testing_plot.savefig(join("plots/binary_plots", "random_forest_testing_plot.png"))
    
    final_model = RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=optimal_num_trees, max_depth=optimal_max_depth)
    final_model.fit(X_train, y_train)
    train_score = final_model.score(X_train, y_train)
    test_score = final_model.score(X_test, y_test)
    
    print(f"\n--------------Random Forest--------------")

    print(f"Training score (optimal model): {train_score:.4f}")
    print(f"Test score (optimal model): {test_score:.4f}")
    print(f"Best cross-validation accuracy: {best_accuracy:.4f}")
    print(f"Optiaml number of trees: {optimal_num_trees}")
    print(f"Optiaml max depth: {optimal_max_depth}\n")

    return final_model


#run_random_forest()