================================================================================
CUSTOMER SEGMENTATION - MACHINE LEARNING ANALYSIS
CSC156 Course Project - Phase 3
================================================================================

AUTHOR: Parsa Jafaripour
DATE: December 10, 2025

================================================================================
PROJECT DESCRIPTION
================================================================================

This project performs comprehensive machine learning analysis on a customer
segmentation dataset. The analysis includes:

- Data preprocessing and exploration
- Training and evaluation of 7 different machine learning models:
  * Decision Tree
  * Random Forest
  * K-Nearest Neighbors (KNN)
  * Support Vector Machine (SVM)
  * Neural Network (MLP)
  * Gradient Boosting
  * Logistic Regression
  
- Hyperparameter tuning using GridSearchCV with 5-fold cross-validation
- Learning curves to analyze model performance vs training set size
- Validation curves to analyze hyperparameter impact
- Comprehensive visualizations for report inclusion
- Test set predictions

================================================================================
REQUIREMENTS
================================================================================

Python 3.7 or higher

Required Python Libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

To install all required libraries, run:

    pip install pandas numpy matplotlib seaborn scikit-learn

OR

    pip3 install pandas numpy matplotlib seaborn scikit-learn

===============================================================================
FILES IN THIS PROJECT
===============================================================================

- train.csv: Training dataset with labels (Segmentation column)
- test.csv: Test dataset without labels
- main.py: Main Python script that performs all analyses
- README.txt: This file with instructions

================================================================================
HOW TO RUN THE CODE
================================================================================

1. make sure you have Python 3.7+ installed
2. I would install required libraries (see REQUIREMENTS section above)
3. Navigate to the project directory in your terminal:
   
   cd /path/to/CSC156

4. Run the main.py script using the virtual environment:

   .venv/bin/python main.py
   
   OR (if the above doesn't work, activate the virtual environment first):
   
   source .venv/bin/activate
   python main.py

5. The script will:
   - Load and preprocess the data
   - Train 7 different machine learning models
   -  perform hyperparameter tuning (this may take 10-30 minutes depending on your computer)
   - Generate learning curves and validation curves
   - create visualizations
   - and also save results and predictions

================================================================================
EXPECTED OUTPUT
==============================================================================

The script will create the following files and directories:

DIRECTORY: figures/
  Contains all generated plots and visualizations:
  - class_distribution.png: Distribution of classes in training set
  - model_comparison.png: Comparison of all models (accuracy and F1 scores)
  - confusion_matrix_best_model.png: Confusion matrix for the best model
  - learning_curves.png: Learning curves for top 3 models
  - validation_curve_dt_maxdepth.png: Decision Tree hyperparameter tuning
  - validation_curve_rf_nestimators.png: Random Forest hyperparameter tuning
  - validation_curve_knn_nneighbors.png: KNN hyperparameter tuning
  - test_predictions_distribution.png: Distribution of predictions on test set

FILES:
  - model_comparison_results.csv: Detailed results for all models including:
    * Model names
    * Best hyperparameters
    * Training and validation accuracy
    * Training and validation F1 scores
    * Precision, recall, and CV scores
  
  - test_predictions.csv: Predictions on the test set with columns:
    * ID: Customer ID
    * Segmentation: Predicted customer segment (A, B, C, or D)

CONSOLE OUTPUT:
  The script prints detailed information during execution including:
  - Data loading and exploration
  - Preprocessing steps
  - Model training progress
  - Hyperparameter tuning results
  - Best model selection
  - Final summary report


================================================================================
INTERPRETING RESULTS
================================================================================

1. MODEL COMPARISON:
   - Check model_comparison_results.csv for detailed metrics
   - Models are sorted by validation F1 score (best first)
   - Look for the model with highest validation F1 and minimal overfitting
     (small gap between training and validation scores)

2. BEST MODEL:
   - The script automatically selects the best model based on validation F1 score
   - Best hyperparameters are displayed in the console and saved in CSV
   - Confusion matrix shows per-class performance

3. LEARNING CURVES:
   - Shows how model performance changes with training set size
   - Converging curves indicate sufficient training data
   - Large gap between training and validation indicates overfitting

4. VALIDATION CURVES:
   - Shows impact of specific hyperparameters on performance
   - Helps identify optimal hyperparameter values
   - Can reveal overfitting (training score increases while validation decreases)

5. TEST PREDICTIONS:
   - Final predictions are saved in test_predictions.csv
   - Can be submitted or evaluated if true labels become available

================================================================================
TROUBLESHOOTING
================================================================================

ISSUE: "ModuleNotFoundError: No module named 'sklearn'"
SOLUTION: Install scikit-learn: pip install scikit-learn

ISSUE: "ModuleNotFoundError: No module named 'pandas'"
SOLUTION: Install pandas: pip install pandas numpy matplotlib seaborn scikit-learn

ISSUE:   Script runs very slowly
 SOLUTION: This is normal. Hyperparameter tuning can take 10-30 minutes.
          The script uses all available CPU cores (n_jobs=-1) for speed.

ISSUE: Memory error
SOLUTION : Close other applications to free up RAM. The script processes
          large datasets and trains multiple models simultaneously.

ISSUE: Figures not displayed
SOLUTION: Figures are automatically saved to the figures/ directory.
          They are not displayed on screen to avoid blocking execution.

================================================================================
FOR REPORT WRITING
================================================================================

All figures in the figures/ directory are publication-ready (300 DPI, high quality).

Suggested structure for your report:

1. INTRODUCTION
   - Problem statement: Customer segmentation
   - Dataset description: Use info from console output

2. METHODS
   - Data preprocessing: Handling missing values, encoding, scaling
   - Models evaluated: 7 different algorithms
   - Hyperparameter tuning: GridSearchCV with 5-fold CV
   - Evaluation metrics: Accuracy, F1 score, precision, recall

3. HYPERPARAMETER TUNING
   - Include validation curve figures
   - Discuss optimal hyperparameters found
   - Explain trade-offs observed

4. LEARNING CURVES
   - Include learning curve figures
   - Discuss model convergence
   - Identify potential overfitting/underfitting

5. RESULTS
   - Include model comparison figure
   - Include confusion matrix of best model
   - Use model_comparison_results.csv for detailed table
   - Report training and validation scores

6. DISCUSSION
   - Compare models: Why did some perform better?
   - Analyze confusion matrix: Which classes are hard to predict?
   - Discuss practical implications

7. CONCLUSION
   - Summary of findings
   - Best model and its performance
   - Future improvements

