"""
Optimized Full Pipeline: Similarity-based Cancer Rate Prediction

This script performs data preprocessing, calculates Z-scores, computes similarity metrics using various methods,
evaluates prediction performance via AUC-ROC, and compares with logistic regression.

Key Features:
- Handles numeric and categorical columns.
- Computes feature weights using T-test (numeric) or Chi-square (categorical).
- Supports multiple similarity methods: sum_abs_z, sum_sq_z, euclidean, manhattan, weighted_sum_abs_z, weighted_sum_sq_z.
- Visualizes ROC curves for each method.
- Requires a CSV file with columns: 'primary_indicator' (target), and result_columns defined below.

Dependencies:
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- tabulate
- tqdm

To install dependencies:
pip install pandas numpy scikit-learn scipy matplotlib tabulate tqdm

Usage:
python cancer_prediction_pipeline.py path/to/your/data.csv

Author: Alex Lin
Date: November 26, 2025
"""

import argparse
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Define result columns (features used for analysis)
result_columns = [
    'age',
    'Lymphocyte, absolute_Result',
    'Carcinoembryonic Ag_Result',
    'Cytology Category'
]

# Define categorical columns
categorical_cols = ['Cytology Category']

def calculate_stats(df):
    """Calculate mean, median, std for result columns."""
    column_stats = {}
    for col in result_columns:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()
        column_stats[col] = {'mean': mean, 'median': median, 'std': std}
    
    mean_std_table = [[col, stats['mean'], stats['median'], stats['std']]
                      for col, stats in column_stats.items()]
    print("\nMean, Median, and Standard Deviation for Result Columns:")
    print(tabulate(mean_std_table, headers=["Column", "Mean", "Median", "Std"],
                   tablefmt="grid", floatfmt=".2f"))
    
    return column_stats

def calculate_z_scores(df, column_stats):
    """Calculate absolute Z-scores for result columns."""
    for col in result_columns:
        if col not in column_stats:
            continue
        mean = column_stats[col]['mean']
        std = column_stats[col]['std']
        if std > 0:
            df[f'{col}_Z_Score'] = np.abs((df[col] - mean) / std)
        else:
            df[f'{col}_Z_Score'] = 0.0
    return df

def calculate_sums(df):
    """Calculate sum of abs Z-scores and sum of squared abs Z-scores."""
    z_score_cols = [f'{col}_Z_Score' for col in result_columns if f'{col}_Z_Score' in df.columns]
    df['Sum_Abs_Z'] = df[z_score_cols].sum(axis=1)
    df['Sum_Sq_Abs_Z'] = (df[z_score_cols] ** 2).sum(axis=1)
    return df

def find_similar_rows(analysis_df, input_sum, sum_col_name, n=1000):
    """Find n similar rows based on absolute difference in a sum column."""
    diff_col_name = 'Difference'
    analysis_df[diff_col_name] = np.abs(analysis_df[sum_col_name] - input_sum)
    similar = analysis_df.nsmallest(n, diff_col_name)
    if len(similar) < n:
        print(f"Warning: Only {len(similar)} similar rows found (requested {n}).")
    return similar

def calculate_prediction(similar_rows):
    """Calculate prediction as the cancer rate in similar rows."""
    if similar_rows.empty:
        return 0.0  # Default if no similar rows
    cancer_rate_similar = (similar_rows['primary_indicator'] == 1).mean()
    return cancer_rate_similar

def process_row(args):
    """Process a single row for similarity calculation and prediction."""
    idx, df, n_similar, method = args
    z_score_cols = [f'{col}_Z_Score' for col in result_columns if f'{col}_Z_Score' in df.columns]
    input_row = df.iloc[idx]
    analysis_df = df.drop(index=idx).copy()
    if method == 'sum_abs_z':
        sum_col_name = 'Sum_Abs_Z'
        input_sum = input_row[sum_col_name]
        similar_rows = find_similar_rows(analysis_df, input_sum, sum_col_name, n=n_similar)
    elif method == 'sum_sq_z':
        sum_col_name = 'Sum_Sq_Abs_Z'
        input_sum = input_row[sum_col_name]
        similar_rows = find_similar_rows(analysis_df, input_sum, sum_col_name, n=n_similar)
    elif method == 'weighted_sum_abs_z':
        sum_col_name = 'Weighted_Sum_Abs_Z'
        input_sum = input_row[sum_col_name]
        similar_rows = find_similar_rows(analysis_df, input_sum, sum_col_name, n=n_similar)
    elif method == 'weighted_sum_sq_z':
        sum_col_name = 'Weighted_Sum_Sq_Abs_Z'
        input_sum = input_row[sum_col_name]
        similar_rows = find_similar_rows(analysis_df, input_sum, sum_col_name, n=n_similar)
    elif method in ['euclidean', 'manhattan']:
        # Force numeric dtype to avoid object array issues
        input_vector = input_row[z_score_cols].values.astype(np.float64)
        vectors = analysis_df[z_score_cols].values.astype(np.float64)
        if method == 'euclidean':
            distances = np.sqrt(np.sum((vectors - input_vector) ** 2, axis=1))
        elif method == 'manhattan':
            distances = np.sum(np.abs(vectors - input_vector), axis=1)
        analysis_df['Distance'] = distances
        similar_rows = analysis_df.nsmallest(n_similar, 'Distance')
        if len(similar_rows) < n_similar:
            print(f"Warning: Only {len(similar_rows)} similar rows found (requested {n_similar}).")
    else:
        raise ValueError(f"Unknown method: {method}")
    # Calculate prediction
    prediction = calculate_prediction(similar_rows)
    # Return actual and predicted values
    actual = input_row['primary_indicator']
    return actual, prediction

def calculate_ttest_weights(df, target_col='primary_indicator', weight_method='log', max_weight=1000.0, normalize=True, epsilon=1e-10):
    """
    Calculate weights for each feature using statistical tests.
    - For numeric features: T-test between groups (target=0 vs target=1).
    - For categorical features: Chi-square test.
    - weight_method: 'reciprocal' (1 / p-value) or 'log' (-log(p-value + epsilon)).
    - max_weight: Cap for extreme values.
    - normalize: Scale weights to sum to 1.
    - epsilon: Avoid log(0) or division by zero.
    """
    weights = {}
    group0 = df[df[target_col] == 0]
    group1 = df[df[target_col] == 1]
    
    for col in result_columns:
        if col not in df.columns:
            continue
        
        if col in categorical_cols:
            # Chi-square for categorical
            contingency_table = pd.crosstab(df[col], df[target_col])
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2 or contingency_table.values.sum() == 0:
                print(f"Skipping chi-square for {col}: Insufficient categories or classes. Setting p_value to 1.0.")
                p_value = 1.0
            else:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                if np.isnan(p_value):
                    print(f"NaN p-value for chi-square on {col}. Setting p_value to 1.0.")
                    p_value = 1.0
        else:
            # T-test for numeric
            if df[col].dtype not in [np.float64, np.int64]:
                print(f"Skipping non-numeric column for T-test: {col}. Setting p_value to 1.0.")
                p_value = 1.0
            else:
                g0_vals = group0[col].dropna()
                g1_vals = group1[col].dropna()
                if len(g0_vals) < 2 or len(g1_vals) < 2:
                    print(f"Insufficient samples for T-test on {col}. Setting p_value to 1.0.")
                    p_value = 1.0
                else:
                    stat, p_value = ttest_ind(g0_vals, g1_vals, equal_var=False)
                    if np.isnan(p_value):
                        print(f"NaN p-value for T-test on {col}. Setting p_value to 1.0.")
                        p_value = 1.0
        
        # Set weight based on p_value
        p_value = max(p_value, epsilon)  # Avoid zero
        if weight_method == 'reciprocal':
            weight = 1 / p_value
        elif weight_method == 'log':
            weight = -np.log(p_value)
        else:
            raise ValueError(f"Unknown weight_method: {weight_method}")
        
        # Cap the weight
        weight = min(weight, max_weight)
        weights[col] = weight
    
    # Normalize if requested
    if normalize:
        total = sum(weights.values())
        if total > 0:
            for col in weights:
                weights[col] /= total
    
    # Print weights table
    weights_table = [[col, weight] for col, weight in weights.items()]
    print("\nFeature Weights (using method '{}')".format(weight_method))
    print(tabulate(weights_table, headers=["Column", "Weight"], tablefmt="grid", floatfmt=".4f"))
    
    return weights

def calculate_weighted_sums(df, weights):
    """Calculate weighted sum of (abs Z-scores * weight) and weighted sum of squared."""
    df['Weighted_Sum_Abs_Z'] = 0.0
    df['Weighted_Sum_Sq_Abs_Z'] = 0.0
    for col in result_columns:
        z_col = f'{col}_Z_Score'
        if z_col in df.columns and col in weights:
            df['Weighted_Sum_Abs_Z'] += df[z_col] * weights[col]
            df['Weighted_Sum_Sq_Abs_Z'] += (df[z_col] ** 2) * weights[col]
    return df

def run_logistic_regression(df, test_size=0.2, random_state=42):
    """Run logistic regression on the features and compute AUC-ROC."""
    X = df[result_columns].copy()
    y = df['primary_indicator']
    
    # Handle NaNs with imputation (mean for numeric, most frequent for categorical)
    imputer = SimpleImputer(strategy='mean')
    X_numeric = imputer.fit_transform(X.select_dtypes(include=[np.number]))
    X_numeric = pd.DataFrame(X_numeric, columns=X.select_dtypes(include=[np.number]).columns, index=X.index)
    
    # For categorical (if any), use most frequent
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_cat = imputer_cat.fit_transform(X[cat_cols])
        X_cat = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)
        X = pd.concat([X_numeric, X_cat], axis=1)
    else:
        X = X_numeric
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC-ROC
    if len(np.unique(y_test)) < 2:
        print("Warning: Only one class in test set. Setting AUC-ROC to 0.5.")
        auc_score = 0.5
    else:
        auc_score = roc_auc_score(y_test, y_pred_prob)
    
    print(f"\n AUC-ROC Score for Logistic Regression: {auc_score:.4f}")
    
    # Visualization
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color='green')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
    plt.title("ROC Curve for Logistic Regression")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    
    return auc_score

def main(df):
    # Ensure at least 2 non-NaN values in result_columns, regardless of primary_indicator
    df = df[df[result_columns].notna().sum(axis=1) >= 2]
    df = df.dropna(subset=['primary_indicator'])  # Drop rows where primary_indicator is NaN
    
    # Handle categorical encoding
    le = LabelEncoder()
    for col in result_columns:
        if col in categorical_cols or df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Reset index for consistent iloc
    df = df.reset_index(drop=True)
    print(f"\nDataFrame shape after preprocessing: {df.shape}")
    
    # Calculate statistics, Z-scores, and sums
    column_stats = calculate_stats(df)
    df = calculate_z_scores(df, column_stats)
    
    # Fill NaN in Z-scores with 0 (consistent with sum behavior)
    z_score_cols = [f'{col}_Z_Score' for col in result_columns if f'{col}_Z_Score' in df.columns]
    df[z_score_cols] = df[z_score_cols].fillna(0)
    df = calculate_sums(df)
    
    # Calculate T-test/Chi-square based weights and weighted sums
    weights = calculate_ttest_weights(df, weight_method='log', max_weight=1000.0, normalize=True)
    df = calculate_weighted_sums(df, weights)
    
    # Select random rows for similarity analysis with stratification
    num_records = 1000  # Number of random rows to analyze
    n_similar = 1000  # Number of similar rows to consider (ensure df is large enough)
    random_seed = 42
    random.seed(random_seed)
    
    # Stratified sampling to ensure both classes are represented
    class0_idx = df.index[df['primary_indicator'] == 0].tolist()
    class1_idx = df.index[df['primary_indicator'] == 1].tolist()
    prop1 = len(class1_idx) / len(df) if len(df) > 0 else 0
    num1 = min(round(num_records * prop1), len(class1_idx))
    num0 = min(num_records - num1, len(class0_idx))
    
    # Ensure at least one from each class if possible
    if num1 == 0 and len(class1_idx) > 0:
        num1 = 1
        num0 = max(0, num_records - 1)
    if num0 == 0 and len(class0_idx) > 0:
        num0 = 1
        num1 = max(0, num_records - 1)
    
    sampled0 = random.sample(class0_idx, min(num0, len(class0_idx)))
    sampled1 = random.sample(class1_idx, min(num1, len(class1_idx)))
    random_indices = sampled0 + sampled1
    random.shuffle(random_indices)
    
    # If only one class, warn
    if len(class0_idx) == 0 or len(class1_idx) == 0:
        print("Warning: Dataset contains only one class. AUC-ROC may not be computable.")
    
    # Define methods to evaluate
    methods = ['sum_abs_z', 'sum_sq_z', 'euclidean', 'manhattan', 'weighted_sum_abs_z', 'weighted_sum_sq_z']
    
    # Collect AUC scores for summary
    auc_scores = {}
    for method in methods:
        print(f"\n--- Evaluating method: {method} ---")
        # Process rows serially
        tqdm_desc = f"Processing {len(random_indices)} input rows for {method}"
        args = [(idx, df, n_similar, method) for idx in random_indices]
        results = []
        for arg in tqdm(args, desc=tqdm_desc):
            results.append(process_row(arg))
        
        # Extract actual and predicted values
        actuals, predictions = zip(*results)
        actuals = np.array(actuals)
        predictions = np.array(predictions)
        
        # Calculate AUC-ROC with check for single class
        if len(np.unique(actuals)) < 2:
            print("Warning: Only one class present in actuals. Setting AUC-ROC to 0.5.")
            auc_score = 0.5
            print("Skipping ROC curve visualization due to single class.")
            auc_scores[method] = auc_score
            continue
        else:
            auc_score = roc_auc_score(actuals, predictions)
            print(f"\n AUC-ROC Score for {method}: {auc_score:.4f}")
            auc_scores[method] = auc_score
            
            # Visualization
            fpr, tpr, thresholds = roc_curve(actuals, predictions)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color='blue')
            plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
            plt.title(f"ROC Curve for {method}")
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.show()
    
    # Run logistic regression for comparison and add to auc_scores
    logreg_auc = run_logistic_regression(df)
    auc_scores['logistic_regression'] = logreg_auc
    
    # Summarize performance in a DataFrame
    summary_df = pd.DataFrame(list(auc_scores.items()), columns=['Method', 'AUC_ROC'])
    summary_df = summary_df.sort_values(by='AUC_ROC', ascending=False).reset_index(drop=True)
    print("\nSummary of Model Performances:")
    print(tabulate(summary_df, headers='keys', tablefmt='grid', floatfmt=".4f"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the similarity-based cancer rate prediction pipeline.")
    parser.add_argument("data_file", type=str, help="Path to the CSV data file.")
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_file)
    main(df)