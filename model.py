"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    T = np.transpose(X)
    c=[]
    for j in range(len(T)) :
        a=0.0 
        for i in range(len(T[0])) :
            if np.isnan(T[j][i])  :
                a=a 
            else : 
                a=a+T[j][i]
        c.append(a)
    k=[]
    for j in range(len(T)) : 
        s=0
        for i in range(len(T[0])) :
            if  np.isnan(T[j][i]) == False : 
                s+=1
        k.append(s)
    for i in range(len(k)) : 
        if k[i] == 0 : 
            T[i]=0.0
    for j in range(len(T)) : 

        for i in range(len(T[0])) : 
            if np.isnan(T[j][i]) and k[j]>0 :
                T[j][i] = c[j]/ k[j]
        
    X1=np.transpose(T)
       
    return X1

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr=q3-q1
    lower = q1 -k*iqr 
    upper = q3 + k*iqr
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    X1=X.copy()
    for i in range(len(X)) : 
        for j in range(len(X[0])): 
            if X1[i][j] < lower[j] :
                X1[i][j]= lower[j] 
            if X1[i][j] > upper[j]:
                X1[i][j]= upper[j] 
    return (X1)

# Step 4 - make_ratio_feature (not yet solved)
# TODO: implement

# Step 5 - append_column (not yet solved)
# TODO: implement

# Step 6 - one_hot_encode (not yet solved)
# TODO: implement

# Step 7 - fit_standardizer (not yet solved)
# TODO: implement

# Step 8 - apply_standardizer (not yet solved)
# TODO: implement

# Step 9 - add_bias_column (not yet solved)
# TODO: implement

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

