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

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    c=[]
    for i in range(len(numerator)): 
        if denominator[i]== 0.0 : 
            res = numerator[i]/eps
        else : 
            res =numerator[i]/(denominator[i]+eps)
        c.append(res)
    a = np.array(c)
    return a

# Step 5 - append_column
def append_column(X, col):
    B = X.tolist()
    for i in range(len(B)): 
        B[i].append(col[i])
    a=np.array(B)
    return a

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    X=labels.tolist()
    A=sorted(set(X))
    B=[[0 for _ in range(len(A))]for _ in range(len(X))]
    for i in range(len(X)): 
        pos=A.index(X[i])
        B[i][pos]=1.0
    k=np.array(B)
    return k

# Step 7 - fit_standardizer
def fit_standardizer(X):
    m=np.mean(X,axis=0)
    std=np.std(X,axis=0) 
    for i in range(len(std)) : 
        if std[i]== 0.0 : 
            std[i]=1.0 
    return m,std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    b=[] 
    a=X.tolist()
    for row in a : 
        res= (row- mean)/std 
        b.append(res)
    x=np.array(b)
    return x

# Step 9 - add_bias_column
def add_bias_column(X):
    A=X.tolist() 
    for row in A: 
        row.insert(0,1.0)
    X=np.array(A)
    return X

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    rng = np.random.default_rng(seed)
    A=np.arange(n_samples)
    B=rng.permutation(A)

    return B

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
  n=int(len(indices)*train_ratio)
  train_idx=np.array(indices[:n])
  m=int(len(indices)*val_ratio)
  val_idx=np.array(indices[n:n+m])
  test_idx=np.array(indices[n+m:])
  return train_idx,val_idx,test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):

    X_sub = X[indices]
    y_sub = y[indices]

    return X_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    c=[]
    for i in range(len(X)) :
        s=0
        for j in range(len(theta)): 
            s=X[i][j]*theta[j]+s
        c.append(s)
    
    r=np.array(c)
    return r

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    mae=mean_absolute_error(y_true,y_pred)
    return mae

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    return rmse

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

