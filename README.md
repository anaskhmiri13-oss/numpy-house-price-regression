# NumPy House Price Regression

Build a complete end-to-end house-price regressor in pure NumPy: clean tabular features, engineer simple derived and categorical encodings, split reproducibly, fit OLS via the normal equation, and report MAE, RMSE, R-squared, plus residual stats on held-out homes.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** impute_nan_with_mean
- [x] **2.** compute_iqr_bounds
- [x] **3.** clip_columns
- [x] **4.** make_ratio_feature
- [x] **5.** append_column
- [x] **6.** one_hot_encode
- [x] **7.** fit_standardizer
- [x] **8.** apply_standardizer
- [x] **9.** add_bias_column
- [x] **10.** make_shuffled_indices
- [x] **11.** partition_indices
- [x] **12.** subset_xy
- [x] **13.** ols_fit
- [x] **14.** ols_predict
- [x] **15.** mean_absolute_error
- [x] **16.** root_mean_squared_error
- [x] **17.** r_squared
- [x] **18.** residual_summary
- [ ] **19.** prepare_cleaned_features
- [ ] **20.** assemble_feature_matrix
- [ ] **21.** make_train_val_test
- [ ] **22.** standardize_and_add_bias
- [ ] **23.** evaluate_predictions
- [ ] **24.** house_price_pipeline

---

Built on Deep-ML.
