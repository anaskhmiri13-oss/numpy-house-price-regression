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
- [ ] **7.** fit_standardizer
- [ ] **8.** apply_standardizer
- [ ] **9.** add_bias_column
- [ ] **10.** make_shuffled_indices
- [ ] **11.** partition_indices
- [ ] **12.** subset_xy
- [ ] **13.** ols_fit
- [ ] **14.** ols_predict
- [ ] **15.** mean_absolute_error
- [ ] **16.** root_mean_squared_error
- [ ] **17.** r_squared
- [ ] **18.** residual_summary
- [ ] **19.** prepare_cleaned_features
- [ ] **20.** assemble_feature_matrix
- [ ] **21.** make_train_val_test
- [ ] **22.** standardize_and_add_bias
- [ ] **23.** evaluate_predictions
- [ ] **24.** house_price_pipeline

---

Built on Deep-ML.
