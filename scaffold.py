"""
NumPy House Price Regression scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Demo: end-to-end NumPy house-price OLS regression pipeline."""
import numpy as np


def main():
    np.random.seed(0)
    n = 200
    # Synthetic tabular features: [rooms, households, age, income]
    rooms = np.random.uniform(2.0, 8.0, size=n)
    households = np.random.uniform(1.0, 5.0, size=n)
    age = np.random.uniform(5.0, 50.0, size=n)
    income = np.random.uniform(1.0, 10.0, size=n)
    X = np.column_stack([rooms, households, age, income])
    # Inject a few NaNs and outliers
    X[5, 0] = np.nan
    X[12, 3] = np.nan
    X[20, 2] = 200.0
    # Categorical district labels
    cat_labels = np.random.choice(["A", "B", "C"], size=n)
    # Target: noisy linear function of rooms/households and income
    y = 50.0 + 30.0 * (rooms / (households + 1e-8)) + 15.0 * income
    y = y + np.random.normal(0.0, 5.0, size=n)
    y[20] = y[20] + 100.0

    result = house_price_pipeline(
        X,
        y,
        ratio_num_idx=0,
        ratio_den_idx=1,
        cat_labels=cat_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        iqr_k=1.5,
    )

    # house_price_pipeline returns a DICT; unpacking it would yield its keys.
    metrics = result["test_metrics"]
    y_test = result["y_test"]
    y_pred = result["y_test_pred"]

    print("Test metrics:")
    print("  MAE :", round(float(metrics["mae"]), 4))
    print("  RMSE:", round(float(metrics["rmse"]), 4))
    print("  R^2 :", round(float(metrics["r2"]), 4))
    print("Residual summary:", metrics["residual_summary"])
    print("y_test[:5]:", np.round(y_test[:5], 3))
    print("y_pred[:5]:", np.round(y_pred[:5], 3))


if __name__ == "__main__":
    main()
