import numpy as np
from scipy.stats import gaussian_kde
from .base_estimator import BaseEstimator


class KDEEstimator(BaseEstimator):
    def __init__(self, quantile_range=(0.2, 0.8), max_samples=50000):
        super().__init__("gaussian_kde")
        self.kde = None
        self.quantile_range = quantile_range
        self._cdf_grid_x = None
        self._cdf_grid_y = None
        self.max_samples = max_samples

    def fit(self, data):
        if len(data) > self.max_samples:
            rng = np.random.default_rng(42)
            data = rng.choice(data, size=self.max_samples, replace=False)
        self.kde = gaussian_kde(data)
        # Mask data out of the quantile_range
        min_val, max_val = np.min(data), np.max(data)
        margin = (max_val - min_val) * 0.2
        self._cdf_grid_x = np.linspace(min_val - margin, max_val + margin, 10000)
        pdf_vals = self.kde(self._cdf_grid_x)
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals = cdf_vals / cdf_vals[-1]
        self._cdf_grid_y = cdf_vals
        return self

    def pdf(self, x):
        if self.kde is None:
            raise ValueError("Model not fitted yet.")
        return self.kde.evaluate(x)

    def cdf(self, x):
        if self._cdf_grid_x is None or self._cdf_grid_y is None:
            raise ValueError("Model not fitted yet.")
        # Interpolate the CDF values
        return np.interp(x, self._cdf_grid_x, self._cdf_grid_y)
