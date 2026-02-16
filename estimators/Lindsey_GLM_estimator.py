import numpy as np
import statsmodels.api as sm
from scipy.integrate import trapezoid, cumulative_trapezoid
from .base_estimator import BaseEstimator


class LindseyGLMEstimator(BaseEstimator):
    def __init__(self, distribution_type="laplace", bins=71, quantile_range=(0.2, 0.8)):
        """
        Estimator using Lindsey's Method (Poisson GLM on histogram counts).

        Args:
            distribution_type (str): 'laplace' or 'gaussian'.
            bins (int): Number of histogram bins.
            quantile_range (tuple): Range of data to use for fitting (e.g., (0.2, 0.8)).
        """
        super().__init__(name=f"LindseyGLM_{distribution_type}")
        self.dist_type = distribution_type.lower()
        self.bins = bins
        self.quantile_range = quantile_range

        # Fit results
        self.glm_result = None
        self.mu = None
        self.n_total = 0
        self.bin_width = 0

    def _make_exog(self, x):
        """Create design matrix based on distribution type."""
        if self.dist_type == "laplace":
            # Linear in log-space with absolute value
            return np.column_stack([np.ones_like(x), np.abs(x - self.mu)])
        elif self.dist_type == "gaussian":
            # Quadratic in log-space
            return np.column_stack([np.ones_like(x), (x - self.mu) ** 2])
        else:
            raise ValueError(f"Unsupported distribution type: {self.dist_type}")

    def fit(self, data: np.ndarray):
        data = np.array(data)
        self.n_total = len(data)
        self.mu = np.median(data)  # Center assumed to be median

        # 1. Bin the Data
        counts, bin_edges = np.histogram(data, bins=self.bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.bin_width = bin_edges[1] - bin_edges[0]

        # 2. Define "Null Domain" (Central Mass)
        q_low, q_high = np.percentile(data, [100 * q for q in self.quantile_range])
        mask = (bin_centers >= q_low) & (bin_centers <= q_high)

        y_train = counts[mask]
        x_train = bin_centers[mask]

        # 3. Fit Poisson GLM
        X_train = self._make_exog(x_train)
        glm_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
        self.glm_result = glm_model.fit()

        return self

    def pdf(self, x: np.ndarray):
        if self.glm_result is None:
            raise RuntimeError("Estimator must be fitted before calling pdf().")

        x = np.array(x)
        # Predict counts and convert to density
        predicted_counts = self.glm_result.predict(self._make_exog(x))
        return predicted_counts / (self.n_total * self.bin_width)

    def cdf(self, x: np.ndarray):
        """
        Computes CDF via numerical integration of the fitted PDF.
        """
        if self.glm_result is None or self.mu is None:
            raise RuntimeError("Estimator must be fitted before calling cdf().")

        x_input = np.array(x)

        # Create a grid covering the data range for integration
        # We need a wide enough grid to capture the tails for accurate CDF
        grid_min = min(np.min(x_input), self.mu - 10 * np.std(x_input))
        grid_max = max(np.max(x_input), self.mu + 10 * np.std(x_input))
        grid = np.linspace(grid_min, grid_max, 10000)

        pdf_vals = self.pdf(grid)

        # Normalize strictly to ensure total area is 1.0 (GLM is approximate)
        total_area = trapezoid(pdf_vals, grid)
        pdf_norm = pdf_vals / total_area

        # Compute CDF on grid
        cdf_grid = cumulative_trapezoid(pdf_norm, grid, initial=0)
        cdf_grid /= cdf_grid[-1]  # Ensure it ends exactly at 1.0

        # Interpolate results for input x
        return np.interp(x_input, grid, cdf_grid)
