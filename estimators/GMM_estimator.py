import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from .base_estimator import BaseEstimator


class GMMNullEstimator(BaseEstimator):
    def __init__(self, n_components=2, random_state=42, quantile_range=(0.2, 0.8)):
        super().__init__(f"gmm_null_{n_components}_components")
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None
        self.quantile_range = quantile_range
        self.null_idx = None
        
    def fit(self, z_scores):
        z_scores = np.array(z_scores)
        q_low, q_high = np.percentile(z_scores, [100 * q for q in self.quantile_range])
        self.null_idx = (z_scores >= q_low) & (z_scores <= q_high)
        z_scores = z_scores[self.null_idx]
        z_scores_reshaped = z_scores.reshape(-1, 1)
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=self.random_state)
        self.gmm.fit(z_scores_reshaped)
    
    def _get_null_params(self):
        if self.gmm is None:
            raise ValueError("Model not fitted yet.")
        means = self.gmm.means_.flatten() # type: ignore
        variances = self.gmm.covariances_.flatten() # type: ignore
        weights = self.gmm.weights_.flatten() # type: ignore
        stds = np.sqrt(variances)
        
        return means, stds, weights
    def pdf(self, x):
        if self.gmm is None:
            raise ValueError("Model not fitted yet.")
        x_reshaped = x.reshape(-1, 1)
        return np.exp(self.gmm.score_samples(x_reshaped))

    def cdf(self, x):
        if self.gmm is None:
            raise ValueError("Model not fitted yet.")
        means, stds, weights = self._get_null_params()
        cdf_val = np.zeros_like(x)
        for i in range(self.n_components):
            cdf_val += weights[i] * norm.cdf(x, loc=means[i], scale=stds[i])
        return cdf_val