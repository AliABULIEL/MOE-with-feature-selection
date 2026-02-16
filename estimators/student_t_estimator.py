import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t, nct
from .base_estimator import BaseEstimator


class StudentTEstimator(BaseEstimator):
    def __init__(self, bins=71, quantile_range=(0.2, 0.8)):
        super().__init__(name="StudentT_NonLinear")
        self.bins = bins
        self.quantile_range = quantile_range

        # Parameters
        self.df = None
        self.loc = None
        self.scale = None
        self.amp = None

    def fit(self, data: np.ndarray):
        data = np.array(data)

        # 1. Bin the Data
        counts, bin_edges = np.histogram(data, bins=self.bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 2. Define "Null Domain"
        q_low, q_high = np.percentile(data, [100 * q for q in self.quantile_range])
        mask = (bin_centers >= q_low) & (bin_centers <= q_high)

        y_train = counts[mask]
        x_train = bin_centers[mask]

        # 3. Define Model
        def t_model_counts(x, df, loc, scale, amp):
            return amp * t.pdf(x, df, loc, scale)

        # Initial Guesses
        p0 = [
            4.0,  # df
            np.median(data),  # loc
            np.std(data),  # scale
            np.max(y_train) * 5,  # Amplitude
        ]

        # Bounds: df>2, scale>0, amp>0
        bounds = ([2, -np.inf, 1e-6, 0], [10, np.inf, np.inf, np.inf])

        try:
            popt, _ = curve_fit(t_model_counts, x_train, y_train, p0=p0, bounds=bounds)
            self.df, self.loc, self.scale, self.amp = popt
        except RuntimeError:
            print(f"{self.name}: Curve fit failed to converge.")
            # Fallback values or raise error
            self.df, self.loc, self.scale, self.amp = (
                4.0,
                np.median(data),
                np.std(data),
                0,
            )

        return self

    def pdf(self, x: np.ndarray):
        if self.df is None:
            raise RuntimeError("Estimator must be fitted before calling pdf().")
        # Note: We ignore 'amp' for PDF/CDF as those must be normalized distributions
        return t.pdf(x, self.df, self.loc, self.scale)

    def cdf(self, x: np.ndarray):
        if self.df is None:
            raise RuntimeError("Estimator must be fitted before calling cdf().")
        return t.cdf(x, self.df, self.loc, self.scale)


class StudentTSkewedEstimator(BaseEstimator):
    def __init__(self, bins=71, quantile_range=(0.05, 0.95)):
        super().__init__(name="StudentT_Skewed_NonLinear")
        self.bins = bins
        self.quantile_range = quantile_range

        # Parameters
        self.df = None
        self.nc = None  # Non-centrality parameter (skewness)
        self.loc = None
        self.scale = None
        self.amp = None

    def fit(self, data: np.ndarray):
        data = np.array(data)

        # 1. Bin the Data
        counts, bin_edges = np.histogram(data, bins=self.bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 2. Define "Null Domain" (Central Mass)
        q_low, q_high = np.percentile(data, [100 * q for q in self.quantile_range])
        mask = (bin_centers >= q_low) & (bin_centers <= q_high)

        y_train = counts[mask]
        x_train = bin_centers[mask]

        # 3. Define Model
        # nct.pdf(x, df, nc, loc, scale)
        def skewed_t_model_counts(x, df, nc, loc, scale, amp):
            return amp * nct.pdf(x, df, nc, loc=loc, scale=scale)

        # Initial Guesses
        # df=4, nc=0 (symmetric start), loc=median, scale=std
        p0 = [
            4.0,  # df
            0.0,  # nc
            np.median(data),  # loc
            np.std(data),  # scale
            np.max(y_train) * 5,  # Amplitude
        ]

        # Bounds:
        # df > 2 (for finite variance)
        # nc: unbounded (-inf to inf)
        # loc: unbounded
        # scale > 0
        # amp > 0
        bounds = (
            [2, -np.inf, -np.inf, 1e-6, 0],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        )

        try:
            popt, _ = curve_fit(
                skewed_t_model_counts, x_train, y_train, p0=p0, bounds=bounds
            )
            self.df, self.nc, self.loc, self.scale, self.amp = popt
        except (RuntimeError, OverflowError):
            print(f"{self.name}: Curve fit failed to converge.")
            # Fallback to standard normal-ish values
            self.df, self.nc, self.loc, self.scale, self.amp = (
                4.0,
                0.0,
                np.median(data),
                np.std(data),
                0,
            )

        return self

    def pdf(self, x: np.ndarray):
        if self.df is None:
            raise RuntimeError("Estimator must be fitted before calling pdf().")
        return nct.pdf(x, self.df, self.nc, loc=self.loc, scale=self.scale)

    def cdf(self, x: np.ndarray):
        if self.df is None:
            raise RuntimeError("Estimator must be fitted before calling cdf().")
        return nct.cdf(x, self.df, self.nc, loc=self.loc, scale=self.scale)
