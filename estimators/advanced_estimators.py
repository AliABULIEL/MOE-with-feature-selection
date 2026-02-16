import numpy as np
from scipy.stats import johnsonsu, gennorm, nct
from scipy.optimize import minimize
from .base_estimator import BaseEstimator

class AdvancedEstimator(BaseEstimator):
    def __init__(self, dist_type="johnsonsu", quantile_cutoff=0.90):
        # אתחול שם המודל
        super().__init__(f"Advanced_{dist_type}")
        self.dist_type = dist_type
        self.quantile_cutoff = quantile_cutoff
        self.params = None
        
        # בחירת אובייקט ההתפלגות של Scipy
        if dist_type == "johnsonsu":
            self.dist = johnsonsu
        elif dist_type == "gennorm": # Generalized Error Distribution (GED)
            self.dist = gennorm
        elif dist_type == "nct": # Non-central t (Skewed t)
            self.dist = nct
        else:
            raise ValueError(f"Unknown dist: {dist_type}")

    def _get_initial_params(self, data):
        """
        מחזירה ניחוש התחלתי לפרמטרים בהתאם לסוג ההתפלגות.
        מחליף את ה-lambda הבעייתית.
        """
        median = np.median(data)
        std = np.std(data)
        
        if self.dist_type == "johnsonsu":
            # Johnson SU params: a, b, loc, scale
            # a=0 (no skew), b=1 (normal kurtosis)
            return [0.0, 1.0, median, std]
        
        elif self.dist_type == "gennorm":
            # GenNorm params: beta, loc, scale
            # beta=2 (Normal), beta=1 (Laplace)
            return [1.5, median, std]
        
        elif self.dist_type == "nct":
            # Non-central t params: df, nc, loc, scale
            return [5.0, 0.0, median, std]
            
        return [median, std] # Fallback

    def fit(self, data):
        # המרה ל-numpy array למניעת בעיות
        data = np.array(data)
        
        # 1. Censored MLE Logic
        # חלוקת הדאטה:
        # observed: הנתונים שמתחת לסף (הרוב) - נשתמש בערך המדויק שלהם
        # censored: הנתונים שמעל הסף (הזנב) - נתייחס אליהם רק כ"גדולים מ-X"
        cutoff_val = np.quantile(data, self.quantile_cutoff)
        
        observed = data[data < cutoff_val]
        # כמות הנתונים המצונזרים
        n_censored = len(data) - len(observed)
        
        # פונקציית המטרה (Negative Log Likelihood)
        def neg_log_likelihood(params):
            # A. Log-Likelihood של החלק הנצפה (הצפיפות הרגילה)
            pdf_vals = self.dist.pdf(observed, *params)
            # הגנה מתמטית מפני log(0)
            pdf_vals = np.maximum(pdf_vals, 1e-300)
            ll_observed = np.sum(np.log(pdf_vals))
            
            # B. Log-Likelihood של החלק המצונזר
            # אנו משתמשים ב-Survival Function (SF = 1 - CDF)
            # ההסתברות להיות גדול מה-cutoff
            sf_at_cutoff = self.dist.sf(cutoff_val, *params)
            sf_at_cutoff = np.maximum(sf_at_cutoff, 1e-300)
            
            # עבור כל נקודה מצונזרת, אנו מוסיפים את הלוג של ההסתברות להיות בזנב
            ll_censored = n_censored * np.log(sf_at_cutoff)
            
            # אנו רוצים למקסם את ה-LL, ולכן ממזערים את המינוס שלו
            return -(ll_observed + ll_censored)

        try:
            # קבלת פרמטרים התחלתיים
            p0 = self._get_initial_params(observed)
            
            # אופטימיזציה
            # Nelder-Mead הוא אלגוריתם רובוסטי שלא דורש נגזרות
            res = minimize(neg_log_likelihood, p0, method='Nelder-Mead')
            
            self.params = res.x
            
        except Exception as e:
            print(f"Custom fitting failed for {self.name}: {e}. Reverting to scipy default fit.")
            # Fallback: אם האופטימיזציה שלנו נכשלה, נשתמש ב-fit הרגיל של scipy
            # (הוא פחות טוב לזנבות אבל עדיף מכלום)
            try:
                self.params = self.dist.fit(data)
            except:
                print(f"Scipy fit also failed for {self.name}")
                self.params = None
            
        return self

    def pdf(self, x):
        if self.params is None:
            # במקרה שהמודל לא התאמן, נחזיר אפסים או נזרוק שגיאה
            return np.zeros_like(x)
        return self.dist.pdf(x, *self.params)

    def cdf(self, x):
        if self.params is None:
            return np.zeros_like(x)
        return self.dist.cdf(x, *self.params)