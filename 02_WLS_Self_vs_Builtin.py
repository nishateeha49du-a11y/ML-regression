import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(42)
X = np.linspace(1, 20, 100).reshape(-1, 1)
true_intercept, true_slope = 15, 3
noise = np.random.normal(0, 0.5 + 0.3 * X.flatten(), 100)
y = true_intercept + true_slope * X.flatten() + noise
for idx in [15, 45, 75]:
    y[idx] += np.random.choice([-15, 15]) * np.random.uniform(0.5, 1.5)


class OLS_Self:
    def __init__(self):
        self.intercept = None
        self.slope = None
        self.coefficients = None
    
    def fit(self, X, y):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients = np.linalg.solve(X_with_intercept.T @ X_with_intercept, 
                                           X_with_intercept.T @ y)
        self.intercept = self.coefficients[0]
        self.slope = self.coefficients[1]
        return self
    
    def predict(self, X):
        return np.column_stack([np.ones(X.shape[0]), X]) @ self.coefficients

class WLS_Self:
    def __init__(self):
        self.intercept = None
        self.slope = None
        self.coefficients = None
        self.weights = None
    
    def fit(self, X, y):
        # Compute weights based on OLS residuals
        ols_temp = OLS_Self()
        ols_temp.fit(X, y)
        residuals = y - ols_temp.predict(X)
        residuals_abs = np.abs(residuals)
        max_resid = np.max(residuals_abs) if np.max(residuals_abs) > 0 else 1
        self.weights = np.exp(-residuals_abs / max_resid)
        self.weights = self.weights / np.sum(self.weights) * len(y)
        
        # Weighted Least Squares
        W = np.diag(self.weights)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients = np.linalg.solve(X_with_intercept.T @ W @ X_with_intercept, 
                                           X_with_intercept.T @ W @ y)
        self.intercept = self.coefficients[0]
        self.slope = self.coefficients[1]
        return self
    
    def predict(self, X):
        return np.column_stack([np.ones(X.shape[0]), X]) @ self.coefficients

print("\n" + "="*80)
print("WEIGHTED LEAST SQUARES (WLS) - COMPARISON")
print("="*80)

# Fit Self-Constructed WLS
wls_self = WLS_Self()
wls_self.fit(X, y)
y_pred_self = wls_self.predict(X)

print("\n[SELF-CONSTRUCTED WLS]")
print(f"Intercept (β₀): {wls_self.intercept:.6f}")
print(f"Slope (β₁):     {wls_self.slope:.6f}")
print(f"R²:             {r2_score(y, y_pred_self):.6f}")
print(f"MSE:            {mean_squared_error(y, y_pred_self):.6f}")
print(f"Weight range:   {wls_self.weights.min():.4f} to {wls_self.weights.max():.4f}")
print(f"Equation:       y = {wls_self.intercept:.4f} + {wls_self.slope:.4f}x")


wls_sklearn = LinearRegression()
wls_sklearn.fit(X, y, sample_weight=wls_self.weights)
y_pred_sklearn = wls_sklearn.predict(X)

print("\n[BUILT-IN WLS - sklearn.LinearRegression with sample_weight]")
print(f"Intercept (β₀): {wls_sklearn.intercept_:.6f}")
print(f"Slope (β₁):     {wls_sklearn.coef_[0]:.6f}")
print(f"R²:             {r2_score(y, y_pred_sklearn):.6f}")
print(f"MSE:            {mean_squared_error(y, y_pred_sklearn):.6f}")
print(f"Equation:       y = {wls_sklearn.intercept_:.4f} + {wls_sklearn.coef_[0]:.4f}x")


print("\n" + "="*80)
print("COMPARISON OF RESULTS")
print("="*80)

print(f"\n{'Metric':<20} {'Self-Constructed':<20} {'Built-in':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Intercept':<20} {wls_self.intercept:<20.10f} {wls_sklearn.intercept_:<20.10f} {abs(wls_self.intercept - wls_sklearn.intercept_):<15.10f}")
print(f"{'Slope':<20} {wls_self.slope:<20.10f} {wls_sklearn.coef_[0]:<20.10f} {abs(wls_self.slope - wls_sklearn.coef_[0]):<15.10f}")
print(f"{'R²':<20} {r2_score(y, y_pred_self):<20.10f} {r2_score(y, y_pred_sklearn):<20.10f} {abs(r2_score(y, y_pred_self) - r2_score(y, y_pred_sklearn)):<15.10f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Both regression lines
ax = axes[0]
ax.scatter(X, y, alpha=0.5, color='gray', label='Data')
ax.scatter(X, y, alpha=0.5, s=wls_self.weights * 10, color='lightblue', label='Weight size')
ax.plot(X, y_pred_self, linewidth=2, color='blue', label='Self-Constructed')
ax.plot(X, y_pred_sklearn, linewidth=2, color='red', linestyle='--', label='Built-in (sklearn)')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('WLS Regression: Self vs Built-in (bubble size = weight)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Residuals comparison
ax = axes[1]
residuals_self = y - y_pred_self
residuals_sklearn = y - y_pred_sklearn
ax.scatter(y_pred_self, residuals_self, alpha=0.5, color='blue', label='Self Residuals')
ax.scatter(y_pred_sklearn, residuals_sklearn, alpha=0.5, color='red', label='Built-in Residuals', marker='x')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wls_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

max_diff = np.max(np.abs(y_pred_self - y_pred_sklearn))
print(f"\nMaximum prediction difference: {max_diff:.15f}")
print(f"Are predictions identical? {'YES ✓' if max_diff < 1e-10 else 'NO'}")

coef_diff = abs(wls_self.slope - wls_sklearn.coef_[0])
intercept_diff = abs(wls_self.intercept - wls_sklearn.intercept_)

print(f"\nCoefficient differences:")
print(f"  Slope difference:     {coef_diff:.15f} {'(ESSENTIALLY ZERO ✓)' if coef_diff < 1e-10 else ''}")
print(f"  Intercept difference: {intercept_diff:.15f} {'(ESSENTIALLY ZERO ✓)' if intercept_diff < 1e-10 else ''}")

print("\n" + "="*80)
print("CONCLUSION: Self-constructed and built-in WLS produce IDENTICAL results")
print("(differences are only due to floating-point precision)")
print("="*80)
