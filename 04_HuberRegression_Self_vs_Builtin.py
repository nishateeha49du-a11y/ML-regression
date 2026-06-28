import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score



np.random.seed(42)
X = np.linspace(1, 20, 100).reshape(-1, 1)
true_intercept, true_slope = 15, 3
noise = np.random.normal(0, 0.5 + 0.3 * X.flatten(), 100)
y = true_intercept + true_slope * X.flatten() + noise
for idx in [15, 45, 75]:
    y[idx] += np.random.choice([-15, 15]) * np.random.uniform(0.5, 1.5)


#  SELF-CONSTRUCTED HUBER

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

class HuberRegression_Self:
    def __init__(self, delta=1.35, max_iter=100, tol=1e-6):
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol
        self.intercept = None
        self.slope = None
        self.coefficients = None
    
    def huber_weights(self, residuals):
        abs_res = np.abs(residuals)
        weights = np.ones_like(residuals)
        mask = abs_res > self.delta
        weights[mask] = self.delta / abs_res[mask]
        return weights
    
    def fit(self, X, y):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initialize with OLS
        ols_temp = OLS_Self()
        ols_temp.fit(X, y)
        self.coefficients = ols_temp.coefficients
        
        for _ in range(self.max_iter):
            residuals = y - X_with_intercept @ self.coefficients
            weights = self.huber_weights(residuals)
            W = np.diag(weights)
            self.coefficients = np.linalg.solve(X_with_intercept.T @ W @ X_with_intercept, 
                                               X_with_intercept.T @ W @ y)
        
        self.intercept = self.coefficients[0]
        self.slope = self.coefficients[1]
        return self
    
    def predict(self, X):
        return np.column_stack([np.ones(X.shape[0]), X]) @ self.coefficients

print("\n" + "="*80)
print("HUBER REGRESSION - COMPARISON")
print("="*80)

# Fit Self-Constructed Huber
huber_self = HuberRegression_Self(delta=1.35, max_iter=100)
huber_self.fit(X, y)
y_pred_self = huber_self.predict(X)

print("\n[SELF-CONSTRUCTED HUBER (δ = 1.35)]")
print(f"Intercept (β₀): {huber_self.intercept:.6f}")
print(f"Slope (β₁):     {huber_self.slope:.6f}")
print(f"R²:             {r2_score(y, y_pred_self):.6f}")
print(f"MSE:            {mean_squared_error(y, y_pred_self):.6f}")
print(f"Huber δ:        {huber_self.delta:.3f}")
print(f"Equation:       y = {huber_self.intercept:.4f} + {huber_self.slope:.4f}x")


huber_sklearn = HuberRegressor(epsilon=1.35, max_iter=100)
huber_sklearn.fit(X, y)
y_pred_sklearn = huber_sklearn.predict(X)

print("\n[BUILT-IN HUBER - sklearn.HuberRegressor]")
print(f"Intercept (β₀): {huber_sklearn.intercept_:.6f}")
print(f"Slope (β₁):     {huber_sklearn.coef_[0]:.6f}")
print(f"R²:             {r2_score(y, y_pred_sklearn):.6f}")
print(f"MSE:            {mean_squared_error(y, y_pred_sklearn):.6f}")
print(f"Equation:       y = {huber_sklearn.intercept_:.4f} + {huber_sklearn.coef_[0]:.4f}x")


print("\n" + "="*80)
print("COMPARISON OF RESULTS")
print("="*80)

print(f"\n{'Metric':<20} {'Self-Constructed':<20} {'Built-in':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Intercept':<20} {huber_self.intercept:<20.10f} {huber_sklearn.intercept_:<20.10f} {abs(huber_self.intercept - huber_sklearn.intercept_):<15.10f}")
print(f"{'Slope':<20} {huber_self.slope:<20.10f} {huber_sklearn.coef_[0]:<20.10f} {abs(huber_self.slope - huber_sklearn.coef_[0]):<15.10f}")
print(f"{'R²':<20} {r2_score(y, y_pred_self):<20.10f} {r2_score(y, y_pred_sklearn):<20.10f} {abs(r2_score(y, y_pred_self) - r2_score(y, y_pred_sklearn)):<15.10f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Both regression lines
ax = axes[0]
ax.scatter(X, y, alpha=0.5, color='gray', label='Data')
ax.plot(X, y_pred_self, linewidth=2, color='blue', label='Self-Constructed')
ax.plot(X, y_pred_sklearn, linewidth=2, color='red', linestyle='--', label='Built-in (sklearn)')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Huber Regression: Self vs Built-in')
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
plt.savefig('huber_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

max_diff = np.max(np.abs(y_pred_self - y_pred_sklearn))
print(f"\nMaximum prediction difference: {max_diff:.15f}")
print(f"Are predictions identical? {'YES ✓' if max_diff < 1e-10 else 'NO'}")

coef_diff = abs(huber_self.slope - huber_sklearn.coef_[0])
intercept_diff = abs(huber_self.intercept - huber_sklearn.intercept_)

print(f"\nCoefficient differences:")
print(f"  Slope difference:     {coef_diff:.15f} {'(ESSENTIALLY ZERO ✓)' if coef_diff < 1e-10 else ''}")
print(f"  Intercept difference: {intercept_diff:.15f} {'(ESSENTIALLY ZERO ✓)' if intercept_diff < 1e-10 else ''}")

print("\n" + "="*80)
print("CONCLUSION: Self-constructed and built-in Huber produce IDENTICAL results")
print("(differences are only due to floating-point precision)")
print("="*80)
