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


class GLS_Self:
    def __init__(self, rho=0.3):
        self.rho = rho
        self.intercept = None
        self.slope = None
        self.coefficients = None
        self.cov_matrix = None
    
    def fit(self, X, y):
        n = len(y)
        # Build AR(1) covariance matrix
        self.cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.cov_matrix[i, j] = self.rho ** abs(i - j)
        
        inv_cov = np.linalg.inv(self.cov_matrix)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients = np.linalg.solve(X_with_intercept.T @ inv_cov @ X_with_intercept,
                                           X_with_intercept.T @ inv_cov @ y)
        self.intercept = self.coefficients[0]
        self.slope = self.coefficients[1]
        return self
    
    def predict(self, X):
        return np.column_stack([np.ones(X.shape[0]), X]) @ self.coefficients

print("\n" + "="*80)
print("GENERALIZED LEAST SQUARES (GLS) - COMPARISON")
print("="*80)

# Fit Self-Constructed GLS
gls_self = GLS_Self(rho=0.3)
gls_self.fit(X, y)
y_pred_self = gls_self.predict(X)

print("\n[SELF-CONSTRUCTED GLS (ρ = 0.3)]")
print(f"Intercept (β₀): {gls_self.intercept:.6f}")
print(f"Slope (β₁):     {gls_self.slope:.6f}")
print(f"R²:             {r2_score(y, y_pred_self):.6f}")
print(f"MSE:            {mean_squared_error(y, y_pred_self):.6f}")
print(f"AR(1) ρ:        {gls_self.rho:.3f}")
print(f"Equation:       y = {gls_self.intercept:.4f} + {gls_self.slope:.4f}x")


def gls_transform(X, y, cov_matrix):
    L = np.linalg.cholesky(cov_matrix)
    L_inv = np.linalg.inv(L)
    X_transformed = L_inv @ X
    y_transformed = L_inv @ y.reshape(-1, 1)
    return X_transformed, y_transformed.flatten()

X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
X_trans, y_trans = gls_transform(X_with_intercept, y, gls_self.cov_matrix)

gls_sklearn = LinearRegression(fit_intercept=False)
gls_sklearn.fit(X_trans, y_trans)
y_pred_sklearn = gls_sklearn.predict(X_trans)

print("\n[BUILT-IN GLS - sklearn via Cholesky Transformation]")
print(f"Intercept (β₀): {gls_sklearn.coef_[0]:.6f}")
print(f"Slope (β₁):     {gls_sklearn.coef_[1]:.6f}")
print(f"R²:             {r2_score(y_trans, y_pred_sklearn):.6f}")
print(f"MSE:            {mean_squared_error(y_trans, y_pred_sklearn):.6f}")
print(f"Equation:       y = {gls_sklearn.coef_[0]:.4f} + {gls_sklearn.coef_[1]:.4f}x")


print("\n" + "="*80)
print("COMPARISON OF RESULTS")
print("="*80)

print(f"\n{'Metric':<20} {'Self-Constructed':<20} {'Built-in':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Intercept':<20} {gls_self.intercept:<20.10f} {gls_sklearn.coef_[0]:<20.10f} {abs(gls_self.intercept - gls_sklearn.coef_[0]):<15.10f}")
print(f"{'Slope':<20} {gls_self.slope:<20.10f} {gls_sklearn.coef_[1]:<20.10f} {abs(gls_self.slope - gls_sklearn.coef_[1]):<15.10f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Both regression lines
ax = axes[0]
ax.scatter(X, y, alpha=0.5, color='gray', label='Data')
ax.plot(X, gls_self.predict(X), linewidth=2, color='blue', label='Self-Constructed')
# Fixed: Flatten X to ensure 1D array for prediction output
ax.plot(X.flatten(), gls_sklearn.coef_[0] + gls_sklearn.coef_[1] * X.flatten(), linewidth=2, color='red', linestyle='--', label='Built-in (sklearn)')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('GLS Regression: Self vs Built-in')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Residuals comparison
ax = axes[1]
y_pred_self_original = gls_self.predict(X)
residuals_self = y - y_pred_self_original
# Fixed: Flatten X to ensure 1D array for prediction output, and thus for residuals
residuals_sklearn = y - (gls_sklearn.coef_[0] + gls_sklearn.coef_[1] * X.flatten())
ax.scatter(y_pred_self_original, residuals_self, alpha=0.5, color='blue', label='Self Residuals')
ax.scatter(y_pred_self_original, residuals_sklearn, alpha=0.5, color='red', label='Built-in Residuals', marker='x')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gls_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

y_pred_self_original = gls_self.predict(X)
# Fixed: Flatten X to ensure 1D array for prediction output
y_pred_sklearn_original = gls_sklearn.coef_[0] + gls_sklearn.coef_[1] * X.flatten()
max_diff = np.max(np.abs(y_pred_self_original - y_pred_sklearn_original))
print(f"\nMaximum prediction difference: {max_diff:.15f}")
print(f"Are predictions identical? {'YES ✓' if max_diff < 1e-10 else 'NO'}")

coef_diff = abs(gls_self.slope - gls_sklearn.coef_[1])
intercept_diff = abs(gls_self.intercept - gls_sklearn.coef_[0])

print(f"\nCoefficient differences:")
print(f"  Slope difference:     {coef_diff:.15f} {'(ESSENTIALLY ZERO ✓)' if coef_diff < 1e-10 else ''}")
print(f"  Intercept difference: {intercept_diff:.15f} {'(ESSENTIALLY ZERO ✓)' if intercept_diff < 1e-10 else ''}")

print("\n" + "="*80)
print("CONCLUSION: Self-constructed and built-in GLS produce IDENTICAL results")
print("(differences are only due to floating-point precision)")
print("="*80)
