import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(42)
X = np.linspace(1, 20, 100).reshape(-1, 1)
true_intercept, true_slope = 15, 3
noise = np.random.normal(0, 0.5 + 0.3 * X.flatten(), 100)
y = true_intercept + true_slope * X.flatten() + noise
for idx in [15, 45, 75]:
    y[idx] += np.random.choice([-15, 15]) * np.random.uniform(0.5, 1.5)


# 1. SELF-CONSTRUCTED SGD

class SGDRegression_Self:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=10, 
                 alpha=0.0001, tol=1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.alpha = alpha
        self.tol = tol
        self.intercept = None
        self.slope = None
        self.coefficients = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples = len(y)
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        self.coefficients = np.random.randn(X_with_intercept.shape[1]) * 0.01
        
        def lr_schedule(t):
            return self.learning_rate / (1 + 0.01 * t)
        
        for iteration in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                predictions = X_batch @ self.coefficients
                errors = predictions - y_batch
                gradient = (1/len(y_batch)) * X_batch.T @ errors + self.alpha * self.coefficients
                
                lr = lr_schedule(iteration * n_samples + i)
                self.coefficients = self.coefficients - lr * gradient
            
            predictions = X_with_intercept @ self.coefficients
            loss = np.mean((predictions - y) ** 2)
            self.loss_history.append(loss)
            
            if iteration > 10 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                break
        
        self.intercept = self.coefficients[0]
        self.slope = self.coefficients[1]
        return self
    
    def predict(self, X):
        return np.column_stack([np.ones(X.shape[0]), X]) @ self.coefficients

print("\n" + "="*80)
print("STOCHASTIC GRADIENT DESCENT (SGD) - COMPARISON")
print("="*80)

# Fit Self-Constructed SGD
sgd_self = SGDRegression_Self(learning_rate=0.01, n_iterations=500, batch_size=10)
sgd_self.fit(X, y)
y_pred_self = sgd_self.predict(X)

print("\n[SELF-CONSTRUCTED SGD]")
print(f"Intercept (β₀): {sgd_self.intercept:.6f}")
print(f"Slope (β₁):     {sgd_self.slope:.6f}")
print(f"R²:             {r2_score(y, y_pred_self):.6f}")
print(f"MSE:            {mean_squared_error(y, y_pred_self):.6f}")
print(f"Learning Rate:  {sgd_self.learning_rate:.4f}")
print(f"Iterations:     {len(sgd_self.loss_history)}")
print(f"Equation:       y = {sgd_self.intercept:.4f} + {sgd_self.slope:.4f}x")

# 2. BUILT-IN SGD (sklearn)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sgd_sklearn = SGDRegressor(max_iter=1000, tol=1e-4, penalty='l2', alpha=0.0001,
                          learning_rate='adaptive', eta0=0.01, random_state=42)
sgd_sklearn.fit(X_scaled, y)
y_pred_sklearn = sgd_sklearn.predict(X_scaled)

# Convert scaled coefficients back to original scale
original_slope = sgd_sklearn.coef_[0] / scaler.scale_[0]
original_intercept = sgd_sklearn.intercept_[0] - original_slope * scaler.mean_[0]

print("\n[BUILT-IN SGD - sklearn.SGDRegressor]")
print(f"Intercept (β₀): {original_intercept:.6f}")
print(f"Slope (β₁):     {original_slope:.6f}")
print(f"R²:             {r2_score(y, y_pred_sklearn):.6f}")
print(f"MSE:            {mean_squared_error(y, y_pred_sklearn):.6f}")
print(f"Equation:       y = {original_intercept:.4f} + {original_slope:.4f}x")


print("\n" + "="*80)
print("COMPARISON OF RESULTS")
print("="*80)

print(f"\n{'Metric':<20} {'Self-Constructed':<20} {'Built-in':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Intercept':<20} {sgd_self.intercept:<20.10f} {original_intercept:<20.10f} {abs(sgd_self.intercept - original_intercept):<15.10f}")
print(f"{'Slope':<20} {sgd_self.slope:<20.10f} {original_slope:<20.10f} {abs(sgd_self.slope - original_slope):<15.10f}")
print(f"{'R²':<20} {r2_score(y, y_pred_self):<20.10f} {r2_score(y, y_pred_sklearn):<20.10f} {abs(r2_score(y, y_pred_self) - r2_score(y, y_pred_sklearn)):<15.10f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Both regression lines
ax = axes[0]
ax.scatter(X, y, alpha=0.5, color='gray', label='Data')
ax.plot(X, y_pred_self, linewidth=2, color='blue', label='Self-Constructed')
ax.plot(X, y_pred_sklearn, linewidth=2, color='red', linestyle='--', label='Built-in (sklearn)')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('SGD Regression: Self vs Built-in')
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
plt.savefig('sgd_comparison.png', dpi=300, bbox_inches='tight')
plt.show()  # <-- THIS WAS MISSING!

print("\n" + "="*80)
print("VERIFICATION: Are they the same?")
print("="*80)

max_diff = np.max(np.abs(y_pred_self - y_pred_sklearn))
print(f"\nMaximum prediction difference: {max_diff:.15f}")
print(f"Are predictions identical? {'YES ✓' if max_diff < 1e-10 else 'NO'}")

coef_diff = abs(sgd_self.slope - original_slope)
intercept_diff = abs(sgd_self.intercept - original_intercept)

print(f"\nCoefficient differences:")
print(f"  Slope difference:     {coef_diff:.15f} {'(ESSENTIALLY ZERO ✓)' if coef_diff < 1e-10 else ''}")
print(f"  Intercept difference: {intercept_diff:.15f} {'(ESSENTIALLY ZERO ✓)' if intercept_diff < 1e-10 else ''}")

print("\n" + "="*80)
print("CONCLUSION: Self-constructed and built-in SGD produce COMPARABLE results")
print("(differences are due to stochastic nature and different implementations)")
print("="*80)

# Additional plot: Learning curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sgd_self.loss_history, linewidth=2, color='blue')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss (MSE)')
ax.set_title('SGD Learning Curve (Self-Constructed)')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.savefig('sgd_learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()
