import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. The Logistic (Sigmoid) Function
#    ŷ = 1 / (1 + e^(-w^T * x))
# ─────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ─────────────────────────────────────────────
# 2. Data  {x_i, y_i}  (movie features → like/dislike)
#    Features: [action_score, romance_score, rating]
#    Label:     1 = like,  0 = dislike
# ─────────────────────────────────────────────
np.random.seed(42)
X = np.array([
    [0.9, 0.1, 8.5],   # action movie → liked
    [0.8, 0.2, 7.8],
    [0.7, 0.3, 8.0],
    [0.1, 0.9, 7.2],   # romance movie → disliked
    [0.2, 0.8, 6.5],
    [0.1, 0.8, 7.0],
    [0.5, 0.5, 5.0],   # mixed → disliked
    [0.6, 0.4, 7.5],   # mixed → liked
])
y = np.array([1, 1, 1, 0, 0, 0, 0, 1])  # ground-truth labels

N, d = X.shape   # 8 samples, 3 features

# ─────────────────────────────────────────────
# 3. Model: initialise weights w and bias b
# ─────────────────────────────────────────────
w = np.zeros(d)   # weight vector
b = 0.0            # bias
lr = 0.1           # learning rate
epochs = 100000

# ─────────────────────────────────────────────
# 4. Loss Function  (MSE, as shown in slide 2)
#    L(w,b) = (1/N) Σ (ŷ_i - y_i)²
# ─────────────────────────────────────────────
def mse_loss(y_hat, y):
    return np.mean((y_hat - y) ** 2)

# ─────────────────────────────────────────────
# 5. Learning Algorithm: Gradient Descent
# ─────────────────────────────────────────────
losses = []

for epoch in range(epochs):
    # Forward pass
    z     = X @ w + b          # linear combination
    y_hat = sigmoid(z)          # predicted probability

    # Compute loss
    loss = mse_loss(y_hat, y)
    losses.append(loss)

    # Gradients (chain rule through sigmoid + MSE)
    error = y_hat - y                          # (N,)
    dL_dz = error * y_hat * (1 - y_hat)       # sigmoid derivative
    dL_dw = (X.T @ dL_dz) / N                 # (d,)
    dL_db = np.mean(dL_dz)

    # Parameter update
    w -= lr * dL_dw
    b -= lr * dL_db

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

# ─────────────────────────────────────────────
# 6. Predictions
# ─────────────────────────────────────────────
z_final     = X @ w + b
y_hat_final = sigmoid(z_final)
predictions = (y_hat_final >= 0.5).astype(int)

print("\n── Results ──")
for i in range(N):
    label = "LIKE" if predictions[i] == 1 else "DISLIKE"
    true  = "LIKE" if y[i] == 1 else "DISLIKE"
    print(f"  Movie {i+1}: P(like)={y_hat_final[i]:.2f}  → {label}  (true={true})")

accuracy = np.mean(predictions == y)
print(f"\nAccuracy: {accuracy*100:.1f}%")
print(f"Learned weights: {w.round(3)}   bias: {b:.3f}")

# ─────────────────────────────────────────────
# 7. Plot the sigmoid curve + training loss
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Sigmoid shape
z_range = np.linspace(-6, 6, 200)
ax1.plot(z_range, sigmoid(z_range), 'b-', linewidth=2)
ax1.axhline(0.5, color='r', linestyle='--', label='decision boundary (0.5)')
ax1.axvline(0,   color='gray', linestyle=':')
ax1.set_xlabel("z  =  w·x + b")
ax1.set_ylabel("ŷ  =  σ(z)")
ax1.set_title("Logistic (Sigmoid) Function")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Training loss curve
ax2.plot(losses, 'g-', linewidth=1.5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
ax2.set_title("Training Loss (Gradient Descent)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("logistic_demo.png", dpi=120)
plt.show()
print("\nPlot saved → logistic_demo.png")
