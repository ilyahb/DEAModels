import pulp as pl
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# CHOOSE BETWEEN "CCR" OR "BCC"
# -----------------------------------------------------------
model_type = "CCR"      # change to "CCR" or "BCC"

# -----------------------------------------------------------
# NEW DATASET with clear CRS vs VRS difference
# -----------------------------------------------------------
# Input (Labor), Output (Production)
X = np.array([10,  6,  9,  4, 12])    # inputs
Y = np.array([40, 30, 65, 20, 55])    # outputs

DMU_names = ["A", "B", "C", "D", "E"]

n = len(X)
dm0 = 0   # Evaluate DMU A

# -----------------------------------------------------------
# BUILD INPUT-ORIENTED DEA MODEL
# -----------------------------------------------------------
prob = pl.LpProblem("DEA_Input_Oriented", pl.LpMinimize)

theta = pl.LpVariable("theta", lowBound=0)
lmb = pl.LpVariable.dicts("lambda", list(range(n)), lowBound=0)

prob += theta

# Input constraint
prob += pl.lpSum(lmb[j] * X[j] for j in range(n)) <= theta * X[dm0]

# Output constraint
prob += pl.lpSum(lmb[j] * Y[j] for j in range(n)) >= Y[dm0]

# BCC convexity constraint
if model_type.upper() == "BCC":
    prob += pl.lpSum(lmb[j] for j in range(n)) == 1
    print("Running BCC (VRS)")
else:
    print("Running CCR (CRS)")

prob.solve()

theta_val = pl.value(theta)
lmb_vals = np.array([pl.value(lmb[j]) for j in range(n)])

print("\nStatus:", pl.LpStatus[prob.status])
print("θ* =", theta_val)
print("λ values:", lmb_vals)

# -----------------------------------------------------------
# PROJECTED POINT A'
# -----------------------------------------------------------
A_proj_input = theta_val * X[dm0]
A_proj_output = Y[dm0]  # output stays fixed for input oriented

# -----------------------------------------------------------
# PLOT FRONTIER
# -----------------------------------------------------------
plt.figure(figsize=(10,6))

# Plot DMUs
plt.scatter(X, Y, s=120)
for i in range(n):
    plt.text(X[i] + 0.2, Y[i] + 1, DMU_names[i], fontsize=11)

# Plot CCR frontier (straight line)
ratio = Y / X
best_idx = np.argmax(ratio)
best_slope = ratio[best_idx]

x_line = np.linspace(0, max(X)+2, 100)
y_line = best_slope * x_line

if model_type.upper() == "CCR":
    plt.plot(x_line, y_line, label="CCR Frontier", linewidth=2)

# Plot BCC frontier (piecewise linear hull)
else:
    sorted_idx = np.argsort(X)
    xs = X[sorted_idx]
    ys = Y[sorted_idx]
    plt.plot(xs, ys, "-o", label="BCC Frontier", linewidth=2)

# Plot projection A'
plt.scatter(A_proj_input, A_proj_output, color="red", s=150, label="Projected A")
plt.text(A_proj_input + 0.2, A_proj_output + 1, "A'", color="red")

plt.xlabel("Input (Labor)")
plt.ylabel("Output")
plt.title(f"DEA {model_type.upper()} Input-Oriented Frontier")
plt.grid(True)
plt.legend()
plt.show()
