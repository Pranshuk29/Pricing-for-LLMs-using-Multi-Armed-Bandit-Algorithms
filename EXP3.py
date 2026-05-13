import torch
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 0. GPU Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Environment Setup (Ground Truth)
# ==========================================
# Move ground truth to GPU
q_true = torch.tensor([0.95, 0.80, 0.70, 0.40, 0.90], device=device)
c = torch.tensor([17.10, 4.00, 1.40, 0.40, 10.80], device=device)
R = 20.0
K_llms = len(q_true)

# Theoretical Optimal Values (Computed on CPU/NumPy for ease)
optimal_principal_utilities = (q_true.cpu().numpy() * R) - c.cpu().numpy()
i_star = np.argmax(optimal_principal_utilities)
t_star = 3.333
V_star = q_true.cpu().numpy()[i_star] * (R - t_star)

# ==========================================
# 2. Continuous EXP3 Initialization
# ==========================================
T = 300000 # Number of rounds

# Continuous EXP3 Hyperparameters
gamma = 0.15
eta = 0.05
sigma_explore = 0.5
sigma_update = 0.5

# PRE-ALLOCATE TENSORS ON GPU FOR SPEED (Avoids list appends)
past_t = torch.zeros(T, device=device)
log_w = torch.zeros(T, device=device)

history_t = np.zeros(T)
history_regret = np.zeros(T)
history_i_agent = np.zeros(T)
cumulative_regret = 0.0

# Helper function for fast Gaussian PDF on GPU
def gaussian_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * math.sqrt(2 * math.pi))

# ==========================================
# 3. Kernelized EXP3 Learning Loop
# ==========================================
for round_num in range(T):

    if round_num == 0:
        t = torch.empty(1, device=device).uniform_(0.1, R).item()
        p_t = 1.0 / R
    else:
        # Get history up to current round
        current_past_t = past_t[:round_num]
        current_log_w = log_w[:round_num]

        # Calculate current weights
        max_log = torch.max(current_log_w)
        weights = torch.exp(current_log_w - max_log)
        weights /= torch.sum(weights)

        # Decide: Uniform Exploration or KDE Exploitation?
        if torch.rand(1, device=device).item() < gamma:
            t = torch.empty(1, device=device).uniform_(0.1, R).item()
        else:
            # Sample base index using multinomial
            base_idx = torch.multinomial(weights, 1).item()
            base_val = current_past_t[base_idx].item()
            t = torch.normal(mean=base_val, std=sigma_explore, size=(1,)).item()
            t = max(0.1, min(t, R)) # Clip to valid range

        # Calculate PDF of picking this t
        uniform_density = gamma / R

        # Vectorized KDE computation on GPU
        pdfs = gaussian_pdf(t, current_past_t, sigma_explore)
        kde_density = (1 - gamma) * torch.sum(weights * pdfs).item()
        p_t = uniform_density + kde_density

    # --- STEP 2: Agent makes a hidden decision ---
    agent_utilities = q_true * t - c
    max_util, max_idx = torch.max(agent_utilities, 0)

    if max_util.item() < 0:
        i_agent = -1 # Rejected
        success = False
    else:
        i_agent = max_idx.item()
        success = torch.rand(1, device=device).item() < q_true[i_agent].item()

    # --- STEP 3: Principal observes actual reward ---
    actual_reward = (R - t) if success else 0.0
    scaled_reward = (actual_reward / R) / p_t

    # --- STEP 4: Continuous Kernel Update ---
    # Store new point
    past_t[round_num] = t
    log_w[round_num] = 0.0

    # Vectorized update of all historical weights on GPU
    # Only update up to round_num + 1
    active_past_t = past_t[:round_num+1]
    kernel_distances = torch.exp(-((active_past_t - t)**2) / (2 * sigma_update**2))

    # In-place update
    log_w[:round_num+1] += eta * scaled_reward * kernel_distances

    # --- Logging & Regret ---
    if i_agent != -1:
        expected_reward = q_true[i_agent].item() * (R - t)
    else:
        expected_reward = 0.0

    cumulative_regret += (V_star - expected_reward)

    # Store history for CPU plotting
    history_t[round_num] = t
    history_regret[round_num] = cumulative_regret
    history_i_agent[round_num] = i_agent

# ==========================================
# 4. Plotting Results (Back to CPU)
# ==========================================
# Transfer data back to CPU for Matplotlib
cpu_past_t = past_t.cpu().numpy()
cpu_log_w = log_w.cpu().numpy()

plt.figure(figsize=(16, 10))
plt.style.use('seaborn-v0_8-darkgrid')

# Graph 1: Cumulative Regret
plt.subplot(2, 2, 1)
plt.plot(history_regret, color='red', linewidth=2)
plt.title('1. Cumulative Regret (Continuous Kernelized EXP3)')
plt.xlabel('Rounds (T)')
plt.ylabel('Cumulative Regret')

# Graph 2: Payment Scatter
plt.subplot(2, 2, 2)
plt.scatter(range(T), history_t, alpha=0.15, s=5, color='teal')
plt.axhline(y=t_star, color='black', linestyle='--', label=f'Optimal $t^*$ = {t_star}')
plt.title('2. Continuous Explored Payments Over Time')
plt.xlabel('Rounds (T)')
plt.ylabel('Payment $t$ Offered')
plt.legend()

# Graph 3: Final Kernel Density Estimation (KDE) curve
plt.subplot(2, 2, 3)
x_plot = np.linspace(0, R, 500)
max_log = np.max(cpu_log_w)
final_weights = np.exp(cpu_log_w - max_log)
final_weights /= np.sum(final_weights)

# Calculate final PDF on CPU since it's just for a 500-point plot
from scipy.stats import norm
pdf = np.zeros_like(x_plot)
for pt, w in zip(cpu_past_t, final_weights):
    pdf += w * norm.pdf(x_plot, loc=pt, scale=sigma_explore)

plt.plot(x_plot, pdf, color='purple', linewidth=2)
plt.axvline(x=t_star, color='black', linestyle='--', label=f'Optimal $t^*$ = {t_star}')
plt.title('3. Final Continuous Probability Density Curve')
plt.xlabel('Payment space [0, 20]')
plt.ylabel('Probability Density')
plt.legend()

# Graph 4: Agent Chosen Action
plt.subplot(2, 2, 4)
plt.scatter(range(T), history_i_agent, alpha=0.05, s=10, color='indigo')
plt.axhline(y=i_star, color='black', linestyle='--', label=f'Optimal Action $i^*$ = {i_star}')
plt.yticks(range(-1, K_llms), ['Rejected'] + [f'LLM {i}' for i in range(K_llms)])
plt.title('4. Agent Chosen Action Over Time')
plt.xlabel('Rounds (T)')
plt.ylabel('Action Selected')
plt.legend()

plt.tight_layout()
plt.show()

best_t = x_plot[np.argmax(pdf)]
print(f"Algorithm Peak Density is at: {best_t:.3f}")
print(f"Theoretical Optimal Payment:  {t_star:.3f}")