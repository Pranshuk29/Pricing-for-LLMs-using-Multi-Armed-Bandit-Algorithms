import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==========================================
# 1. Environment Setup (Ground Truth)
# ==========================================
q_true = np.array([0.95, 0.80, 0.70, 0.40, 0.90])
c = np.array([17.10, 4.00, 1.40, 0.40, 10.80])
R = 20.0
K_llms = len(q_true)

# Theoretical Optimal Values
optimal_principal_utilities = q_true * R - c
i_star = np.argmax(optimal_principal_utilities)
t_star = 3.333
V_star = q_true[i_star] * (R - t_star)

# ==========================================
# 2. Continuous EXP3 Initialization (Fixed Memory)
# ==========================================
T = 300000
M = 200 # Number of "Anchor Points" to bound memory usage

# We maintain weights for fixed anchors, NOT an infinite history list
anchors = np.linspace(0.1, R, M)
log_w = np.zeros(M)

# Hyperparameters
sigma_explore = 0.5   # How wide we search around an anchor
sigma_update = 0.2    # How wide the reward "splashes" onto nearby anchors

history_t = np.zeros(T)
history_regret = np.zeros(T)
history_i_agent = np.zeros(T)
cumulative_regret = 0.0

# ==========================================
# 3. Optimized Continuous Learning Loop
# ==========================================
for step in range(T):
    round_num = step + 1

    # 1. Decaying hyperparameters (This fixes the Linear Regret!)
    # Gamma shrinks over time, forcing exploitation
    gamma_t = min(1.0, 2.0 / np.sqrt(round_num))
    eta_t = min(0.5, 2.0 / np.sqrt(round_num))

    # 2. Calculate stable weights for our anchors
    max_log = np.max(log_w)
    weights = np.exp(log_w - max_log)
    weights /= np.sum(weights)

    # 3. Select a Continuous Payment t
    if np.random.rand() < gamma_t:
        # Uniform Exploration
        t = np.random.uniform(0.1, R)
    else:
        # Kernel Exploitation: Pick an anchor based on weight, add continuous noise
        base_idx = np.random.choice(M, p=weights)
        t = np.random.normal(anchors[base_idx], sigma_explore)
        t = np.clip(t, 0.1, R)

    # 4. Calculate the Probability Density of picking this exact t
    uniform_density = gamma_t / (R - 0.1)
    kde_density = (1 - gamma_t) * np.sum(weights * norm.pdf(t, loc=anchors, scale=sigma_explore))
    p_t = uniform_density + kde_density

    # 5. Agent makes a hidden decision
    agent_utilities = q_true * t - c
    if np.max(agent_utilities) < 0:
        i_agent = -1
        success = False
    else:
        i_agent = np.argmax(agent_utilities)
        success = np.random.rand() < q_true[i_agent]

    # 6. Observe reward and apply Importance Sampling
    actual_reward = (R - t) if success else 0.0
    scaled_reward = (actual_reward / R) / p_t

    # 7. Anchor Point Kernel Update (This fixes the Memory Crash!)
    # We update all 200 anchors based on how close they are to 't'
    kernel_distances = np.exp(-((anchors - t)**2) / (2 * sigma_update**2))
    log_w += eta_t * scaled_reward * kernel_distances

    # 8. Regret Tracking
    expected_reward = q_true[i_agent] * (R - t) if i_agent != -1 else 0.0
    cumulative_regret += (V_star - expected_reward)

    history_t[step] = t
    history_regret[step] = cumulative_regret
    history_i_agent[step] = i_agent

# ==========================================
# 4. Plotting Results
# ==========================================
plt.figure(figsize=(16, 10))
plt.style.use('seaborn-v0_8-darkgrid')

plt.subplot(2, 2, 1)
plt.plot(history_regret, color='red', linewidth=2)
plt.title('1. Cumulative Regret (Optimized Continuous EXP3)')
plt.xlabel('Rounds (T)')
plt.ylabel('Cumulative Regret')

plt.subplot(2, 2, 2)
plt.scatter(range(T), history_t, alpha=0.05, s=2, color='teal')
plt.axhline(y=t_star, color='black', linestyle='--', label=f'Optimal $t^*$ = {t_star}')
plt.title('2. Continuous Explored Payments Over Time')
plt.xlabel('Rounds (T)')
plt.ylabel('Payment $t$ Offered')
plt.legend()

plt.subplot(2, 2, 3)
max_log = np.max(log_w)
final_weights = np.exp(log_w - max_log)
final_weights /= np.sum(final_weights)
plt.plot(anchors, final_weights, color='purple', linewidth=2, label='Learned Anchor Density')
plt.axvline(x=t_star, color='black', linestyle='--', label=f'Optimal $t^*$ = {t_star}')
plt.title('3. Final Anchor Probability Weights')
plt.xlabel('Payment space [0, 20]')
plt.ylabel('Probability Density')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(range(T), history_i_agent, alpha=0.05, s=5, color='indigo')
plt.axhline(y=i_star, color='black', linestyle='--', label=f'Optimal Action $i^*$ = {i_star}')
plt.yticks(range(-1, K_llms), ['Rejected'] + [f'LLM {i}' for i in range(K_llms)])
plt.title('4. Agent Chosen Action Over Time')
plt.xlabel('Rounds (T)')
plt.ylabel('Action Selected')
plt.legend()

plt.tight_layout()
plt.show()

best_t = anchors[np.argmax(final_weights)]
print(f"Algorithm Peak Density is near: {best_t:.3f}")
print(f"Theoretical Optimal Payment:    {t_star:.3f}")