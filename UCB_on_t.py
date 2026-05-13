import numpy as np
import matplotlib.pyplot as plt

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
# 2. Discretization of the Payment Space
# ==========================================
T = 300000 # Total number of rounds
M = 200    # Number of discrete candidate payments (Grid size)

# Create an array of candidate payments evenly spaced between 0.1 and R
payments = np.linspace(0.1, R, M)

# UCB Trackers (Notice these track PAYMENTS, not LLMs)
N = np.zeros(M)       # Number of times each payment has been offered
V_hat = np.zeros(M)   # Estimated expected principal utility for each payment

# History trackers
history_t = np.zeros(T)
history_regret = np.zeros(T)
history_i_agent = np.zeros(T)
cumulative_regret = 0.0

# ==========================================
# 3. UCB Learning Loop (Over Payments)
# ==========================================
for round_num in range(T):

    # --- Step 1: Select a Payment using UCB ---
    if round_num < M:
        # Play each payment arm exactly once to initialize
        m_star = round_num
    else:
        # Calculate UCB for each payment.
        # Note: We scale the exploration term by R because rewards are [0, R], not [0, 1]
        ucb_values = V_hat + R * np.sqrt((2 * np.log(round_num)) / N)
        m_star = np.argmax(ucb_values)

    t = payments[m_star]

    # --- Step 2: Agent evaluates privately ---
    agent_utilities = q_true * t - c
    if np.max(agent_utilities) < 0:
        i_agent = -1 # Agent rejects
        success = False
        actual_reward = 0.0
    else:
        i_agent = np.argmax(agent_utilities)
        success = np.random.rand() < q_true[i_agent]
        actual_reward = (R - t) if success else 0.0

    # --- Step 3: Principal observes ONLY the reward and updates the Payment Arm ---
    N[m_star] += 1
    # Incremental update of the average expected utility for THIS specific payment
    V_hat[m_star] += (actual_reward - V_hat[m_star]) / N[m_star]

    # --- Regret Tracking ---
    if i_agent != -1:
        expected_reward = q_true[i_agent] * (R - t)
    else:
        expected_reward = 0.0

    cumulative_regret += (V_star - expected_reward)

    history_t[round_num] = t
    history_i_agent[round_num] = i_agent
    history_regret[round_num] = cumulative_regret

# ==========================================
# 4. Plotting Results
# ==========================================
plt.figure(figsize=(16, 10))
plt.style.use('seaborn-v0_8-darkgrid')

# Graph 1: Cumulative Regret
plt.subplot(2, 2, 1)
plt.plot(history_regret, color='red', linewidth=2)
plt.title('1. Cumulative Regret (Discretized UCB)')
plt.xlabel('Rounds (T)')
plt.ylabel('Cumulative Regret')

# Graph 2: Payment Scatter (Showing Exploration vs Exploitation)
plt.subplot(2, 2, 2)
plt.scatter(range(T), history_t, alpha=0.1, s=5, color='teal')
plt.axhline(y=t_star, color='black', linestyle='--', label=f'Theoretical $t^*$ = {t_star}')
plt.title('2. Explored Candidate Payments Over Time')
plt.xlabel('Rounds (T)')
plt.ylabel('Payment $t$ Offered')
plt.legend()

# Graph 3: Estimated Expected Utility across the Payment Grid
plt.subplot(2, 2, 3)
plt.plot(payments, V_hat, color='purple', linewidth=2, label='Estimated Utility $\hat{V}(t)$')
plt.axvline(x=t_star, color='black', linestyle='--', label=f'Optimal $t^*$ = {t_star}')
plt.title('3. Learned Utility Curve of the "Black Box"')
plt.xlabel('Candidate Payments [0, 20]')
plt.ylabel('Expected Principal Utility $\hat{V}(t)$')
plt.legend()

# Graph 4: Agent Chosen Action
plt.subplot(2, 2, 4)
plt.scatter(range(T), history_i_agent, alpha=0.05, s=10, color='indigo')
plt.axhline(y=i_star, color='black', linestyle='--', label=f'Optimal Action $i^*$ = {i_star}')
plt.yticks(range(-1, K_llms), ['Rejected'] + [f'LLM {i}' for i in range(K_llms)])
plt.title('4. Hidden Agent Actions Over Time')
plt.xlabel('Rounds (T)')
plt.ylabel('Action Selected')
plt.legend()

plt.tight_layout()
plt.show()

best_payment_idx = np.argmax(V_hat)
print(f"Algorithm Selected Best Payment: {payments[best_payment_idx]:.3f}")
print(f"Theoretical Optimal Payment:     {t_star:.3f}")