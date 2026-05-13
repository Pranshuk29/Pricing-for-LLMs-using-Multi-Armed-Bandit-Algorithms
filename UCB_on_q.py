import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Environment Setup (From Illustrative Example)
# ==========================================
q_true = np.array([0.95, 0.80, 0.70, 0.40, 0.90])
c = np.array([17.10, 4.00, 1.40, 0.40, 10.80])
R = 20.0
K = len(q_true) # Number of LLMs (actions)

# Theoretical Optimal Values (Calculated Analytically)
optimal_principal_utilities = q_true * R - c
i_star = np.argmax(optimal_principal_utilities) # Should be 2 (LLM 2)
t_star = 3.333 # Calculated analytically from IR/IC constraints
V_star = q_true[i_star] * (R - t_star) # Optimal Expected Principal Utility

# ==========================================
# 2. Initialization for Learning
# ==========================================
T = 500000 # Total number of rounds

# Tracking variables
q_hat = np.ones(K) # Optimistic initialization
N = np.ones(K)     # Start with 1 to avoid division by zero
S = np.ones(K)     # Start with 1 (optimistic)

# History trackers for plotting
history_t = np.zeros(T)
history_i_agent = np.zeros(T)
history_regret = np.zeros(T)
history_q_hat = np.zeros((T, K))

cumulative_regret = 0.0

# ==========================================
# 3. UCB Learning Loop
# ==========================================
for round_num in range(1, T + 1):

    # Step 2: Construct optimistic estimates (UCB)
    q_ucb = q_hat + np.sqrt((2 * np.log(round_num)) / N)
    q_ucb = np.clip(q_ucb, 0.001, 1.0) # Clip between >0 and 1 for validity

    # Step 3: Select optimistic best action
    i_hat = np.argmax(q_ucb * R - c)

    # Step 4: Compute payment using q_ucb (IR and IC constraints)
    t_candidates = [c[i_hat] / q_ucb[i_hat]] # Start with IR constraint

    # Add IC constraints for all j where q_ucb[j] < q_ucb[i_hat]
    for j in range(K):
        if q_ucb[j] < q_ucb[i_hat]:
            t_candidates.append((c[i_hat] - c[j]) / (q_ucb[i_hat] - q_ucb[j]))

    # Optimal optimistic payment
    t = max(t_candidates)
    if t > R:
        t = R # Feasibility constraint

    # Step 5 & 6: Offer payment, Agent selects actual action
    # Agent knows true q and c, maximizes its own utility
    agent_utilities = q_true * t - c
    i_agent = np.argmax(agent_utilities)

    # Check if agent's IR is satisfied (utility >= 0)
    if agent_utilities[i_agent] < 0:
        i_agent = -1 # Agent rejects the contract
        success = False
        actual_principal_utility = 0
    else:
        # Step 7: Observe outcome (Bernoulli trial based on true q)
        success = np.random.rand() < q_true[i_agent]
        actual_principal_utility = (R - t) if success else 0

    # Step 8: Update estimates
    # The principal infers the agent took the intended action (i_hat)
    # because the payment was designed to make i_hat Incentive Compatible under q_ucb.
    N[i_hat] += 1
    S[i_hat] += 1 if success else 0
    q_hat[i_hat] = S[i_hat] / N[i_hat]

    # Calculate Regret
    # Regret is difference between theoretical optimal utility and actual expected utility
    if i_agent != -1:
        expected_utility_this_round = q_true[i_agent] * (R - t)
    else:
        expected_utility_this_round = 0

    regret_this_round = V_star - expected_utility_this_round
    cumulative_regret += regret_this_round

    # Record history
    history_t[round_num - 1] = t
    history_i_agent[round_num - 1] = i_agent
    history_regret[round_num - 1] = cumulative_regret
    history_q_hat[round_num - 1, :] = q_hat.copy()

# ==========================================
# 4. Plotting the Results
# ==========================================
plt.figure(figsize=(16, 10))
plt.style.use('seaborn-v0_8-darkgrid')

# Graph 1: Cumulative Regret
plt.subplot(2, 2, 1)
plt.plot(history_regret, label='Cumulative Regret', color='red', linewidth=2)
plt.title('1. Cumulative Regret over Time(UCB on q)')
plt.xlabel('Rounds (T)')
plt.ylabel('Cumulative Regret')
plt.legend()

# Graph 2: Convergence of Estimates (q_hat to True q)
plt.subplot(2, 2, 2)
colors = ['blue', 'orange', 'green', 'purple', 'brown']
for i in range(K):
    plt.plot(history_q_hat[:, i], label=f'q_hat_{i}', color=colors[i], alpha=0.7)
    plt.axhline(y=q_true[i], color=colors[i], linestyle='--', alpha=0.5)
plt.title('2. Convergence of Estimates ($\hat{q}$)')
plt.xlabel('Rounds (T)')
plt.ylabel('Success Probability')
plt.legend(loc='lower right')

# Graph 3: Payment convergence
plt.subplot(2, 2, 3)
# Use a rolling average to smooth the payment plot
rolling_window = 50
rolling_t = np.convolve(history_t, np.ones(rolling_window)/rolling_window, mode='valid')
plt.plot(range(rolling_window-1, T), rolling_t, label='Offered Payment $t$ (Smoothed)', color='teal', linewidth=2)
plt.axhline(y=t_star, color='black', linestyle='--', label=f'Optimal $t^*$ = {t_star}')
plt.title('3. Payment Convergence')
plt.xlabel('Rounds (T)')
plt.ylabel('Payment $t$')
plt.legend()

# Graph 4: Induced Action Frequency
plt.subplot(2, 2, 4)
plt.scatter(range(T), history_i_agent, alpha=0.1, s=10, color='indigo')
plt.axhline(y=i_star, color='black', linestyle='--', label=f'Optimal Action $i^*$ = {i_star}')
plt.yticks(range(-1, K), ['Rejected'] + [f'LLM {i}' for i in range(K)])
plt.title('4. Agent Chosen Action Over Time')
plt.xlabel('Rounds (T)')
plt.ylabel('Action Selected')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Final Estimated q: {np.round(q_hat, 3)}")
print(f"True q:            {q_true}")