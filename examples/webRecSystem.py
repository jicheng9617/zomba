import numpy as np
import matplotlib.pyplot as plt
from mpire import WorkerPool
import copy
import math

from zomba.singleObjective.applications import webRecEnv
from zomba.singleObjective.stochastic import SLBUCB
from zomba.core import contextMABAgent


def ucb_scalar(observed_rewards, action_history, counts):
    total_counts = np.sum(counts)
    arm_index = action_history[-1]  # Index of the most recent chosen arm
    arm_count = counts[arm_index]
    
    if arm_count == 0:
        return float('inf')  # Return infinity to ensure unexplored arms are chosen
    
    UCB_coefficient = np.sqrt(2 * np.log(total_counts) / arm_count)
    
    # Adjust UCB coefficient based on arm counts and selection frequency
    adjusted_UCB_coefficient = UCB_coefficient * np.sqrt(np.log(total_counts) / (counts + 1)) * np.sqrt(arm_count / total_counts) * np.sqrt(np.log(total_counts) / (arm_count + 1)) * np.sqrt(np.log(total_counts) / (total_counts + 1)) * np.sqrt(1 / (counts + 1)) * np.sqrt(arm_count / total_counts)
    
    return adjusted_UCB_coefficient

def ucb_scalar_log(round: int) -> float: 
    return 1 / math.log(round+2)

class ucb_log(SLBUCB): 
    def take_action(self, context: np.ndarray, alpha: float = None) -> int:
        alpha = ucb_scalar_log(round=self.t)
        return super().take_action(context, alpha)

class ucb_gpt(SLBUCB): 
    def take_action(self, context: np.ndarray, alpha: float = None) -> int:
        alpha = ucb_scalar(observed_rewards=self.reward_list,
                           action_history=self.action_list,
                           counts=self.counts)
        return super().take_action(context, alpha)

def heuristic(round: int, context: np.ndarray, context_history: np.ndarray, action_history: list, rewards_history: list, counts: np.ndarray) -> int:
    n_arms = context.shape[0]
    n_features = context.shape[1]
    epsilon = 1e-6  # Small constant to avoid division by zero
    initial_exploration_factor = 0.15  # Adjusted initial exploration factor
    exploration_decay = 0.97  # Decay factor for exploration factor
    
    # Adjusted exploration factor for balancing exploration and exploitation
    exploration_factor = initial_exploration_factor * exploration_decay ** round
    
    # UCB calculation for each arm
    ucb_values = np.zeros(n_arms)
    for arm in range(n_arms):
        if counts[arm] == 0:
            ucb_values[arm] = float('inf')
        else:
            context_arm = context[arm].reshape(1, -1)
            X = np.vstack([context_history, context_arm])
            y = np.array(rewards_history + [0])  # Include a placeholder for the new arm
            theta = np.linalg.lstsq(X, y, rcond=None)[0]
            est_reward = np.dot(theta, context_arm.T)
            uncertainty = np.sqrt(np.dot(context_arm, np.linalg.inv(np.dot(X.T, X) + epsilon * np.eye(n_features))).dot(context_arm.T))
            ucb_values[arm] = est_reward + exploration_factor * np.sqrt(2 * np.log(round) / counts[arm]) * uncertainty
    
    return np.argmax(ucb_values)

class cmab(contextMABAgent):
    def take_action(self, context: np.ndarray) -> int:
        return heuristic(round=self.round, context=context, context_history=self.context_his,
                         action_history=self.action_list, rewards_history=self.reward_list, counts=self.counts)


def offlineEvaluate(mab, env, num_rounds: int = None):

    env.reset() 
    mab.reset() 
    R = []          # save the total payoff
    # Play each arm once at first
    for i in range(env.num_arm): 
        while env.has_next(): 
            X = env.observe_context() 
            a_t = i
            r_t = env.get_reward(arm=a_t)
            if r_t == None: 
                env.update_id()
                continue 
            R.append(r_t)
            mab.update((a_t, r_t, X[a_t]))
            env.update_id()
            break

    while env.has_next(): 
        if mab.round == num_rounds: 
            break

        X = env.observe_context() 
        a_t = mab.take_action(context=X) 
        r_t = env.get_reward(arm=a_t)
        if r_t == None: 
            env.update_id()
            continue 

        R.append(r_t)
        mab.update((a_t, r_t, X[a_t]))
        env.update_id()

    return R 



def main():
    dataset = np.loadtxt("data/dataset.txt")
    env = webRecEnv(data=dataset)
    K = env.num_arm 
    d = env.num_dim
    env.reset(verbose=1)
    banditAlgs = {"cmab(LLM)": cmab(num_arm=K, num_dim=d), 
                  "ucb(LLM)": ucb_gpt(num_arm=K, num_dim=d), 
                  "ucb(0)": SLBUCB(num_arm=K, num_dim=d, alpha=0.0), 
                  "ucb(0.1)": SLBUCB(num_arm=K, num_dim=d, alpha=0.1), 
                  "ucb(1)": SLBUCB(num_arm=K, num_dim=d, alpha=1), }
    with WorkerPool() as Pool: 
        args = [(mab, copy.deepcopy(env)) for mab in banditAlgs.values()]
        rewards = Pool.map(offlineEvaluate, args)

    fig = plt.figure()
    for i, r in enumerate(rewards): 
        cum_r = np.array(r).cumsum() / np.arange(len(r))
        plt.plot(range(len(cum_r)), cum_r, label=list(banditAlgs.keys())[i])
    plt.legend()
    plt.xlabel("Rounds")
    plt.ylabel("Reward per round")
    plt.grid()
    fig.savefig('webRec.png', dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    main()

