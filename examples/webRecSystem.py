import numpy as np
import matplotlib.pyplot as plt
from mpire import WorkerPool
import copy

from zomba.singleObjective.applications import webRecEnv
from zomba.singleObjective.stochastic import SLBUCB


def ucb_scalar1(round: int, observed_rewards: list, action_history: list) -> float:
    return 1 / (round+1)


class ucb1(SLBUCB): 
    def take_action(self, context: np.ndarray, alpha: float = None) -> int:
        alpha = ucb_scalar1(round=self.t,
                            observed_rewards=self.reward_list,
                            action_history=self.action_list)
        return super().take_action(context, alpha)


def offlineEvaluate(mab, env, num_rounds: int = None):

    env.reset() 
    mab.reset() 
    R = []          # save the total payoff

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
    env = webRecEnv()
    K = env.num_arm 
    d = env.num_dim
    env.reset(verbose=1)
    banditAlgs = {"ucb(gpt)": ucb1(num_arm=K, num_dim=d), 
                  "ucb(0.01)": SLBUCB(num_arm=K, num_dim=d, alpha=0.01), 
                  "ucb(0.05)": SLBUCB(num_arm=K, num_dim=d, alpha=0.05), 
                  "ucb(0.1)": SLBUCB(num_arm=K, num_dim=d, alpha=0.1), 
                  "ucb(0.5)": SLBUCB(num_arm=K, num_dim=d, alpha=0.5), }
    with WorkerPool() as Pool: 
        args = [(mab, copy.deepcopy(env)) for mab in banditAlgs.values()]
        rewards = Pool.map(offlineEvaluate, args)

    fig = plt.figure()
    for i, r in enumerate(rewards): 
        cum_r = np.array(r).cumsum()
        plt.plot(range(len(cum_r)), cum_r, label=list(banditAlgs.keys())[i])
    plt.legend()
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative rewards")
    plt.grid()
    fig.savefig('webRec.png', dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    main()

