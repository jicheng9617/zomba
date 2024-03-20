import numpy as np 
from mpire import WorkerPool
from typing import Union
import copy

import matplotlib.pyplot as plt

def experimenter(banditAlgrithms: dict, 
                 environment: any, 
                 num_rep: int, 
                 num_round: int, 
                 num_arm: int, 
                 ) -> None:
    # regrets = dict((key, []) for key in banditAlgrithms.keys())
    # for i in range(num_rep): 
    #     environment.reset(num_arm=num_arm)
    #     with WorkerPool() as Pool: 
    #         args = [(alg, environment, num_arm, num_round) for alg in banditAlgrithms.values()]
    #         regs = Pool.map(single_run_regret, args)
    #     for i, value in enumerate(regrets.values()): 
    #         value.append(regs[i])
    # for key, value in regrets.items(): 
    #     regrets[key] = np.cumsum(np.vstack(value), axis=1)
    # return regrets
    regrets = dict((key, []) for key in banditAlgrithms.keys())
    for alg_name, alg in banditAlgrithms.items(): 
        alg_pool = [copy.deepcopy(alg) for _ in range(num_rep)] 
        env_pool = [copy.deepcopy(environment) for _ in range(num_rep)] 
        with WorkerPool() as Pool: 
            args = [(alg_pool[i], env_pool[i], num_arm, num_round) for i in range(num_rep)]
            regs_alg = Pool.map(single_run_regret, args) 
        regrets[alg_name] = np.vstack(regs_alg).cumsum(axis=1)
    return regrets



def single_run_regret(banditAlgorithm: any, 
                      environment: any, 
                      num_arm: int, 
                      num_round: int, 
                      ) -> list: 
    regrets = [] 
    banditAlgorithm.reset(num_arm=num_arm)
    environment.reset(num_arm=num_arm)
    for _ in range(num_round): 
        X = environment.observe_context()
        a_t = banditAlgorithm.take_action(context=X)
        reg_t = environment.get_regret(arm=a_t)
        regrets.append(reg_t)
        r_t = environment.get_reward(arm=a_t)
        banditAlgorithm.update((a_t, r_t, X[a_t]))

    return regrets


def plot_regrets(regrets: dict, 
                 fill_between: bool = True): 
    
    fig, ax = plt.subplots()
    for key, value in regrets.items(): 
        mean = np.mean(value, axis=0).squeeze() 
        max = np.max(value, axis=0).squeeze() 
        min = np.min(value, axis=0).squeeze() 
        ax.plot(np.arange(1, value.shape[1]+1), mean, linestyle='-', label=key) 
        if fill_between: ax.fill_between(np.arange(1, value.shape[1]+1), max, min, alpha=.1, linewidth=0)

    return fig, ax





if __name__ == "__main__": 
    print()
    from zomba.singleObjective.stochastic import SLBUCB
    from zomba.singleObjective.simulator import SLBSimulator

    d = 5
    K = 10
    T = 3000
    algorithms = {"UCB(1)": SLBUCB(num_dim=d, num_arm=K, alpha=1.),
                  "UCB(2)": SLBUCB(num_dim=d, num_arm=K, alpha=.5),
                  "UCB(3)": SLBUCB(num_dim=d, num_arm=K, alpha=.1),
                  "UCB(4)": SLBUCB(num_dim=d, num_arm=K, alpha=.05),
                  "UCB(5)": SLBUCB(num_dim=d, num_arm=K, alpha=.01),
                  "UCB(6)": SLBUCB(num_dim=d, num_arm=K, alpha=.005),}

    regrets = experimenter(
        banditAlgrithms=algorithms, 
        environment=SLBSimulator(num_dim=d, num_arm=K),
        num_arm=K, 
        num_rep=5, 
        num_round=T,
    )

    print(regrets)

    fig, ax = plot_regrets(regrets)
    fig.legend() 
    fig.show() 
