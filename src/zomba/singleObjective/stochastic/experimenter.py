import numpy as np 
from mpire import WorkerPool
from typing import Union

import matplotlib.pyplot as plt

def experimenter(self, 
                 banditAlgrithms: dict, 
                 environment: any, 
                 num_rep: int, 
                 num_round: int, 
                 num_arm: int, 
                 ) -> None:
    
    # initialize records
    regrets = dict.fromkeys(banditAlgrithms, [])
    for i in range(self.n_rep): 
        environment.reset(num_arm=num_arm)
        with WorkerPool() as Pool: 
            args = [(alg, environment, num_arm, num_round) for alg in banditAlgrithms.values()]
            regs = Pool.map(single_run_regret, args)
        for i, key in enumerate(regrets.keys()): regrets[key].append(regs[i])
    for key, value in regrets: regrets[key] = np.cumsum(np.vstack(value), axis=1)
    return regrets


def single_run_regret(banditAlgorithm: any, 
                      environment: any, 
                      num_arm: int, 
                      num_round: int, 
                      ) -> list: 
    regrets = [] 
    banditAlgorithm.reset(num_arm=num_arm)
    for _ in range(num_round): 
        X = environment.observe_context()
        a_t = banditAlgorithm.take_action(context=X)
        r_t = environment.get_reward(arm=a_t)
        reg_t = environment.get_regret(arm=a_t)
        regrets.append(reg_t)
        banditAlgorithm.update(action=a_t, reward=r_t, context=X)

    return regrets


def plot_regrets(regrets: dict, 
                 fill_between: bool = True): 
    
    fig, ax = plt.subplots()
    for key, value in regrets: 
        mean = np.mean(value, axis=0).squeeze() 
        max = np.max(value, axis=0).squeeze() 
        min = np.min(value, axis=0).squeeze() 
        ax.plot(np.arange(1, value.shape[1]+1), mean, linestyle='-', label=key) 
        if fill_between: ax.fill_between(np.arange(1, value.shape[1]+1), max, min, alpha=.5, linewidth=0)

    return ax





if __name__ == "__main__": 
    print()
    from zomba.singleObjective.stochastic import SLBUCB
    from zomba.singleObjective.simulator import SLBSimulator

    algorithms = [
        epsilonGreedy(epsilon=1e-4),
        epsilonGreedy(epsilon=1e-3), 
        epsilonGreedy(epsilon=1e-2), 
    ]

    solver = experimenter(
        banditAlgrithm=algorithms, 
        environment=soSimulatorBernoulliMAB()
    )

    solver.reset(
        num_rep=10, 
        num_arm=5, 
        num_round=3000
    )

    solver.run()

    fig, ax = solver.visualize(label=["0", '1', '2'])

    plt.legend() 
    plt.show() 

    input()