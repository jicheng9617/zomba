import numpy as np 

import matplotlib.pyplot as plt

class experimenter: 
    def __init__(self, 
                 banditAlgrithm, 
                 environment, 
                 ) -> None:
        self.alg_pool = banditAlgrithm 
        self.env = environment 

    @property
    def num_alg(self): 
        return len(self.alg_pool)

    def reset(self, 
              num_rep: int, 
              num_round: int, 
              num_arm: int, 
              **kwargs, 
              ) -> None: 
        self.n_rep = num_rep 
        self.T = num_round 
        # initialize records
        self.cum_regret = [] 
        self.rewards = [] 
        # create environment and algorithms
        if isinstance(num_arm, int): self.n_arm = num_arm

    def run(self): 
        #TODO parallel computing
        self.regret = []
        for i in range(self.n_rep): 
            self._reset()
            cum_reg = np.zeros((self.T,self.num_alg))
            reg = np.zeros((self.num_alg))

            for t in range(self.T): 
                actions = self._take_action() 
                reg += np.array(self._get_regret(actions))
                cum_reg[t,:] = reg
                rew = self._observe_reward(actions)
                for k, alg in enumerate(self.alg_pool): alg.update(action=actions[k],
                                                                   reward=rew[k])

            self.regret.append(cum_reg)

        self.regret = np.array(self.regret)

    def visualize(self, 
                  label: list, 
                  fill: bool=True, 
                  ): 
        
        fig, ax = plt.subplots()
        for i in range(self.num_alg): 
            reg = self.regret[:, :, i]
            mean = np.mean(reg, axis=0).squeeze() 
            max = np.max(reg, axis=0).squeeze() 
            min = np.min(reg, axis=0).squeeze() 
            ax.plot(np.arange(1, self.T+1), mean, linestyle='-', label=label[i]) # r"$\epsilon=$ %f" % (eps)
            
            if fill: ax.fill_between(np.arange(1, self.T+1), max, min, alpha=.5, linewidth=0)

        # plt.legend()
        # plt.show()
        return fig, ax

    def _take_action(self): 
        return [
            alg.take_action() for alg in self.alg_pool
        ]
    
    def _get_regret(self, actions: list): 
        return [
            self.env.get_regret(arm=i) for i in actions
        ]
    
    def _observe_reward(self, actions: list): 
        return [ 
            self.env.get_reward(arm=i) for i in actions
        ]
    
    def _reset(self): 
        self.env.reset(num_arm=self.n_arm) 
        for alg in self.alg_pool: alg.reset(num_arm=self.n_arm) 




if __name__ == "__main__": 
    print()
    from zomba.singleObjective.mab import epsilonGreedy
    from zomba.singleObjective.simulator import soSimulatorBernoulliMAB

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