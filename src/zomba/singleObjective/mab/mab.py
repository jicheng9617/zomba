import numpy as np 

from ...core import soMABAgent



class epsilonGreedy(soMABAgent):
    def __init__(self, 
                 num_arm: int=None, 
                 epsilon: float=.01,
                 ) -> None:
        """_summary_

        Parameters
        ----------
        num_arm : int, optional
            _description_, by default None
        """
        super(epsilonGreedy, self).__init__(num_arm=num_arm)
        self.epsilon = epsilon 

    # def reset(self, 
    #           num_arm: int = None, 
    #           init_estimates: float = 1., 
    #           epsilon: float=None, 
    #           ) -> None:
    #     super().reset(num_arm=num_arm, init_estimates=init_estimates) 
    #     if epsilon is not None: self.epsilon = epsilon 
        
    def _eval_epsilon(self): 
        return self.epsilon

    def take_action(self):

        epsilon = self._eval_epsilon()

        if np.random.random() < epsilon:
            return np.random.randint(0, self.K)  # random selection
        else:
            return np.argmax(self.estimates)  # greedy selection


class upperConfidenceBound(soMABAgent): 
    def __init__(self, 
                 num_arm: int =None, 
                 delta: float=None, 
                 coff: float = 1., 
                 ) -> None:
        """
        Upper confidence bound (UCB) algorithm 

        Parameters
        ----------
        num_arm : int, optional
            # arms, by default None
        delta : float, optional
            confidence level, by default .05
        c : float, optional
            multiplier for uncertainty, by default 1.
        """
        super().__init__(num_arm=num_arm)
        self.delta = delta 
        self.c = coff 

    @property
    def uncertainty(self): 
        if self.delta is None: 
            return np.sqrt( np.log(self.total_counts) / 2*(self.counts+1) )
        else: 
            return np.sqrt( -np.log(self.delta) / 2*(self.counts+1) )
    
    def take_action(self, 
                    delta: float=None, 
                    coff: float=None, 
                    ) -> int:
        if delta is not None: self.delta = delta 
        if coff is not None: self.c = coff 

        ucb = self.estimates + self.c * self.uncertainty 
        return np.argmax(ucb)


class ThompsonSampling(soMABAgent): 
    def __init__(self, 
                 num_arm: int = None, 
                 ) -> None:
        super().__init__(num_arm=num_arm) 

    def reset(self, 
              num_arm: int=None, 
              ) -> None: 
        if num_arm is not None: self.K = num_arm 

        self.reward_1 = np.ones((self.K, )) 
        self.reward_0 = np.ones((self.K, ))

    def take_action(self) -> int:
        return np.argmax(
            np.random.beta(self.reward_1, self.reward_0)
        )
    
    def update(self, 
               action: int, 
               reward: float,
               ):
        self.counts[action] += 1 
        self.reward_1[action] += reward 
        self.reward_0[action] += (1 - reward) 
        





