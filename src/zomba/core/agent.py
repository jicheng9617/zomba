import numpy as np


class Agent(): 

    def __init__(self) -> None:
        pass



class soMABAgent: 
    def __init__(self, 
                 num_arm: int=None, 
                 ) -> None:
        """_summary_

        Parameters
        ----------
        num_arm : int, optional
            _description_, by default None
        """
        if num_arm is not None: self.K = num_arm 

    @property 
    def num_arm(self): 
        return self.K
    
    @property
    def total_counts(self): 
        return np.sum(self.counts)
    
    @property
    def observed_rewards(self): 
        return self.reward_his
    
    @property 
    def past_actions(self): 
        return self.action_his 
    
    def reset(self, 
              num_arm: int = None, 
              init_estimates: float=1.0, 
              ) -> None:
        
        if num_arm is not None: self.K = num_arm
        self.counts = np.zeros(self.K)
        self.estimates = np.array([init_estimates] * self.K) 
        self.reward_his = [0]
        self.action_his = [0]

    def take_action(self, 
                    ) -> int: 
        return np.random.randint(0, self.K)

    def update(self, 
               action: int, 
               reward: float, 
               ): 
        
        self.counts[action] += 1
        self.estimates[action] += 1. / (self.counts[action] + 1) * (reward - self.estimates[action])
        self.reward_his.append(reward)
        self.action_his.append(action)