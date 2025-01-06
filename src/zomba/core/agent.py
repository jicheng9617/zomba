import numpy as np
from typing import Union


class MABAgent: 
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
    def round(self): 
        return self.t 
    
    @property
    def total_counts(self): 
        return np.sum(self.counts)
    
    @property
    def reward_history(self): 
        return np.array(self._reward_list)
    
    @property
    def action_history(self): 
        return np.array(self._action_list)
    
    def reset(self, 
              num_arm: int = None, 
              ) -> None:
        
        if num_arm is not None: self.K = num_arm
        assert self.K is not None, "Please assign the number of arms!"
        self.t = 0
        self.counts = np.zeros(self.K)
        self.mean_reward = np.zeros((self.K,))
        self._reward_list = []
        self._action_list = []

    def take_action(self, 
                    ) -> int: 
        return np.random.randint(0, self.K)

    def update(self, 
               info: Union[int, float]
               ) -> None: 
        """
        Update parameters in the algorithm.

        Parameters
        ----------
        info : Union[int, float]
            contains variables of action and reward
        """
        action, reward = info
        self.t += 1 
        self.mean_reward[action] = (self.mean_reward[action] * self.counts[action] + reward) / (self.counts[action] + 1)
        self.counts[action] += 1
        self._reward_list.append(reward)
        self._action_list.append(action)
        
        
class moMABAgent(MABAgent): 
    def __init__(self, 
                 num_arm = None, 
                 num_obj = None, 
                 ) -> None:
        super().__init__(num_arm)
        self.m = num_obj
        
    @property
    def num_obj(self): 
        return self.m
    
    def reset(self, 
              num_arm = None, 
              num_obj: int = None, 
              ) -> None:
        super().reset(num_arm)
        if num_obj is not None: self.m = num_obj
        self.mean_reward = np.zeros((self.K, self.m))
     
        
class contextMABAgent(MABAgent): 
    def __init__(self, 
                 num_dim: int, 
                 num_arm: int = None,
                 lamda:float=1.,
                 delta:float=.05, 
                 ) -> None:
        super().__init__(num_arm)
        self.d = num_dim 
        self.lamda = lamda 
        self.delta = delta 
        
    @property 
    def num_dim(self): 
        return self.d  
    
    @property 
    def context_his(self): 
        return np.vstack(self.X_list) 
    
    def reset(self, 
              num_arm: int = None, 
              num_dim: int = None, 
              ) -> None:
        super().reset(num_arm)
        if num_dim is not None: self.d = num_dim
        assert self.d is not None, "Please define dimension of arms' context!"
        self.X_list = [] 
        
    def take_action(self, 
                    context: np.ndarray) -> int:
        return super().take_action() 
    
    def update(self, 
               info: Union[int, float, np.ndarray]
               ) -> None:
        """
        Update parameters of the algorithm.

        Parameters
        ----------
        info : Union[int, float, np.ndarray]
            information containing the last chosen action, observed reward, and the context for that arm
        """
        action, reward, context = info
        super().update((action, reward))
        self.X_list.append(context)
        
    
        