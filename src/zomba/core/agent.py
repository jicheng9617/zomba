import numpy as np



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
    def round_index(self): 
        return self.t 
    
    @property
    def total_counts(self): 
        return np.sum(self.counts)
    
    @property
    def reward_his(self): 
        return np.array(self.reward_list)
    
    def reset(self, 
              num_arm: int = None, 
              ) -> None:
        
        if num_arm is not None: self.K = num_arm
        assert self.K is not None, "Please assign the number of arms!"
        self.t = 0
        self.counts = np.zeros(self.K)
        self.reward_list = []
        self.action_his = []

    def take_action(self, 
                    ) -> int: 
        return np.random.randint(0, self.K)

    def update(self, 
               action: int, 
               reward: float, 
               ): 
        self.t += 1 
        self.counts[action] += 1
        self.reward_list.append(reward)
        self.action_his.append(action)
        
     
        
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
               action: int, 
               reward: float, 
               context: np.ndarray, 
               ):
        super().update(action, reward)
        self.X_list.append(context[action])
        
    
        