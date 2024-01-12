import numpy as np 

class soSimulatorMAB: 
    def __init__(self, num_arm: int) -> None:
        self.K = num_arm 

    @property
    def num_arm(self): 
        return self.K
        
    def get_regret(self, action: int) -> float: 

        return self._eval_regret_arm(action) 
    
    def _eval_regret_arm(self, arm: int) -> float: 
        raise NotImplementedError("Subclasses should implement this method.")



class soSimulatorBernoulli(soSimulatorMAB):
    
    def __init__(self, 
                 num_arm: int
                 ) -> None:
        super(soSimulatorBernoulli, self).__init__(num_arm=num_arm)

    @property
    def optimal_reward(self): 
        return self.probs[self.optimal_arm] 
    
    @property
    def optimal_arm(self): 
        return np.argmax(self.probs) 

    def reset(self):

        self.probs = np.random.uniform(size=self.K)  

    def get_reward(self, action):
        
        if np.random.rand() < self.probs[action]:
            return 1
        else:
            return 0
        
    def _eval_regret_arm(self, action: int) -> float: 

        return self.optimal_reward - self.probs[action]
    

