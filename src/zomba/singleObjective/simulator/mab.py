import numpy as np 

from zomba.core import MABEnv

class MABSimulator(MABEnv): 
    def __init__(self, 
                 num_arm: int=None,
                 ) -> None:
        
        super(MABSimulator, self).__init__(num_arm=num_arm)
    
    @property
    def expected_rewards(self): 
        return self.p
    
    @property
    def optimal_arm(self): 
        return self.opt_ind 
    
    @property 
    def optimal_reward(self): 
        return self.p[self.optimal_arm]

    def reset(self, 
              num_arm: int=None, 
              seed: int = None, 
              ) -> None: 
        
        if num_arm is not None: self.K = num_arm
        np.random.seed(seed)
        self.p = np.random.uniform(size=self.K) 
        self._eval_optimal()

    def _eval_optimal(self):
        self.opt_ind = np.argmax(self.p) 
        
    def get_regret(self, arm: int) -> float: 
        if isinstance(arm, np.ndarray): 
            return np.array(
                [self._eval_regret_arm(a_i) for a_i in arm]
            )
        else: 
            return self._eval_regret_arm(arm) 
    
    def _eval_regret_arm(self, arm: int) -> float: 
        return self.optimal_reward - self.p[arm] 
    
    def get_reward(self, arm: int) -> float:
        return super().get_reward(arm)



class MABSimulator_Bernoulli(MABSimulator):
    def __init__(self, 
                 num_arm: int=None, 
                 ) -> None:
        
        super(MABSimulator_Bernoulli, self).__init__(num_arm=num_arm)

    def get_reward(self, arm: int):
        return 1 if np.random.rand() < self.p[arm] else 0
        
        

class mabSimulator_Gaussian(MABSimulator): 
    def __init__(self, num_arm: int = None) -> None:
        #TODO
        pass
    

    

