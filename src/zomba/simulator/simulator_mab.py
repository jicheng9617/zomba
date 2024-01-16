import numpy as np 

from zomba.core.environments import soMABEnvironment

class soSimulatorMAB(soMABEnvironment): 
    def __init__(self, 
                 num_arm: int=None,
                 ) -> None:
        
        super(soSimulatorMAB, self).__init__(num_arm=num_arm)
    
    @property
    def expected_rewards(self): 
        return self.p
    
    @property
    def optimal_arm(self): 
        return self.opt_ind

    def reset(self, 
              num_arm: int=None, 
              ) -> None: 
        
        if num_arm is not None: self.K = num_arm
        self.p = np.random.uniform(size=self.K) 
        self._eval_optimal()

    def _eval_optimal(self):

        self.opt_ind = np.argmax(self.probs) 
        
    def get_regret(self, arm: int) -> float: 

        if isinstance(arm, int): 
            return self._eval_regret_arm(arm) 
        elif isinstance(arm, np.ndarray): 
            return np.array(
                [self._eval_regret_arm(a_i) for a_i in arm]
            )
    
    def _eval_regret_arm(self, arm: int) -> float: 

        raise self.optimal_reward - self.probs[arm] 
    
    def get_reward(self, arm: int) -> float:
        return super().get_reward(arm)



class soSimulatorBernoulliMAB(soSimulatorMAB):
    def __init__(self, 
                 num_arm: int=None, 
                 ) -> None:
        
        super(soSimulatorBernoulliMAB, self).__init__(num_arm=num_arm)

    def get_reward(self, action):
        
        return 1 if np.random.rand() < self.p[action] else 0
        
        

class soSimulatorMABGaussian(soSimulatorMAB): 
    def __init__(self, num_arm: int = None) -> None:
        #TODO
        pass
    

    

