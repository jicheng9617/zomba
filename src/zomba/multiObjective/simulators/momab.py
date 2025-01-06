import numpy as np 

from zomba.core import MABEnv
from zomba.multiObjective.utils import lex_dominated_sorting


class moMABSimulator(MABEnv):
    def __init__(self, 
                 num_arm = None, 
                 num_obj: int = None):
        super().__init__(num_arm)
        self.m = num_obj
        
    @property
    def num_obj(self): 
        return self.m
        
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
              num_arm: int = None,
              num_obj: int = None, 
              seed: int = None, ): 
        
        if num_arm is not None: self.K = num_arm
        if num_obj is not None: self.m = num_obj
        np.random.seed(seed)
        self._generate_distribution()
        self._eval_optimal()
        
    def get_regret(self, arm: int): 
        return (self.optimal_reward - self.p[arm]).tolist()

    def _generate_distribution(self): 
        """
        Generate the distribution of the reward for each arm, by default Gaussian.
        """
        self.p = np.random.uniform(0, 1, size=(self.K, self.m))
        self.var = 0.1
    
    def _eval_optimal(self):
        self.opt_ind = lex_dominated_sorting(self.p)

        
        
class moMABSimulator_Gaussian(moMABSimulator): 
    def __init__(self, num_arm=None, num_obj = None):
        super().__init__(num_arm, num_obj)
        
    def get_reward(self, arm: int):
        
        return np.random.normal(
            loc=self.p[arm], 
            scale=np.sqrt(self.var), 
            size=(self.num_obj, )
        )