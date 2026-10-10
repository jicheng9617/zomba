import numpy as np 

from zomba.legacy.core import MABEnv
from zomba.multiObjective.utils import lex_dominated_sorting, par_non_dominated_sorting, par_suboptimal_gap


class moMABSimulator(MABEnv):
    def __init__(self, 
                 num_arm = None, 
                 num_obj: int = None,
                 optimal_type: str = 'Pareto',
                ):
        super().__init__(num_arm)
        assert num_obj > 1, "number of objectives should be larger than 1!"
        self.m = num_obj
        self.optimal_type = optimal_type
        
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
        match self.optimal_type.lower(): 
            case "pareto": 
                # evaluate the Pareto suboptimal gap for the arm 
                return par_suboptimal_gap(self.p[arm], self.p[self.opt_ind])

    def _generate_distribution(self): 
        """
        Generate the distribution of the reward for each arm, by default Gaussian.
        """
        self.p = np.random.uniform(0, 1, size=(self.K, self.m))
    
    def _eval_optimal(self):
        match self.optimal_type.lower():
            case "lexicographic":
                self.opt_ind = lex_dominated_sorting(self.p)
            
            case "pareto": 
                self.opt_ind = par_non_dominated_sorting(self.p)

        
class moMABSimulator_Gaussian(moMABSimulator): 
    def __init__(self, 
                 num_arm=None, 
                 num_obj = None,
                 optimal_type: str = 'Pareto',
                 ):
        super().__init__(num_arm, num_obj, optimal_type)
        
    def get_reward(self, arm: int):
        
        return np.random.normal(
            loc=self.p[arm], 
            scale=np.sqrt(self.var), 
            size=(self.num_obj, )
        )

    def _generate_distribution(self): 
        """
        Generate the distribution of the reward for each arm, by default Gaussian.
        """
        self.p = np.random.uniform(0, 1, size=(self.K, self.m))
        self.var = 0.1
        
        
class moMABSimulator_Bernoulli(moMABSimulator): 
    def __init__(self, 
                 num_arm = None, 
                 num_obj = None,
                 optimal_type: str = 'Pareto',
                 prob_dist: list | np.ndarray = None, 
                 ):
        super().__init__(num_arm, num_obj, optimal_type)
        self.p = prob_dist

    @property
    def regrets(self): 
        return np.array([self.get_regret(i) for i in np.arange(self.K)])
        
    def get_reward(self, arm: int):
        
        return np.random.binomial(1, p=self.p[arm])
    
    def _generate_distribution(self):
        if self.p is None: 
            super()._generate_distribution()