import numpy as np
from typing import Union

from zomba.core import contextMABEnv

class contextMABSimulator(contextMABEnv): 
    def __init__(self, 
                 num_arm: int = None, 
                 num_dim: int = None, 
                 arm_context: np.ndarray = None,
                 ) -> None:
        super().__init__(num_arm, num_dim, arm_context)
    
    @property 
    def optimal_arm(self): 
        return self.opt_arm
    
    def reset(self): 
        raise NotImplementedError("Subclasses should implement this method.") 
        
    def observe_context(self) -> np.ndarray:
        if self.vary_context: 
            self._sample_context()
            self._eval_optimal()
        return self.arm_context
    
    def get_regret(self, arm: Union[int | np.ndarray | list]) -> np.ndarray: 
        """
        Get regrets for arms

        Parameters
        ----------
        arm : any
            arms' index, can be int, list or ndarray

        Returns
        -------
        np.ndarray
            regrets
        """
        if isinstance(arm, list): arm = np.array(arm)
        if isinstance(arm, np.ndarray): 
            return np.array(
                [self._eval_regret_arm(a_i) for a_i in arm]
            )
        else: 
            return self._eval_regret_arm(arm) 
            
    def _sample_context(self):
        """
        Generate arms' context randomly within unit sphere.

        Parameters
        ----------
        num_arm : int
            number of arms
        """
        self.A = np.zeros(shape=(self.K, self.d))
        for i in range(self.K): 
            unitVec = np.random.normal(size=self.d)
            unitVec /= np.linalg.norm(unitVec)
            self.A[i] = np.random.uniform() ** (1 / self.d) * unitVec
    
    def _eval_optimal(self): 
        raise NotImplementedError("Subclasses should implement this method.") 
    
    def _eval_regret_arm(self, arm):
        raise NotImplementedError("Subclasses should implement this method.") 
    
    def _eval_expected_reward(self, arm): 
        raise NotImplementedError("Subclasses should implement this method.") 
    
    def _h(self,
           arm: list | int | np.ndarray): 
        if isinstance(arm, int) or isinstance(arm, np.integer): 
            return self._eval_expected_reward(arm=self.A[arm])
        elif isinstance(arm, list): 
            return np.vstack([self._eval_expected_reward(arm=self.A[i]) for i in arm]).squeeze()
        elif isinstance(arm, np.ndarray): 
            arm = np.atleast_2d(arm) 
            return np.vstack([self._eval_expected_reward(arm=a) for a in arm])
    
    def _print_info(self): 
        """
        Print the information of the environment. 
        """
        print('# dimension: {}'.format(self.d))
        print('# arms: {}'.format(self.num_arm))
        print("Optimal arm: {}".format(self.optimal_arm))
        print('Regret for each arm: {}'.format(
            [round(self._eval_regret_arm(i),4) for i in range(self.num_arm)]
            ))
    
        
        
class SLBSimulator(contextMABSimulator): 
    def __init__(self, 
                 num_arm: int = None, 
                 num_dim: int = None, 
                 arm_context: np.ndarray = None, 
                 vary_context: bool = False, 
                 noise_var: float = 0.1, 
                 ) -> None:
        """
        Simulator for stochastic linear bandits

        Parameters
        ----------
        num_arm : int, optional
            # arms, by default None
        num_dim : int, optional
            # dims of arms' context, by default None
        arm_context : np.ndarray, optional
            arms' context, by default None
        vary_context : bool, optional
            whether to regenerate arms' context at each round, by default False
        noise_var : float, optional
            variance proxy of noise, by default .1
        """
        super().__init__(num_arm, num_dim, arm_context)
        self.vary_context = vary_context 
        self.R = noise_var 
        
    def reset(self,
              num_arm : int = None, 
              num_dim : int = None, 
              noise_var : float = None, 
              seed: int = None, 
              verbose : bool = False, 
              ): 
        """
        Reset the environment 

        Parameters
        ----------
        num_arm : int, optional
            number of arms, by default None
        num_dim : int, optional
            number of dimension, by default None
        noise_var : float, optional
            variance proxy of noise, by default None
        verbose : bool, optional
            whether to print the information, by default False
        """
        if num_arm is not None: self.K = num_arm 
        if num_dim is not None: self.d = num_dim 
        if noise_var is not None: self.R = noise_var 
        
        assert self.K is not None, "Please assign number of arms!"
        assert self.d is not None, "Please define dimension of arms' context!" 
        
        self._sample_theta(seed=seed) 
        self._sample_context()
        self._eval_optimal() 

        if verbose: self._print_info()
    
    def get_reward(self, arm: int) -> np.ndarray:
        if isinstance(arm, np.ndarray):
            return self.expected_rewards[arm] + self._noise(size=len(arm))
        else: 
            return self.expected_rewards[arm] + self._noise(size=None)

    def _sample_theta(self, seed): 
        np.random.seed(seed)
        unitVec = np.random.normal(size=self.d)
        unitVec /= np.linalg.norm(unitVec)
        np.random.seed(seed)
        self.theta = np.random.uniform() ** (1 / self.d) * unitVec
            
    def _eval_optimal(self):
        self.expected_rewards = self.A @ self.theta.T
        self.opt_arm = np.argmax(self.expected_rewards) 
        
    def _eval_regret_arm(self, arm):
        return self.expected_rewards[self.optimal_arm] - self.expected_rewards[arm]
    
    def _noise(self, size: int): 
        return np.random.normal(loc=0.0, scale=self.R, size=size)
    



if __name__ == "__main__": 
    print() 
    env = SLBSimulator(num_arm=100, ) 
    env.reset(num_dim=7, verbose=1, seed=123456) 
    print(env.get_reward(5))