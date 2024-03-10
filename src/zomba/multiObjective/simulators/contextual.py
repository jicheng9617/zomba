import numpy as np 
import os 
import pickle

from zomba.multiObjective.utils import (par_non_dominated_sorting, 
                                        par_suboptimal_gap,
                                        pc_non_dominated_sorting,
                                        pc_suboptimal_gap)
from zomba.singleObjective.simulator import contextMABSimulator 


class moContextMABSimulator(contextMABSimulator):
    def __init__(self, 
                 num_arm: int = None, 
                 num_dim: int = None, 
                 num_obj: int = None, 
                 arm_context: np.ndarray = None,
                 ) -> None:
        super().__init__(num_arm, num_dim, arm_context)
        self.m = num_obj 

    @property 
    def num_obj(self): 
        return self.m 


class moSLBSimulator(moContextMABSimulator):
    def __init__(self, 
                 num_arm: int = None, 
                 num_dim: int = None, 
                 num_obj: int = None, 
                 arm_context: np.ndarray = None, 
                 obj_preference: str = 'Pareto', 
                 vary_context: bool = False, 
                 noise_var: float = 0.1, 
                 priority: list = None, 
                 ) -> None:
        """
        Simulator for multi-objective stochastic linear bandits

        Parameters
        ----------
        num_arm : int, optional
            # arms, by default None
        num_dim : int, optional
            number of dimension, by default None
        num_obj : int, optional
            number of objectives, by default None
        arm_context : np.ndarray, optional
            arms' context, by default None
        obj_preference : str, optional
            preference type among objectives (choose from 'Pareto', 'Lexicographic', 'MPL-PC', 'MPL-PL'), by default 'Pareto'
        vary_context : bool, optional
            whether to regenerate arms' context at each round, by default False
        noise_var : float, optional
            variance proxy of noise, by default 0.1
        """
        super().__init__(num_arm, num_dim, num_obj, arm_context)
        self.m = num_obj
        self.obj_type = obj_preference
        self.vary_context = vary_context 
        self.R = noise_var 
        self.priority = priority

    def reset(self, 
              num_arm: int = None, 
              num_dim: int = None, 
              num_obj: int = None, 
              noise_var : float = None, 
              seed: int = None, 
              verbose: bool = False,
              ) -> None: 
        """
        Initialize the environment and sample the unknown parameters randomly. 
        """
        if num_arm is not None: self.K = num_arm 
        if num_dim is not None: self.d = num_dim 
        if num_obj is not None: self.m = num_obj 
        if noise_var is not None: self.R = noise_var 
        # check the setting
        assert self.K is not None, "Please assign number of arms!"
        assert self.d is not None, "Please define dimension of arms' context!" 
        assert self.m is not None, "Please set number of objectives!" 
        if (self.obj_type.lower() in 'mpl-pc'+'mpl-pl') and self.priority is None: 
            raise NotImplementedError('Please assign the preference relationship between objectives!')
        # initialize
        self._sample_thetas(seed=seed)
        self._sample_context()
        self._eval_optimal()
        if verbose: self.print_info()

    def observe_context(self) -> np.ndarray:
        return super().observe_context()
    
    def get_regret(self, arm: any) -> np.ndarray:
        return super().get_regret(arm)
    
    def get_reward(self, arm: int) -> np.ndarray:
        if isinstance(arm, np.ndarray):
            return self.expected_rewards[arm] + self._noise(size=len(arm))
        else: 
            return self.expected_rewards[arm] + self._noise(size=1).squeeze()

    def _sample_thetas(self, seed: int) -> None: 
        """
        Generate the unknown parameters from unit sphere
        """
        np.random.seed(seed) # seed 
        self.thetas = np.zeros(shape=(self.m,self.d))
        for j in range(self.m): 
            unitVec = np.random.normal(size=self.d)
            unitVec /= np.linalg.norm(unitVec)
            self.thetas[j] = np.random.uniform() ** (1 / self.d) * unitVec

    def _eval_optimal(self):
        self.expected_rewards = self.A @ self.thetas.T 
        # round the expected rewards if the priority type is 'MPL-PC'
        if self.obj_type.lower() == 'mpl-pc': 
            self.expected_rewards = self.expected_rewards.round(decimals=1)
        # evaluate the optimal indexs
        match self.obj_type.lower():
            case 'pareto': 
                self.opt_arm = par_non_dominated_sorting(self.expected_rewards)
            case 'lexicographic': 
                self.opt_arm = None
            case 'mpl-pc': 
                tmp_y = [self.expected_rewards[:,ind] for ind in self.priority]
                self.opt_arm = pc_non_dominated_sorting(tmp_y)
            case 'mpl-pl': 
                opt_arm = [np.arange(self.num_arm)]
                # evaluate the optimal arms for each priority level 
                for i in range(len(self.priority)): 
                    opt_arm.append(
                        opt_arm[-1][par_non_dominated_sorting(self.expected_rewards[opt_arm[-1]][:, self.priority[i]])]
                    )
                self.opt_arm = opt_arm[1:]

    def _noise(self, size: int): 
        return np.random.normal(loc=0.0, scale=self.R, size=(size, self.m))

    def _eval_regret_arm(self, arm: int) -> float: 
        """
        Evaluate the regret gap for an input arm. 

        Parameters
        ----------
        arm : arm
            arm's index

        Returns
        -------
        float
            regret gap 
        """
        arm_y = self.expected_rewards[arm]

        match self.obj_type.lower(): 
            case 'pareto': 
                # evaluate the Pareto suboptimal gap for the arm 
                gap = par_suboptimal_gap(arm_y, self.expected_rewards[self.opt_arm])
            case 'lexicographic': 
                pass 
            case 'mpl-pc': 
                arm_y = [arm_y[ind] for ind in self.priority]
                optimal_y = [self.expected_rewards[self.opt_arm][:, ind] for ind in self.priority]
                gap = pc_suboptimal_gap(arm_y, optimal_y)
            case 'mpl-pl': 
                # gap for MPL-PL order with form [float, ..., float]
                l = len(self.priority)
                gap = np.zeros((l, ))
                for i in range(l): 
                    delta_x = par_suboptimal_gap(arm_y[self.priority[i]], self.expected_rewards[self.opt_arm[i]][:, self.priority[i]])
                    if delta_x > 0: 
                        gap[i] = delta_x 
                        # priority based regret 
                        break 
        return gap

    def print_info(self): 
        """
        Print the information of the environment. 
        """
        print(
            f"# objective: {self.m}, \n# dimension: {self.d},\n# arms: {self.num_arm},\n# optimal arms: {self.opt_arm},\nRegret for each arm: {[self._eval_regret_arm(i) for i in range(self.num_arm)]},\nExpected rewards for each arm: \n{self.expected_rewards}"
        )
        print(f"Arms' context: {self.A}")



if __name__ == "__main__": 
    print() 
    import numpy as np 
    from zomba.multiObjective.simulators import moSLBSimulator
    K = 100
    d = 8 
    m = 4
    priority = [[0,1], [2,3]]
    env = moSLBSimulator(K, d, m, obj_preference='MPL-PC', priority=priority)
    env.reset(verbose=1, seed=1234)