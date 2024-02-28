import numpy as np 
import os 
import pickle

from zomba.multiObjective.utils import (par_non_dominated_sorting, 
                                        ar_dominance, 
                                        pc_non_dominated_sorting)
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
        
        assert self.K is not None, "Please assign number of arms!"
        assert self.d is not None, "Please define dimension of arms' context!" 
        assert self.m is not None, "Please set number of objectives!"

        self._sample_thetas(seed=seed)
        self._sample_context()
        # self._expected_reward()
        # self._optimal_arms()
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

    def _sample_thetas(self, seed) -> None: 
        """
        Generate the unknown parameters from unit sphere
        """
        self.thetas = np.zeros(shape=(self.m,self.d))
        for j in range(self.m): 
            np.random.seed(seed)
            unitVec = np.random.normal(size=self.d)
            unitVec /= np.linalg.norm(unitVec)
            np.random.seed(seed)
            self.thetas[j] = np.random.uniform() ** (1 / self.d) * unitVec

    def _eval_optimal(self):
        self.expected_rewards = self.A @ self.thetas.T 
        # round the expected rewards if the priority type is 'MPL-PC'
        if self.obj_type == 'MPL-PC': self.expected_rewards = self.expected_rewards.round(decimals=2)
        # evaluate the optimal indexs
        match self.obj_type:
            case 'Pareto': 
                self.opt_arm = par_non_dominated_sorting(self.expected_rewards)
            case 'Lexicographic': 
                self.opt_arm = None
            case 'MPL-PC': 
                tmp_y = [self.expected_rewards[:,ind] for ind in self.priority]
                self.opt_arm = pc_non_dominated_sorting(tmp_y)
            case 'MPL-PL': 
                tmp_y = [self.expected_rewards[:,ind] for ind in self.priority]

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
        arm_y = np.matmul(arm, self.th.T)
        psg = 0
        for j in range(len(self.opt_ind)): 
            opt_y = self.opt_y[j]
            if par_dominance(opt_y, arm_y): 
                tmp = np.min(opt_y-arm_y)
                if tmp > psg: psg = tmp
        return psg

    def _eval_regret(self) -> None: 
        """
        Evaluate the regret for all the arms.
        """
        self.reg = np.vstack([self._eval_regret_arm(self.A[i]) for i in range(self.get_num_arm)])

    def regret(self, arm: np.ndarray) -> float: 
        """
        Return the regret for the chosen arm.

        Parameters
        ----------
        arm : np.ndarray
            arm context or index

        Returns
        -------
        float (or np.ndarray in pc and pl)
            the regret value(s)
        """
        if not isinstance(arm, int): 
            ind = np.where(np.isclose(self.A, arm).all(axis=1))
        else: 
            ind = arm
        return self.reg[ind]

    def print_info(self): 
        """
        Print the information of the environment. 
        """
        print(
            {'#objective': self.m, 
             '#dimension': self.d, 
             '#arms': self.A.shape[0], 
             '#optimal arms': len(self.opt_ind), 
             'Regret for each arm': self.reg
             }
        )