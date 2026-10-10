# src/zomba/multiObjective/simulators/cluster.py

import numpy as np 

from zomba.multiObjective.simulators import moContextMABSimulator 
from zomba.multiObjective.utils import (par_non_dominated_sorting, 
                                        par_suboptimal_gap,
)


class linearCluster(moContextMABSimulator):
    def __init__(self, 
                 num_arm: int, 
                 num_dim: int, 
                 num_obj: int, 
                 num_user: int, 
                 num_cluster: int, 
                 arm_context = None, 
                 obj_preference = 'Pareto', 
                 sclarization_type='weighted_sum', 
                 opt_type = 'max', 
                 vary_context = True, 
                 noise_var = 0.1):
        super().__init__(num_arm, num_dim, num_obj, arm_context, obj_preference, sclarization_type, opt_type, vary_context, noise_var)
        self.num_user = num_user
        self.num_cluster = num_cluster 

    def reset(self, 
              num_arm = None, 
              num_dim = None, 
              num_obj = None, 
              num_user = None, 
              num_cluster = None, 
              noise_var = None, 
              seed = None, 
              verbose = False):
        super().reset(num_arm, num_dim, num_obj, noise_var, seed, verbose)
        if num_user is not None: self.num_user = num_user 
        if num_cluster is not None: self.num_cluster = num_cluster
        # initialization 
        self._sample_thetas(seed=seed) 
        # assign user-cluster pair
        self.user_cluster = np.random.choice(self.num_cluster, size=self.num_user)

    def _eval_expected_reward(self, arm):
        thetas = self.thetas[self.current_cluster]
        return arm @ thetas.T

    def _sample_thetas(self, seed): 
        np.random.seed(seed)
        self.thetas = []
        for _ in range(self.num_cluster):
            thetas = np.zeros(shape=(self.m,self.d))
            for j in range(self.m): 
                unitVec = np.random.normal(size=self.d)
                unitVec /= np.linalg.norm(unitVec)
                thetas[j] = np.random.uniform() ** (1 / self.d) * unitVec
            self.thetas.append(thetas)

    def observe_context(self):
        # randomly select an user
        self.current_user = np.random.randint(self.num_user)
        # the cluster index for the user
        self.current_cluster = self.user_cluster[self.current_user]
        # sample context for K items
        if self.vary_context: 
            self._sample_context()
        # return the user index and the context information
        return self.current_user, self.arm_context
    
    def _eval_optimal(self, **kwargs):
        """
        Evaluate the set of Pareto-optimal arms for the current user.
        """
        # Compute the expected reward matrix: shape (K, m)
        self.expected_rewards = self._reward_func(self.A)  # A: (K, d), output: (K, m)
        
        # Perform Pareto non-dominated sorting
        self.opt_arm = par_non_dominated_sorting(self.expected_rewards)

    def _eval_regret_arm(self, arm, *args, **kwargs):
        """
        Compute the Pareto regret of a selected arm for the current user.

        Parameters
        ----------
        arm : int
            Index of the selected arm

        Returns
        -------
        float
            Pareto regret value
        """
        # Get expected reward of the arm (shape: m,)
        arm_y = self._reward_func(arm)

        # Compare to the set of Pareto-optimal arms
        return par_suboptimal_gap(arm_y, self.expected_rewards[self.opt_arm])
    

class nonLinearCluster(moContextMABSimulator):
    def __init__(self, 
                 num_arm: int, 
                 num_dim: int, 
                 num_obj: int, 
                 num_user: int, 
                 num_cluster: int, 
                 arm_context = None, 
                 obj_preference = 'Pareto', 
                 sclarization_type='weighted_sum', 
                 opt_type = 'max', 
                 vary_context = True, 
                 noise_var = 0.1,
                 reward_type: str = 'quadratic'):  # 'quadratic' or 'cosine'
        super().__init__(num_arm, num_dim, num_obj, arm_context, obj_preference, sclarization_type, opt_type, vary_context, noise_var)
        self.num_user = num_user
        self.num_cluster = num_cluster
        self.reward_type = reward_type.lower()
        assert self.reward_type in ['quadratic', 'cosine'], "reward_type must be 'quadratic' or 'cosine'"

    def reset(self, 
              num_arm = None, 
              num_dim = None, 
              num_obj = None, 
              num_user = None, 
              num_cluster = None, 
              noise_var = None, 
              seed = None, 
              verbose = False):
        super().reset(num_arm, num_dim, num_obj, noise_var, seed, verbose)
        if num_user is not None: self.num_user = num_user 
        if num_cluster is not None: self.num_cluster = num_cluster
        self._sample_parameters(seed=seed)
        self.user_cluster = np.random.choice(self.num_cluster, size=self.num_user)

    def _sample_parameters(self, seed):
        np.random.seed(seed)
        self.parameters = []

        for _ in range(self.num_cluster):
            if self.reward_type == 'quadratic':
                A = np.zeros((self.m, self.d, self.d))
                for j in range(self.m):
                    M = np.random.randn(self.d, self.d)
                    A[j] = 2* M / np.linalg.norm(M, ord='fro')
                self.parameters.append(A)

            elif self.reward_type == 'cosine':
                theta = np.zeros((self.m, self.d))
                for j in range(self.m):
                    v = np.random.normal(size=self.d)
                    v /= np.linalg.norm(v)
                    theta[j] = v
                self.parameters.append(theta)

    def observe_context(self):
        self.current_user = np.random.randint(self.num_user)
        self.current_cluster = self.user_cluster[self.current_user]
        if self.vary_context:
            self._sample_context()
        return self.current_user, self.arm_context

    def _eval_expected_reward(self, arm: np.ndarray) -> np.ndarray:
        cluster_id = self.current_cluster

        if self.reward_type == 'quadratic':
            A = self.parameters[cluster_id]  # shape (m, d, d)
            return np.array([arm @ A[i].T @ A[i] @ arm for i in range(self.m)])

        elif self.reward_type == 'cosine':
            theta = self.parameters[cluster_id]  # shape (m, d)
            return np.array([np.cos(3 * arm @ theta[i]) for i in range(self.m)])

    def _eval_optimal(self, **kwargs):
        self.expected_rewards = self._reward_func(self.A)
        self.opt_arm = par_non_dominated_sorting(self.expected_rewards)

    def _eval_regret_arm(self, arm, *args, **kwargs):
        arm_y = self._reward_func(arm)
        return par_suboptimal_gap(arm_y, self.expected_rewards[self.opt_arm])
