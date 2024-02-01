import numpy as np 
import os 
import pickle

from zomba.multiObjective.utils import par_non_dominated_sorting, par_dominance

from pprint import pprint


class env_moslb:
    def __init__(
        self, 
        num_obj:int, 
        num_dim:int, 
        noise:any=np.random.normal, 
        R:int=1
    ) -> None:
        """
        Environment for multi-objective stochastic linear bandits

        Parameters
        ----------
        num_obj : int
            number of objectives 
        num_dim : int
            number of dimension of the arms' context
        noise : any, optional
            R-sub-Gaussian noise for reward, by default np.random.normal
        R : int, optional
            parameter in noise, by default 1
        """
        self.m = num_obj
        self.d = num_dim
        self.noise = noise
        self.R = R

    @property
    def num_obj(self) -> int:
        return self.m
    
    @property
    def num_arm(self) -> int: 
        return self.A.shape[0]
    
    @property
    def num_dim(self) -> int: 
        return self.d
    
    @property
    def arm_comtext(self) -> np.ndarray: 
        return self.A

    def _sample_theta(self) -> None: 
        """
        Generate the unknown parameters from unit sphere
        """
        theta = np.zeros(shape=(self.m,self.d))
        for j in range(self.m): 
            temp = np.random.normal(size=self.d)
            temp /= np.linalg.norm(temp) 
            rad = np.random.uniform() ** (1 / self.d) 
            theta[j] = temp * rad
        self.th = theta
    
    def _sample_arms(self, num_arms:int) -> None:
        """
        Generate arms' context randomly within unit sphere.

        Parameters
        ----------
        num_arms : int
            number of arms
        """
        self.A = np.zeros(shape=(num_arms, self.d))
        for i in range(num_arms): 
            point = np.random.normal(size=self.d)
            point /= np.linalg.norm(point)
            rad = np.random.uniform() ** (1 / self.d)
            self.A[i] = rad * point

    def _expected_reward(self) -> None: 
        """
        Evaluate the expected loss for the arms.
        """
        self.y = np.matmul(self.A, self.th.T)

    def _optimal_arms(self) -> None: 
        """
        Assess the non-dominated arms (optimal arms). 
        """
        self.opt_ind = par_non_dominated_sorting(self.y)
        self.opt_arm = self.A[self.opt_ind] 
        self.opt_y = self.y[self.opt_ind]

    def _eval_regret_arm(self, arm: np.ndarray) -> float: 
        """
        Evaluate the Pareto suboptimal gap for an input arm. 

        Parameters
        ----------
        arm : np.ndarray
            contextual information for the arm

        Returns
        -------
        float
            Pareto suboptimal gap
        """
        arm_y = np.matmul(arm, self.th.T)
        psg = 0
        for j in range(len(self.opt_ind)): 
            opt_y = self.opt_y[j]
            if par_dominance(opt_y, arm_y): 
                tmp = np.min(opt_y-arm_y)
                if tmp > psg: psg = tmp
        return psg
    
    def save_env(self, filename:str=None) -> None: 
        """
        Save the environment to the "data" folder

        Parameters
        ----------
        filename : str, optional
            saving path, by default None
        """
        if filename is None: 
            if not os.path.exists('.\\data\\'): os.makedirs('.\\data\\')
            filename = '.\\data\\ParetoOrder_d_' + str(self.get_num_dim) + '.pkl'
        data = {
            'theta': self.th, 
            'arms': self.A
                }
        with open(filename, 'wb') as f: 
            pickle.dump(data, f)

    def load_env(self, filename: str=None, verbose:bool=False) -> None: 
        """
        Load the existing environment for MOSLB with Pareto order

        Parameters
        ----------
        filename : str, optional
            file path, by default None. If default, the file in subfolder 'data' named
            ParetoOrder_d_*.pkl will be loaded. 
        verbose : bool, optional
            whether to print the information, by default False
        """
        if filename is None: 
            filename = '.\\data\\ParetoOrder_d_' + str(self.get_num_dim) + '.pkl'
        with open(filename, 'rb') as f: 
            data = pickle.load(f)
        self.th = data['theta']
        self.A = data['arms']
        # initialize the parameters 
        self._expected_reward()
        self._optimal_arms()
        self._eval_regret()
        if verbose: self.print_info()

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

    def reset(self, num_arms:int, verbose:bool=False) -> None: 
        """
        Initialize the environment and sample the unknown parameters randomly. 

        Parameters
        ----------
        num_arms : int
            number of the arms 
        verbose : bool, optional
            whether to print the information, by default False
        """
        self._sample_theta()
        self._sample_arms(num_arms)
        self._expected_reward()
        self._optimal_arms()
        self._eval_regret()
        if verbose: self.print_info()

    def print_info(self): 
        """
        Print the information of the environment. 
        """
        pprint(
            {'#objective': self.m, 
             '#dimension': self.d, 
             '#arms': self.A.shape[0], 
             '#optimal arms': len(self.opt_ind), 
             'Regret for each arm': self.reg
             }
        )