import numpy as np


class soEnvironment: 
    def __init__(self, 
                 num_dim: int, 
                 ) -> None:
        pass



class moStochasticEnvironment: 
    def __init__(
        self, 
        num_obj:int, 
        num_dim:int, 
        num_arm:int, 
        noise:any= np.random.normal, 
        R:int= 1
    ) -> None:
        """
        Environment for multi-objective stochastic contextual bandits

        Parameters
        ----------
        num_obj : int
            number of objectives 
        num_dim : int
            number of dimension of the arms' context
        num_arm : int
            number of arms
        noise : any, optional
            R-sub-Gaussian noise for reward, by default normal distribution
        R : int, optional
            variance proxy in subGaussian, by default 1
        """
        self.m = num_obj
        self.d = num_dim
        self.K = num_arm
        self.noise = noise
        self.R = R

    @property
    def get_num_obj(self) -> int:
        return self.m
    
    @property
    def get_num_arm(self) -> int: 
        return self.K
    
    @property
    def get_num_dim(self) -> int: 
        return self.d
    
    @property
    def get_arms(self) -> np.ndarray: 
        return self.A
    
    def observe_context(self) -> np.ndarray: 
        """
        Output the arms' context at round $t$
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_reward(self, arm:int) -> float: 
        """
        Get the reward for the arm

        Parameters
        ----------
        arm : int
            index of selected arm

        Returns
        -------
        float
            reward value
        """
        return self.expected_reward(arm) + self.noise(0,self.R,(self.m,))
    
    def expected_reward(self, arm:int) -> np.ndarray: 
        """
        Method to evaluate the unknown reward function
        """
        raise NotImplementedError("Subclasses should implement this method.")
