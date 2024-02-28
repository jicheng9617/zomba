import numpy as np


class MABEnv: 
    def __init__(self, 
                 num_arm: int = None, 
                 ) -> None:
        self.K = num_arm

    @property
    def num_arm(self): 
        return self.K 
    
    def get_reward(self, 
                   arm: int,
                   ) -> float: 
        """
        Get the reward for the arm

        Parameters
        ----------
        arm : int
            index of the selected arm

        Returns
        -------
        float
            reward value
        """
        raise NotImplementedError("Subclasses should implement this method.") 
    

class contextMABEnv(MABEnv): 
    def __init__(self, 
                 num_arm: int = None, 
                 num_dim: int = None, 
                 arm_context: np.ndarray = None, 
                 ) -> None:
        super().__init__(num_arm) 
        self.d = num_dim 
        self.A = np.array(arm_context) if isinstance(arm_context, list) else arm_context

    @property
    def arm_context(self) -> np.ndarray: 
        return self.A
    
    @property
    def num_dim(self) -> int: 
        return self.d
    
    def observe_context(self) -> np.ndarray: 
        """
        Output the arms' context at round $t$
        """
        raise NotImplementedError("Subclasses should implement this method.")

    