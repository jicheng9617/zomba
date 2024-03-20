import numpy as np
from zomba.core import contextMABEnv 

# load dataset 
data = np.loadtxt("./data/dataset.txt")

class webRecEnv(contextMABEnv): 
    def __init__(self, data: np.ndarray = data) -> None:
        arms, rewards, contexts = data[:,0], data[:,1], data[:,2:]
        self.arms = arms.astype(int)
        self.rewards = rewards.astype(float)
        contexts = contexts.astype(float)
        num_arm = len(np.unique(arms))
        self.num_events = len(contexts)
        num_dim = int(len(contexts[0])/num_arm)
        contexts = contexts.reshape(self.num_events, num_arm, num_dim)
        super().__init__(num_arm, num_dim, contexts)

    def reset(self, 
              verbose: bool = False) -> None: 
        self.event_id = 0 
        if verbose: 
            print(f"# events: {self.num_events},\n# arms: {self.num_arm},\n# dimension: {self.num_dim}.")

    def get_reward(self, arm: int) -> float:
        return self.rewards[self.event_id] if arm == self.arms[self.event_id] else None
    
    def observe_context(self) -> np.ndarray:
        return self.arm_context[self.event_id]

    def has_next(self) -> bool: 
        """
        Whether all the events have been iterated.

        Returns
        -------
        bool
            continue or not
        """
        return self.event_id < (self.num_events-1)
    
    def update_id(self) -> None: 
        self.event_id += 1