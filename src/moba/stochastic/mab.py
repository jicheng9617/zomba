import numpy as np 


class multiArmedBandit: 
    def __init__(self) -> None:
        pass 

    @property 
    def num_arm(self): 
        return self.K



class epsilonGreedy(multiArmedBandit):
    
    def __init__(self, 
                 num_arm: int, 
                 ) -> None:
        
        super(epsilonGreedy, self).__init__()
        self.K = num_arm 

    def reset(self, 
              epsilon: float=0.01, 
              init_prob: float=1.0,
              ) -> None:
        
        self.counts = np.zeros(self.K)
        self.epsilon = epsilon
        self.reward_estimates = np.array([init_prob] * self.K)

    def take_action(self, epsilon: float=None):

        if epsilon is not None: self.epsilon = epsilon 

        if np.random.random() < self.epsilon:
            self.action = np.random.randint(0, self.K)  # random selection
        else:
            self.action = np.argmax(self.reward_estimates)  # greedy selection
            
        return self.action

    def update(self, reward): 
        
        self.reward_estimates[self.action] += \
            1. / (self.counts[self.action] + 1) * (reward - self.reward_estimates[self.action])
