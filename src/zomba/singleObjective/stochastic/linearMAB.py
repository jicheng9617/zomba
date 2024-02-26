import numpy as np 

from zomba.core import contextMABAgent

class slbUCB(contextMABAgent): 
    
    def __init__(self, 
                 num_dim: int, 
                 num_arm: int = None, 
                 lamda: float = 1, 
                 delta: float = 0.05, 
                 ) -> None:
        super().__init__(num_dim, num_arm, lamda, delta)
    
    def _eval_uncertainty(self, context): 
        gamma_t = np.sqrt(self.d * np.log((1 + self.t/self.lamda) / self.delta)) + np.sqrt(self.lamda)
        mahanalobis_dis = [np.sqrt(context[i] @ self.V_inv @ context[i].T) 
                           for i in range(context.shape[0])]
        return gamma_t * np.array(mahanalobis_dis)
        
    def reset(self, 
              num_arm: int = None, 
              num_dim: int = None, 
              ) -> None:
        super().reset(num_arm=num_arm, num_dim=num_dim)
        
        self.V = self.lamda * np.eye(self.d) 
        self.V_inv = 1 / self.lamda * np.eye(self.d) 
        self.theta = np.zeros((self.d,)) 
    
    def take_action(self, 
                    context: np.ndarray, 
                    alpha: float = 1., 
                    ) -> int:
        """
        Choose the arm with highest upper confidence bound 

        Parameters
        ----------
        context : np.ndarray
            arms' contexts
        alpha : float, optional
            cofficient for the uncertainty, by default 1.

        Returns
        -------
        int
            index of selected arm
        """
        self.X = context 
        ucb = self.X @ self.theta.T + alpha * self._eval_uncertainty(self.X)
        return np.argmax(ucb).item()
        
    def update(self, 
               action: int, 
               reward: float, 
               context: np.ndarray,
               ):
        super().update(action, reward, context)
        self.V += np.outer(context[action], context[action]) 
        self.V_inv = np.linalg.inv(self.V) 
        self.theta = self.V_inv @ self.context_his.T @ self.reward_his.T
        
    
if __name__ == "__main__": 
    print() 
    from zomba.singleObjective.simulator import slbSimulator 
    env = slbSimulator(num_arm=5, num_dim=3) 
    env.reset(verbose=1)
    alg = slbUCB(num_arm=5, num_dim=3) 
    alg.reset()
    cum_reg = 0 
    for _ in range(5000): 
        X = env.observe_context() 
        a_t = alg.take_action(X)
        r_t = env.get_reward(arm=a_t) 
        cum_reg += env.get_regret(arm=a_t)
        alg.update(action=a_t, reward=r_t, context=X) 
        
    print(cum_reg)
    print(alg.counts)