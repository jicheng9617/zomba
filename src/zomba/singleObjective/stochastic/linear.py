import numpy as np 

from typing import Union

from zomba.core import contextMABAgent

class SLBUCB(contextMABAgent): 
    
    def __init__(self, 
                 num_dim: int, 
                 num_arm: int = None, 
                 lamda: float = 1, 
                 delta: float = 0.05, 
                 rarely_switch: bool = False, 
                 alpha: float = 1.,
                 ) -> None:
        super().__init__(num_dim, num_arm, lamda, delta)
        self.rarely_switch = rarely_switch
        self.alpha = alpha
        
    def reset(self, 
              num_arm: int = None, 
              num_dim: int = None, 
              ) -> None:
        super().reset(num_arm=num_arm, num_dim=num_dim)
        
        self.V = self.lamda * np.eye(self.d) 
        if self.rarely_switch: self.V_tau = np.copy(self.V)
        self.V_inv = 1 / self.lamda * np.eye(self.d) 
        self.theta = np.zeros((self.d,)) 
    
    def take_action(self, 
                    context: np.ndarray, 
                    alpha: float = None, 
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
        if alpha is not None: 
            self.alpha = alpha
        # else:
            # self.alpha = np.sqrt(self.d * np.log((1 + self.t/self.lamda) / self.delta)) + np.sqrt(self.lamda)
        X = np.atleast_2d(context)
        ucb = X @ self.theta.T + self.alpha * self._eval_uncertainty(X)
        return np.argmax(ucb).item()
        
    def update(self, 
               info: Union[int, float, np.ndarray]
               ) -> None:
        """
        Update parameters 

        Parameters
        ----------
        info : Union[int, float, np.ndarray]
            information containing last action, observed reward, and context for that action
        """
        action, reward, context = info
        super().update((action, reward, context))
        self.V += np.outer(context, context) 
        # rarely switching OFUL algorithm 
        if self.rarely_switch: 
            if np.linalg.det(self.V) > (1 + 1.) * np.linalg.det(self.V_tau):
                self.V_tau = np.copy(self.V)
                self.V_inv = np.linalg.pinv(self.V) 
        else: 
            self.V_inv = np.linalg.pinv(self.V)
        self.theta = self.V_inv @ self.context_his.T @ self.reward_his.T
    
    def _eval_uncertainty(self, context: np.ndarray) -> np.ndarray: 
        mahanalobis_dis = [np.sqrt(context[i] @ self.V_inv @ context[i].T) 
                           for i in range(context.shape[0])]
        return np.array(mahanalobis_dis)
        


class ucb_gpt(SLBUCB): 
    
    def update(self, info) -> None:
        super().update(info) 
        self.alpha = np.sqrt(np.log(self.round + 1) / (self.reward_his.size + self.round + 1 + 1e-8))


    
if __name__ == "__main__": 
    print() 
    from zomba.singleObjective.simulator import SLBSimulator 
    env = SLBSimulator(num_arm=5, num_dim=3) 
    env.reset(verbose=1)
    alg = SLBUCB(num_arm=5, num_dim=3, rarely_switch=1) 
    alg.reset()
    cum_reg = 0 
    for _ in range(5000): 
        X = env.observe_context() 
        a_t = alg.take_action(X)
        r_t = env.get_reward(arm=a_t) 
        cum_reg += env.get_regret(arm=a_t)
        alg.update(info=(a_t, r_t, X[a_t])) 
        
    print(cum_reg)
    print(alg.counts)