from typing import Union
import numpy as np 

from zomba.core import contextMABAgent


class UCB_TS(contextMABAgent): 
    
    def __init__(self, 
                 num_dim: int, 
                 num_arm: int = None, 
                 lamda: float = 1, 
                 delta: float = 0.05, 
                 rarely_switch: bool = False, 
                 action_type: str = 'UCB', 
                 alpha: float = 1.,
                 ) -> None:
        super().__init__(num_dim, num_arm, lamda, delta)
        self.rarely_switch = rarely_switch
        self.action_type = action_type
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
        if self.t == 5000: 
            print()
        if self.total_counts < self.K: 
            return np.random.choice(np.where(self.counts<1)[0])
        if alpha is not None: 
            self.alpha = alpha
        # else:
            # self.alpha = np.sqrt(self.d * np.log((1 + self.t/self.lamda) / self.delta)) + np.sqrt(self.lamda)
        X = np.atleast_2d(context)
        predict_mean = X @ self.theta
        if self.action_type.lower() == 'ucb': 
            metric = predict_mean + self.alpha * self._eval_uncertainty(X)
        elif self.action_type.lower() == 'ts': 
            nu = np.sqrt(24. / self.alpha * self.d * np.log(1/self.delta))
            theta_tilde = np.random.multivariate_normal(mean=predict_mean, cov=nu**2 * self.V_inv)
            metric = X @ theta_tilde
        return np.argmax(metric).item()
        
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
        self.theta = self.V_inv @ self.context_history.T @ self.reward_history
    
    def _eval_uncertainty(self, context: np.ndarray) -> np.ndarray: 
        mahanalobis_dis = [np.sqrt(context[i] @ self.V_inv @ context[i].T) 
                           for i in range(context.shape[0])]
        return np.array(mahanalobis_dis)
        


    
if __name__ == "__main__": 
    print() 