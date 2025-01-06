import numpy as np 

from zomba.core import moMABAgent
from zomba.multiObjective.utils import prior_free_lexi_filter, max_filter


class PF_LEX(moMABAgent):
    def __init__(self, 
                 epsilon: float, 
                 num_arm=None, 
                 num_obj=None,
                 ):
        super().__init__(num_arm, num_obj)
        self.epsilon = epsilon
    
    def take_action(self, 
                    explore_scale: float = 0.01, 
                    confidence_level: float = 0.01,
                    ):
        
        if self.total_counts < self.num_arm: 
            return np.random.choice(np.where(self.counts<1)[0])
        else: 
            alpha = explore_scale * np.sqrt(2 * np.log(self.K * self.m * self.counts ** 0.5 / confidence_level) + 1)
            uncertainty = alpha * np.sqrt(1 + self.counts) / self.counts
            if np.max(uncertainty) > self.epsilon: 
                return np.argmax(uncertainty) 
            else:
                ucb, lcb = self.mean_reward+uncertainty.reshape(-1, 1), self.mean_reward-uncertainty.reshape(-1, 1)
                sub_opt_ind = prior_free_lexi_filter(ucb, lcb)
                return np.random.choice(sub_opt_ind)


class DK_TSLB(moMABAgent): 
    def __init__(self, 
                 num_arm=None, 
                 num_obj=None,
                 mu0: list | np.ndarray = None, 
                 sigma0: float = 1.0, 
                 sigma: float = 1.0, 
                 budget: int = None
                 ):
        super().__init__(num_arm, num_obj)
        if mu0 is not None: 
            if isinstance(mu0, float):
                self.mu0 = mu0 * np.ones((self.K, self.m)) 
            else:
                self.mu0 = mu0
        else:
            self.mu0 = np.zeros((self.K, self.m))
        self.sigma0 = sigma0 
        self.sigma = sigma # Gaussian noise
        self.budget = budget if budget is not None else 2e4
        
    def take_action(self, 
                    delta: float = 0.01, 
                    lamda: float = 1.0,
                    conf_scale: float = 0.1,
                    ):
        sigma2 = np.square(self.sigma)
        sigma02 = np.square(self.sigma0)
        post_var = 1.0 / (1.0 / sigma02 + self.counts / sigma2)
        post_mean = np.zeros((self.K, self.m))
        for i in range(self.m): 
            post_mean[:, i] = post_var * (self.mu0[:, i] / sigma02 + (self.mean_reward[:,i]*self.counts) / sigma2)

        # posterior UCBs
        conf_term = np.sqrt(2 * np.log(self.K * self.m / delta) * post_var)
        
        # filter the sub-optimal arms based on AAE (active arm elimination)
        opt_ind = [np.arange(self.K)]
        for i in range(self.m): 
            opt_ind.append(
                max_filter(
                    opt_ind[i], 
                    post_mean[:, i], 
                    2.+np.sum([4*lamda**j for j in range(i+1)]),
                    conf_scale * conf_term
                           )
            )
        for i in reversed(range(len(opt_ind))): 
            if len(opt_ind[i]) > 0: 
                return np.random.choice(opt_ind[i])