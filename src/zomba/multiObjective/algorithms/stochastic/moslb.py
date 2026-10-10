import numpy as np 

from zomba.legacy.core import moContextMABAgent
from zomba.multiObjective.utils import par_non_dominated_sorting, max_filter



class moslb(moContextMABAgent):
    def __init__(self,
                 num_arm: int, 
                 num_dim: int,
                 num_obj: int,
                 lamda: float=1.,
                 delta: float=.05,
                 opt_type: str = 'max', 
                 obj_preference: str = 'scalarization',
                 sclarization_type = 'weighted_sum',
                 ) -> None:
        """
        Multi-objective stochastic linear bandit
        UCB algorithm considering Pareto order

        Parameters
        ----------
        num_dim : int
            number of context's dimension
        num_obj : int
            number of objectives
        lamda : float, optional
            regularization term, lambda, by default 1.0
        delta : float, optional 
            confidence level value, by default 0.05
        """
        super().__init__(num_dim=num_dim, 
                         num_obj=num_obj, 
                         num_arm=num_arm, 
                         lamda=lamda, 
                         delta=delta)
        self.opt_type = opt_type
        self.obj_preference = obj_preference
        self.sclarization_type = sclarization_type

    def reset(self) -> None: 
        """
        Initialize the parameters 
        """
        super().reset()
        self.V = self.lamda * np.eye(self.d) 
        self.theta = np.zeros((self.m, self.d))
        self.V_inv = 1/self.lamda * self.V 

    def estimate_reward(self, arm: np.ndarray) -> float: 
        """
        Estimate the expected reward for an arm

        Parameters
        ----------
        arm : np.ndarray
            arm's context

        Returns
        -------
        float
            estimated reward
        """
        assert arm.ndim == 1
        return np.dot(arm, self.theta.T) 
    
    def estimate_uncertainty(self, arm:np.ndarray) -> float: 
        """
        Estimate the width of confidence level for an arm

        Parameters
        ----------
        arm : np.ndarray
            arm's context

        Returns
        -------
        float
            estimated variance
        """
        assert arm.ndim == 1
        # gamma_t = np.sqrt(self.d * np.log(self.m * (1 + self.t) / self.delta)) + 1 # scalars in theoretical analysis
        gamma_t = 1. 
        w_t = gamma_t * np.sqrt(np.dot(arm, self.V_inv).dot(arm))
        return w_t

    def _eval_ucb(self, arm:np.ndarray, alpha: float=1.) -> np.ndarray: 
        """
        Evaluate the upper confidence bound for an arm

        Returns
        -------
        np.ndarray
            upper confidence bound of the estimated reward
        """
        return self.estimate_reward(arm) + alpha * self.estimate_uncertainty(arm)

    def _eval_lcb(self, arm:np.ndarray, alpha: float=1.) -> np.ndarray: 
        """
        Evaluate the lower confidence bound for an arm

        Returns
        -------
        np.ndarray
            lower confidence bound of the estimated reward
        """
        return self.estimate_reward(arm) - alpha * self.estimate_uncertainty(arm)

    def take_action(self, 
        context:np.ndarray, 
        weight_vector: np.ndarray = None, 
        alpha: float = 1.
        ) -> int: 
        """
        Take an action based on P-UCB algorithm

        Parameters
        ----------
        arm : np.ndarray
            arms' context
        alpha : float, optional
            parameter to control the uncertainty level, by default 1.

        Returns
        -------
        int
            index of the selected arm
        """
        arm = np.atleast_2d(context)
        ucb = np.vstack([self._eval_ucb(arm=arm[i],alpha=alpha) for i in range(arm.shape[0])])
        match self.obj_preference.lower():
            case 'pareto': 
                self.opt_ind = par_non_dominated_sorting(ucb)
                return np.random.choice(self.opt_ind, size=1).item()
            case 'scalarization': 
                assert weight_vector is not None
                metric = self._scalarization(ucb, weight_vector=weight_vector)
                return np.argmax(metric) if self.opt_type.lower() == 'max' else np.argmin(metric)

    def update(self, 
               info) -> None: 
        """
        Update the parameters

        Parameters
        ----------
        arm_context : np.ndarray
            context of the selected arm
        reward : np.ndarray
            observed reward of the arm
        """
        action, reward, arm_context = info
        super().update(info)
        self.V += np.outer(arm_context, arm_context)
        self.V_inv = np.linalg.inv(self.V) 
        for i in range(self.m): 
            self.theta[i] = self.V_inv @ self.context_history.T @ self.reward_history[:, i]

    def _scalarization(self, y, weight_vector): 
        y = np.atleast_2d(y)
        match self.sclarization_type.lower(): 
            case 'weighted_sum': 
                return np.sum(y * weight_vector, axis=1)
            

class DF_TSLB(moslb): 
    def __init__(self, 
                 num_arm,
                 num_dim, 
                 num_obj, 
                 budget: int, 
                 lamda = 1, 
                 delta = 0.05, 
                 opt_type = 'max', 
                 B: float = 1.0, 
                 ):
        super().__init__(num_arm, num_dim, num_obj, lamda, delta, opt_type)
        self.B = B
        self.T = budget
        
    def reset(self) -> None: 
        super().reset() 
        self.V = self.lamda * np.eye(self.d) 
        self.theta = np.zeros((self.m, self.d))
        self.V_inv = 1/self.lamda * self.V 
        # 
        self.s = 1
        self.A_s = np.arange(self.K)
        
    def take_action(self, 
                    context: np.array,
                    problem_lamda: float = 1.,  
                    alpha = 1):
        if self.total_counts < self.K: 
            return np.random.choice(np.where(self.counts<1)[0])
        
        arm_uncertainty = np.array([
            self.estimate_uncertainty(arm) for arm in context
        ]) / alpha
        # exploration parameters 
        R = 1.0
        self.alpha_t = (R * np.sqrt(self.d * np.log(4*self.t*self.m*self.T/self.delta)) + self.B) / alpha 
        self.beta_t = self.alpha_t * np.sqrt(2 * self.d * np.log(4*self.t*self.m*self.T/self.delta))
        # decision making process
        arm_reward =  context @ self.theta.T
        if np.max(arm_uncertainty) <= (1 / self.T): 
            opt_ind = [self.A_s]
            for i in range(self.m): 
                opt_ind.append(
                    self.max_filter(opt_ind[-1],
                                    arm_reward[:,i],
                                    2.+np.sum([4*problem_lamda**j for j in range(i+1)]), 
                                    (self.alpha_t+self.beta_t)/np.sqrt(self.T))
                )
            for i in reversed(range(len(opt_ind))): 
                if len(opt_ind[i]) > 0: 
                    return np.random.choice(opt_ind[i]).item()
        elif np.max(arm_uncertainty) > 2**(-self.s): 
            return np.random.choice(np.where(arm_uncertainty>2**(-self.s))[0]).item()
        else: 
            opt_ind = [self.A_s] 
            for i in range(self.m): 
                opt_ind.append(
                    self.max_filter(opt_ind[-1],
                                    arm_reward[:,i],
                                    2.+np.sum([4*problem_lamda**j for j in range(i+1)]), 
                                    (self.alpha_t+self.beta_t)/(2**(-self.s)))
                )
            for i in reversed(range(len(opt_ind))): 
                if len(opt_ind[i]) > 0: 
                    self.A_s = opt_ind[i] 
                    self.s += 1
                    return self.take_action(context, problem_lamda, alpha)
        
    @classmethod
    def max_filter(cls, 
                   opt_ind: np.ndarray, 
                   mean: np.ndarray, 
                   problem_lamda: float, 
                   C: float, 
                   ): 
        max_mean = np.max(mean[opt_ind]) 
        sub_opt_ind = [i for i in opt_ind if (max_mean - mean[i]) <= problem_lamda * C]
        return np.array(sub_opt_ind)
        
        
        
    
 

if __name__ == '__main__': 
    print() 
    import numpy as np 

    from zomba.multiObjective.algorithms.stochastic import MONeural, moslb
    from zomba.multiObjective.utils import runif_in_simplex
    from pymoo.problems import get_problem 

    from zomba.multiObjective.simulators import moContextMABSimulator

    class mooBandits(moContextMABSimulator): 
        def __init__(self, num_arm: int = None, num_dim: int = None, num_obj: int = None, arm_context: np.ndarray = None, obj_preference: str = 'scalarization', sclarization_type='weighted_sum', vary_context: bool = False, noise_var: float = 0.1) -> None:
            super().__init__(num_arm, num_dim, num_obj, arm_context, obj_preference, sclarization_type, vary_context, noise_var)
        
        def _sample_context(self):
            self.A = np.random.rand(self.K, self.d)
        
        def _eval_expected_reward(self, arm):
            return p.evaluate(arm)
    K = 10

    p = get_problem("zdt6")
    d = p.n_var
    m = p.n_obj
    env = mooBandits(
        num_arm=K, 
        num_dim=d, 
        num_obj=m, 
        vary_context=1,
    )
    moslb_ucb = moslb(
        num_dim=d, 
        num_obj=m, 
        lamda=1., 
        delta=.01, 
        opt_type='min', 
    )
    env.reset()
    moslb_ucb.reset()
    T = 1000 
    tot_reg = 0
    tot_reg_his = []
    for t in range(T): 
        # sample preference vector 
        weight_vector = runif_in_simplex(m)
        X = env.observe_context()
        a_t = moslb_ucb.take_action(context=X, alpha=0.01, weight_vector=weight_vector)
        reg_t = env.get_regret(arm=a_t, weight_vector=weight_vector).item()
        r_t = env.get_reward(arm=a_t)
        moslb_ucb.update(arm_context=X[a_t], reward=r_t)
        print(f"Round: {t}, instantaneous regret: {reg_t:f}.")
        tot_reg += reg_t
        tot_reg_his.append(tot_reg)