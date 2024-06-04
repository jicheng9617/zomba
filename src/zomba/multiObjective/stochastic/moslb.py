import numpy as np 

from zomba.multiObjective.utils import par_non_dominated_sorting

class moslb:
    def __init__(self,
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
        self.d = num_dim
        self.m = num_obj
        self.lamda = lamda
        self.delta = delta
        self.opt_type = opt_type
        self.obj_preference = obj_preference
        self.sclarization_type = sclarization_type

    @property
    def num_obj(self) -> int:
        return self.m
    
    @property
    def num_dim(self) -> int: 
        return self.d
    
    @property
    def optimal_arms(self) -> np.ndarray: 
        return self.opt_ind

    def reset(self) -> None: 
        """
        Initialize the parameters 
        """
        self.t = 1
        self.X_list = [] 
        self.Y_list = [] 
        self.V = self.lamda * np.eye(self.d) 
        self.theta = np.zeros((self.m, self.d))
        self.V_inv = 1/self.lamda * self.V 

    def estimate_reward(self, arm:np.ndarray) -> float: 
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
        return self.estimate_reward(arm) + alpha*self.estimate_uncertainty(arm)

    def _eval_lcb(self, arm:np.ndarray, alpha: float=1.) -> np.ndarray: 
        """
        Evaluate the lower confidence bound for an arm

        Returns
        -------
        np.ndarray
            lower confidence bound of the estimated reward
        """
        return self.estimate_reward(arm) - alpha*self.estimate_uncertainty(arm)

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
        self.t += 1
        self.X_list.append(arm_context) 
        self.Y_list.append(reward)
        X = np.vstack(self.X_list) 
        Y = np.vstack(self.Y_list) 
        self.V += np.outer(arm_context, arm_context)
        self.V_inv = np.linalg.inv(self.V) 
        for i in range(self.m): 
            self.theta[i] = self.V_inv @ X.T @ Y[:, i]

    def _scalarization(self, y, weight_vector): 
        y = np.atleast_2d(y)
        match self.sclarization_type.lower(): 
            case 'weighted_sum': 
                return np.sum(y * weight_vector, axis=1)
            

if __name__ == '__main__': 
    print() 
    import numpy as np 

    from zomba.multiObjective.stochastic import MONeural, moslb
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