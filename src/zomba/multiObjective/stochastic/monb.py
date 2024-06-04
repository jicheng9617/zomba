import numpy as np 
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from zomba.core import contextMABAgent



class ffNet(nn.Module):
    def __init__(self, dim, hidden_size=100, hidden_layer=1):
        super(ffNet, self).__init__()
        self.d = dim
        self.m = hidden_size
        self.L = hidden_layer
        self.modules = [nn.Linear(dim, hidden_size),nn.ReLU()]
        for i in range (1, hidden_layer-1):
            self.modules.append(nn.Linear(hidden_size, hidden_size))
            self.modules.append(nn.ReLU())
            
        self.modules.append(nn.Linear(hidden_size, 1))
        self.modules.append(nn.ReLU())
        self.sequential = nn.ModuleList(self.modules)

    def forward(self, x):
        x = x.float()
        for layer in self.sequential: 
            x = layer(x)
        return x

class MONeural(contextMABAgent):
    """
    Multi-objective neural bandits agents with UCB and TS
    """
    def __init__(self, 
                 num_dim: int = None, 
                 num_arm: int = None, 
                 num_obj: int = None, 
                 hidden_size: int = 100, 
                 hidden_layer: int = 1, 
                 rho: float = 1., 
                 lamda: float = 1, 
                 delta: float = 0.05, 
                 style: str = 'ucb', 
                 opt_type: str = 'max', 
                 obj_preference: str = 'scalarization', 
                 sclarization_type = 'weighted_sum',
                 ) -> None:
        super().__init__(num_dim, num_arm, lamda, delta)
        self.m = num_obj
        self.rho = rho 
        self.style = style
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.opt_type = opt_type
        self.obj_type = obj_preference
        self.sclarization_type = sclarization_type

    @property
    def num_obj(self): 
        return self.m
    
    def reset(self,
              num_dim: int = None,
              num_arm: int = None,
              num_obj: int = None,
              hidden: int = None,
              rho: float = None,
              lamda: float = None,
              delta: float = None,
              style: str = None,
              ) -> None:
        super().reset(num_arm=num_arm, num_dim=num_dim)
        if num_obj is not None: self.m = num_obj 
        if hidden is not None: self.hidden_size = hidden 
        if rho is not None: self.rho = rho 
        if lamda is not None: self.lamda = lamda 
        if delta is not None: self.delta = delta 
        if style is not None: self.style = style 
        assert self.m is not None, "#objectives not define!" 

        # initialize m neural networks for m objectives respectively
        self.f = [ffNet(dim=self.d, hidden_size=self.hidden_size, hidden_layer=self.hidden_layer).cuda() for _ in range(self.m)]
        self.total_param = [sum(p.numel() for p in func.parameters() if p.requires_grad) for func in self.f]
        self.U = [self.lamda * torch.ones((total_param,)).cuda() for total_param in self.total_param]
        self.update_threshold = 1.0
        self.U_prime = copy.deepcopy(self.U)
        self.optimizer = [torch.optim.Adam(net.parameters(), lr = 5e-3) for net in self.f] 
        # self.optimizer = [optim.SGD(net.parameters(), lr=2e-3) for net in self.f] 

    def take_action(self, 
                    context: np.ndarray, 
                    weight_vector: list | np.ndarray, 
                    ) -> int:
        tensor = torch.from_numpy(context).float().cuda()
        mu = torch.hstack([fx(tensor) for fx in self.f])
        samples = []
        for fx in mu: 
            g = self._eval_grad(fx)
            sigma2 = [self.lamda * g_i * g_i / U for g_i, U in zip(g, self.U)]
            sigma = torch.hstack([torch.sqrt(torch.sum(tmp)) for tmp in sigma2])
            match self.style.lower():
                case 'ucb': sample_r = fx + self.rho * sigma
                case 'ts': sample_r = torch.normal(mean=fx, std=self.rho * sigma)
            samples.append(sample_r)
        scalrized_samples = torch.sum(torch.vstack(samples) * torch.tensor(weight_vector).cuda(), dim=1)
        arm = torch.argmax(scalrized_samples) if self.opt_type.lower() == 'max' else torch.argmin(scalrized_samples)
        return arm.item()
    
    def update(self, info: int | float | np.ndarray) -> None:
        super().update(info)
        action, reward, context = info
        # update U 
        c = torch.tensor(context).cuda()
        fx = torch.hstack([fx(c) for fx in self.f])
        g = self._eval_grad(torch.atleast_1d(fx))
        self.U = [self.U[i] + g[i]*g[i] for i in range(self.m)]
        # neural networks updates 
        # if self.round % 100 == 0:
        if torch.max(torch.tensor([torch.sum(self.U[i]).item() / torch.sum(self.U_prime[i]).item() for i in range(self.m)])) >= self.update_threshold:
            print("***************Network Updating***************")
            self.U_prime = copy.deepcopy(self.U)
            length = len(self.reward_his)
            index = np.arange(length) 
            np.random.shuffle(index)
            self._train(index)

    def _train(self, index): 
        loss_obj = []
        for j in range(self.m): 

            tot_loss = 0 
            self.optimizer[j].zero_grad() 
            current_loss = torch.sum((self.f[j](torch.tensor(np.vstack(self.X_list)).cuda()).reshape(-1, ) - torch.tensor(self.reward_his[:, j]).cuda()) ** 2)
            if self.t == 0: 
                current_loss.backward(retain_graph=True) 
            else: 
                current_loss.backward()
            tot_loss += current_loss.item()
            self.optimizer[j].step()

            # cnt = 0
            # tot_loss = 0 
            # tot_step = 1000
            # length = len(self.reward_his)
            # while True: 
            #     batch_loss = 0 
            #     for idx in index: 
            #         c = torch.tensor(self.context_his[idx]).cuda()
            #         r = self.reward_his[idx, j].item()
            #         self.optimizer[j].zero_grad()
            #         r_prime = self.f[j](c)
            #         delta = r_prime - r
            #         loss = delta * delta 
            #         loss.backward()
            #         self.optimizer[j].step()
            #         batch_loss += loss.item() 
            #         tot_loss += loss.item() 
            #         cnt += 1 
            #     if cnt >= tot_step: 
            #         break
            #     if batch_loss / length <= 1e-3: 
            #         break
            loss_obj.append(tot_loss)
        # print(f"Losses for each objective: {loss_obj}")

    def _scalarization(self, 
                       y: np.ndarray, 
                       weight_vector: np.ndarray) -> np.ndarray: 
        y = np.atleast_2d(y)
        match self.sclarization_type.lower(): 
            case 'weighted_sum': 
                return np.sum(y*weight_vector, axis=1).reshape(-1, )
            
    def _eval_grad(self, 
                   fx): 
        g = []
        for fx_i, func in zip(fx, self.f): 
            func.zero_grad()
            fx_i.backward(retain_graph=True)
            g.append(torch.cat([p.grad.flatten().detach() for p in func.parameters()]))
        return g

            
    


if __name__ == "__main__": 
    print() 
    from pymoo.problems import get_problem 
    import numpy as np 
    from zomba.multiObjective.stochastic import MONeural
    from zomba.multiObjective.utils import runif_in_simplex
    from zomba.multiObjective.simulators import moContextMABSimulator

    d = 10
    m = 2
    # quadratic rewards
    A = [np.random.normal(size=(d,d)) for _ in range(m)]
    class moBandits(moContextMABSimulator): 
        
        def _sample_context(self):
            self.A = np.random.rand(self.K, self.d)
        
        def _eval_expected_reward(self, arm):
            return np.hstack([arm @ A[i].T @ A[i] @ arm.T for i in range(self.m)])
    K = 10

    env = moBandits(
        num_arm=K, 
        num_dim=d, 
        num_obj=m, 
        vary_context=1,
        opt_type='min',
    )
    mon_ucb = MONeural(
        num_arm=K, 
        num_dim=d, 
        num_obj=m,
        opt_type='min',
        style='ucb', 
        lamda=1., 
        delta=.05, 
        hidden=100, 
        rho=0.1, 
    )
    env.reset()
    mon_ucb.reset()


    T = 2000 

    for t in range(T): 
        # sample preference vector 
        weight_vector = runif_in_simplex(m)
        X = env.observe_context()
        a_t = mon_ucb.take_action(context=X, weight_vector=weight_vector)
        reg_t = env.get_regret(arm=a_t, weight_vector=weight_vector).item()
        r_t = env.get_reward(arm=a_t)
        mon_ucb.update(info=(a_t, r_t, X[a_t]))
        print(f"Round: {t}, instantaneous regret: {reg_t:f}.")
