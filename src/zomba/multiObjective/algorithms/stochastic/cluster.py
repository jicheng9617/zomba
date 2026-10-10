import numpy as np 
from typing import Tuple

import torch
from torch import nn, optim
import torch.nn.functional as F

import networkx as nx

from zomba.core import moContextMABAgent
from zomba.multiObjective.utils import par_non_dominated_sorting


class MOLinB_Ind(moContextMABAgent):
    def __init__(self,
                 num_user: int,
                 num_dim: int,
                 num_arm: int,
                 num_obj: int,
                 lamda: float = 1.,
                 delta: float = 0.05,
                 beta_c: float = 0.01):
        super().__init__(num_dim, num_arm, num_obj, lamda, delta)
        self.num_user = num_user
        self.beta_c = beta_c  # Optional function to compute exploration term
        # self.reset()

    def reset(self,
              num_arm: int = None,
              num_obj: int = None,
              num_dim: int = None,
              num_user: int = None):
        super().reset(num_arm, num_obj, num_dim)
        if num_user is not None: self.num_user = num_user

        self.t = 0
        # For each user, maintain m reward estimators
        self.M = [self.lamda * np.eye(self.d) for _ in range(self.num_user)]
        self.b = [np.array([np.zeros(self.d) for _ in range(self.m)]) for _ in range(self.num_user)]
        self.counts_user = np.zeros(self.num_user)

    def estimate_reward(self, user_id: int, arm: np.ndarray) -> np.ndarray:
        """
        Estimate the expected reward vector for a given user and arm.

        Parameters
        ----------
        user_id : int
            User index
        arm : np.ndarray
            Arm context vector of shape (d,)

        Returns
        -------
        np.ndarray
            Estimated reward vector of shape (m,)
        """
        M_inv = np.linalg.inv(self.M[user_id])
        theta_hat = M_inv @ self.b[user_id].T  # Shape: (d, m)
        return arm @ theta_hat  # Shape: (m,)
    
    def estimate_uncertainty(self, user_id: int, arm: np.ndarray) -> float:
        """
        Estimate the shared uncertainty (confidence width) for a given arm.

        Parameters
        ----------
        user_id : int
            User index
        arm : np.ndarray
            Arm context vector (d,)

        Returns
        -------
        float
            Confidence width (scalar)
        """
        M_inv = np.linalg.inv(self.M[user_id])
        return np.sqrt(arm @ M_inv @ arm)

    def take_action(self, user_id: int, context: np.ndarray) -> int:
        """
        Select the arm using Pareto-based UCB selection.

        Parameters
        ----------
        user_id : int
            Current user index
        context : np.ndarray
            Arm context matrix of shape (K, d)

        Returns
        -------
        int
            Selected arm index
        """
        K = context.shape[0]
        ucb = np.zeros((K, self.m))

        for k in range(K):
            x = context[k]
            mean = self.estimate_reward(user_id, x)      # shape: (m,)
            uncert = self.estimate_uncertainty(user_id, x)  # scalar
            ucb[k] = mean + self._compute_beta(user_id=user_id) * uncert  # Broadcast: shape (m,)

        # Perform Pareto-based selection
        self.opt_ind = par_non_dominated_sorting(ucb)
        return np.random.choice(self.opt_ind)

    def update(self, info: Tuple[int, int, np.ndarray, np.ndarray]):
        """
        Update model parameters for a specific user.

        Parameters
        ----------
        info : Tuple[int, int, np.ndarray, np.ndarray]
            (user_id, action, context, reward)
        """
        user_id, action, context, reward = info
        self.t += 1
        self.counts_user[user_id] += 1

        self.M[user_id] += np.outer(context, context)
        for j in range(self.m):
            self.b[user_id][j] += reward[j] * context

        self._action_list.append((user_id, action))
        self._reward_list.append(reward)
        self.X_list.append(context)

    def _compute_beta(self, user_id: int) -> float:
        """
        Compute the exploration coefficient β.

        Parameters
        ----------
        user_id : int

        Returns
        -------
        float
        """
        if self.beta_c:
            return self.beta_c
        # Default β for UCB (can be replaced)
        return np.sqrt(self.lamda) + np.sqrt(2 * np.log(1 / self.delta) + self.d * np.log(1 + self.counts_user[user_id] / self.lamda))


class MOLinB_Shared(moContextMABAgent):
    def __init__(self,
                 num_user: int,
                 num_dim: int,
                 num_arm: int,
                 num_obj: int,
                 lamda: float = 1.,
                 delta: float = 0.05,
                 beta_c: float = 0.01):
        super().__init__(num_dim, num_arm, num_obj, lamda, delta)
        self.num_user = num_user
        self.beta_c = beta_c  # constant scalar or None
        # self.reset()

    def reset(self,
              num_arm: int = None,
              num_obj: int = None,
              num_dim: int = None,
              num_user: int = None):
        super().reset(num_arm, num_obj, num_dim)
        if num_user is not None: self.num_user = num_user

        self.t = 0
        # Shared parameter statistics
        self.M = self.lamda * np.eye(self.d)                # (d, d)
        self.b = np.zeros((self.m, self.d))                 # (m, d)
        self.counts_user = np.zeros(self.num_user)

    def estimate_reward(self, user_id: int, arm: np.ndarray) -> np.ndarray:
        """
        Estimate the expected reward vector for any user and arm (shared model).

        Parameters
        ----------
        user_id : int
            User index (unused in shared setting)
        arm : np.ndarray
            Arm context vector of shape (d,)

        Returns
        -------
        np.ndarray
            Estimated reward vector of shape (m,)
        """
        M_inv = np.linalg.inv(self.M)
        theta_hat = M_inv @ self.b.T  # (d, m)
        return arm @ theta_hat        # (m,)

    def estimate_uncertainty(self, user_id: int, arm: np.ndarray) -> float:
        """
        Estimate shared uncertainty (confidence width) for any user and arm.

        Parameters
        ----------
        user_id : int
            User index (unused in shared setting)
        arm : np.ndarray
            Arm context vector of shape (d,)

        Returns
        -------
        float
            Confidence width (scalar)
        """
        M_inv = np.linalg.inv(self.M)
        return np.sqrt(arm @ M_inv @ arm)

    def take_action(self, user_id: int, context: np.ndarray) -> int:
        """
        Select arm using Pareto-based UCB with a shared model.

        Parameters
        ----------
        user_id : int
            Current user index (unused internally)
        context : np.ndarray
            Arm context matrix (K, d)

        Returns
        -------
        int
            Selected arm index
        """
        K = context.shape[0]
        ucb = np.zeros((K, self.m))

        for k in range(K):
            x = context[k]
            mean = self.estimate_reward(user_id, x)        # (m,)
            uncert = self.estimate_uncertainty(user_id, x) # scalar
            ucb[k] = mean + self._compute_beta(user_id) * uncert

        self.opt_ind = par_non_dominated_sorting(ucb)
        return np.random.choice(self.opt_ind)

    def update(self, info: Tuple[int, int, np.ndarray, np.ndarray]):
        """
        Update the shared model using observation.

        Parameters
        ----------
        info : Tuple[int, int, np.ndarray, np.ndarray]
            (user_id, action, context, reward)
        """
        user_id, action, context, reward = info
        self.t += 1
        self.counts_user[user_id] += 1

        self.M += np.outer(context, context)
        for j in range(self.m):
            self.b[j] += reward[j] * context

        self._action_list.append((user_id, action))
        self._reward_list.append(reward)
        self.X_list.append(context)

    def _compute_beta(self, user_id: int) -> float:
        """
        Compute the exploration coefficient β (can use user_id for adaptive scaling).

        Parameters
        ----------
        user_id : int

        Returns
        -------
        float
        """
        if self.beta_c:
            return self.beta_c
        return np.sqrt(self.lamda) + np.sqrt(
            2 * np.log(1 / self.delta) + self.d * np.log(1 + self.t / self.lamda)
        )


class MOGLinB_Ind(moContextMABAgent):
    def __init__(self, num_user, num_dim, num_arm, num_obj, lamda=1., delta=0.05, beta_c=0.01):
        super().__init__(num_dim, num_arm, num_obj, lamda, delta)
        self.num_user = num_user
        self.beta_c = beta_c

    def reset(self, num_arm=None, num_obj=None, num_dim=None, num_user=None):
        super().reset(num_arm, num_obj, num_dim)
        if num_user is not None:
            self.num_user = num_user

        self.t = 0
        self.M = [self.lamda * np.eye(self.d) for _ in range(self.num_user)]
        self.b = [np.array([np.zeros(self.d) for _ in range(self.m)]) for _ in range(self.num_user)]
        self.counts_user = np.zeros(self.num_user)

    def _phi(self, z: np.ndarray) -> np.ndarray:
        """Apply element-wise link function for generalized linear modeling."""
        return np.tanh(z)

    def estimate_reward(self, user_id: int, arm: np.ndarray) -> np.ndarray:
        M_inv = np.linalg.inv(self.M[user_id])
        theta_hat = M_inv @ self.b[user_id].T
        return self._phi(arm @ theta_hat)

    def estimate_uncertainty(self, user_id: int, arm: np.ndarray) -> float:
        M_inv = np.linalg.inv(self.M[user_id])
        return np.sqrt(arm @ M_inv @ arm)

    def take_action(self, user_id: int, context: np.ndarray) -> int:
        K = context.shape[0]
        ucb = np.zeros((K, self.m))
        beta = self._compute_beta(user_id)

        for k in range(K):
            x = context[k]
            mean = self.estimate_reward(user_id, x)
            uncert = self.estimate_uncertainty(user_id, x)
            ucb[k] = mean + beta * uncert

        self.opt_ind = par_non_dominated_sorting(ucb)
        return np.random.choice(self.opt_ind)

    def update(self, info: Tuple[int, int, np.ndarray, np.ndarray]):
        user_id, action, context, reward = info
        self.t += 1
        self.counts_user[user_id] += 1

        self.M[user_id] += np.outer(context, context)
        for j in range(self.m):
            self.b[user_id][j] += reward[j] * context

        self._action_list.append((user_id, action))
        self._reward_list.append(reward)
        self.X_list.append(context)

    def _compute_beta(self, user_id: int) -> float:
        if self.beta_c:
            return self.beta_c
        return np.sqrt(self.lamda) + np.sqrt(2 * np.log(1 / self.delta) + self.d * np.log(1 + self.counts_user[user_id] / self.lamda))


class MOGLinB_Shared(moContextMABAgent):
    def __init__(self, num_user, num_dim, num_arm, num_obj, lamda=1., delta=0.05, beta_c=0.01):
        super().__init__(num_dim, num_arm, num_obj, lamda, delta)
        self.num_user = num_user
        self.beta_c = beta_c

    def reset(self, num_arm=None, num_obj=None, num_dim=None, num_user=None):
        super().reset(num_arm, num_obj, num_dim)
        if num_user is not None:
            self.num_user = num_user

        self.t = 0
        self.M = self.lamda * np.eye(self.d)
        self.b = np.zeros((self.m, self.d))
        self.counts_user = np.zeros(self.num_user)

    def _phi(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def estimate_reward(self, user_id: int, arm: np.ndarray) -> np.ndarray:
        M_inv = np.linalg.inv(self.M)
        theta_hat = M_inv @ self.b.T
        return self._phi(arm @ theta_hat)

    def estimate_uncertainty(self, user_id: int, arm: np.ndarray) -> float:
        M_inv = np.linalg.inv(self.M)
        return np.sqrt(arm @ M_inv @ arm)

    def take_action(self, user_id: int, context: np.ndarray) -> int:
        K = context.shape[0]
        ucb = np.zeros((K, self.m))
        beta = self._compute_beta(user_id)

        for k in range(K):
            x = context[k]
            mean = self.estimate_reward(user_id, x)
            uncert = self.estimate_uncertainty(user_id, x)
            ucb[k] = mean + beta * uncert

        self.opt_ind = par_non_dominated_sorting(ucb)
        return np.random.choice(self.opt_ind)

    def update(self, info: Tuple[int, int, np.ndarray, np.ndarray]):
        user_id, action, context, reward = info
        self.t += 1
        self.counts_user[user_id] += 1

        self.M += np.outer(context, context)
        for j in range(self.m):
            self.b[j] += reward[j] * context

        self._action_list.append((user_id, action))
        self._reward_list.append(reward)
        self.X_list.append(context)

    def _compute_beta(self, user_id: int) -> float:
        if self.beta_c:
            return self.beta_c
        return np.sqrt(self.lamda) + np.sqrt(2 * np.log(1 / self.delta) + self.d * np.log(1 + self.t / self.lamda))


class COMOLB(moContextMABAgent):
    def __init__(self,
                 num_user: int,
                 num_dim: int,
                 num_arm: int,
                 num_obj: int,
                 lamda: float = 1.0,
                 delta: float = 0.05,
                 beta_c: float = .1, 
                 alpha_1: float = 1.0):
        super().__init__(num_dim, num_arm, num_obj, lamda, delta)
        self.num_user = num_user
        self.alpha_1 = alpha_1
        self.beta_c = beta_c
        # self.reset()

    def reset(self,
              num_arm: int = None,
              num_obj: int = None,
              num_dim: int = None,
              num_user: int = None):
        super().reset(num_arm, num_obj, num_dim)
        if num_user is not None:
            self.num_user = num_user

        self.t = 0
        self.M = [self.lamda * np.eye(self.d) for _ in range(self.num_user)]
        self.b = [np.zeros((self.m, self.d)) for _ in range(self.num_user)]
        self.T = np.zeros(self.num_user, dtype=int)

        self.graph = nx.complete_graph(self.num_user)

    def _f(self, T_val):
        # Confidence width scaling function
        return (np.sqrt(self.lamda) + np.sqrt(self.d * np.log(1 + 100000 / (self.d * self.lamda)) + 2 * np.log(self.num_user / self.delta))) / np.sqrt(2*max(T_val, 1e-6))

    def _beta(self, t):
        if self.beta_c: 
            return self.beta_c
        return np.sqrt(self.lamda) + np.sqrt(self.d * np.log(1 + t / (self.d * self.lamda)) + 2 * np.log(self.num_user * self.m / self.delta))

    def _form_cluster(self, user_id):
        neighbors = list(self.graph.neighbors(user_id))
        return [user_id] + neighbors

    def take_action(self, user_id: int, context: np.ndarray) -> int:
        """
        Select an arm based on cluster-level UCB and Pareto-optimal filtering.
        """
        cluster = self._form_cluster(user_id)

        M_bar = self.lamda * np.eye(self.d)
        b_bar = np.zeros((self.m, self.d))
        for i in cluster:
            M_bar += self.M[i]
            b_bar += self.b[i]

        M_bar_inv = np.linalg.inv(M_bar)
        theta_bar = M_bar_inv @ b_bar.T  # (d, m)

        beta_t = self._beta(self.t + 1)

        K = context.shape[0]
        ucb = np.zeros((K, self.m))
        for k in range(K):
            x = context[k]
            
            mean = x @ theta_bar
            ucb[k] = mean + beta_t * np.sqrt(x @ M_bar_inv @ x)

        self.opt_ind = par_non_dominated_sorting(ucb)
        return np.random.choice(self.opt_ind)

    def update(self, info: tuple):
        """
        Update statistics for the current user and prune the graph.
        """
        user_id, action, context_vec, reward_vec = info
        self.t += 1
        self.M[user_id] += np.outer(context_vec, context_vec)
        for j in range(self.m):
            self.b[user_id][j] += reward_vec[j] * context_vec
        self.T[user_id] += 1

        # Update theta estimate
        M_inv = np.linalg.inv(self.M[user_id])
        theta_hat_i = M_inv @ self.b[user_id].T  # (d, m)

        # Edge deletion
        neighbors = list(self.graph.neighbors(user_id))
        for neighbor in neighbors:
            M_inv_n = np.linalg.inv(self.M[neighbor])
            theta_hat_n = M_inv_n @ self.b[neighbor].T  # (d, m)
            for j in range(self.m):
                diff = np.linalg.norm(theta_hat_i[:, j] - theta_hat_n[:, j])
                threshold = self.alpha_1 * (self._f(self.T[user_id]) + self._f(self.T[neighbor]))
                if diff >= threshold:
                    self.graph.remove_edge(user_id, neighbor)
                    # print(f"diff:{diff:.3f}, threshold: {threshold:.3f}")
                    break

        self._action_list.append((user_id, action))
        self._reward_list.append(reward_vec)
        self.X_list.append(context_vec)


class SharedFeatureMultiObjectiveNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_obj: int, num_layers: int = 2):
        """
        Neural network with shared feature extractor and separate output head.

        Parameters
        ----------
        input_dim : int
            Input feature dimension (d)
        hidden_dim : int
            Hidden layer and shared feature size (also output of shared extractor)
        num_obj : int
            Number of objectives (m_o)
        num_layers : int
            Number of shared layers before the final objective layer (total = 2L)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_obj = num_obj
        self.num_layers = num_layers

        # Shared encoder (first L layers)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        self.shared_encoder = nn.Sequential(*layers)

        # Shared final linear output for all objectives (d -> m_o)
        # self.output_layer = nn.Linear(hidden_dim, num_obj)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.num_obj))
        self.output_net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, d)

        Returns
        -------
        torch.Tensor of shape (batch_size, m_o)
        """
        features = self.shared_encoder(x)  # shape: (batch_size, d)
        out = self.output_net(features)  # shape: (batch_size, m_o)
        return out
    
    def get_ntk_dim(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_ntk_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute NTK feature vector (gradient of total output wrt all parameters).
        The output is flattened into one vector across all shared and output layer parameters.

        Parameters
        ----------
        x : torch.Tensor of shape (1, d)

        Returns
        -------
        torch.Tensor of shape (m_nn,)
            Flattened gradient vector over all network parameters.
        """
        self.zero_grad()
        x = x.requires_grad_(True)

        # Forward pass to get all outputs
        shared_feat = self.shared_encoder(x)
        out_vec = self.output_net(shared_feat).squeeze(0)  # shape: (m_o,)

        # Combine output objectives (e.g., sum or mean)
        out = out_vec.sum()  # or use out_vec.mean()

        # Backward to compute gradient
        out.backward(retain_graph=False)

        # Collect and flatten all gradients
        grad_list = []
        for param in self.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.view(-1))

        ntk_feature = torch.cat(grad_list).detach()
        self.zero_grad()
        return ntk_feature
    

class COMONB_Ind(moContextMABAgent):
    def __init__(self,
                 num_user: int,
                 num_dim: int,
                 num_arm: int,
                 num_obj: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 lamda: float = 1.0,
                 delta: float = 0.05,
                 beta_c: float = None,
                 device='cpu'):
        super().__init__(num_dim, num_arm, num_obj, lamda, delta)
        self.hidden_dim = hidden_dim
        self.num_user = num_user
        self.device = device
        self.beta_c = beta_c

        self.net_users = [SharedFeatureMultiObjectiveNN(num_dim, hidden_dim, num_obj, num_layers).to(device) for _ in range(self.num_user)]

    def reset(self,
              num_arm: int = None,
              num_obj: int = None,
              num_dim: int = None,
              num_user: int = None):
        super().reset(num_arm, num_obj, num_dim)
        if num_user is not None:
            self.num_user = num_user

        self.user_data = [[] for _ in range(self.num_user)]
        self.T = np.zeros(self.num_user, dtype=int)
        self.M = [self.lamda * torch.ones((self.net_users[i].get_ntk_dim(), ), device=self.device) for i in range(self.num_user)]

    def _beta(self, T, K):
        if self.beta_c is not None:
            return self.beta_c
        d_ntk = self.net.get_ntk_dim()
        return 2 * np.sqrt(d_ntk * np.log(1 + T * K / self.lamda) + 1 + 2 * np.log(self.num_user * self.m / self.delta))
    
    def _fit_nn(self, cluster_data, lr=1e-2, steps=10, model = None):
        if model is None: 
            model = SharedFeatureMultiObjectiveNN(self.d, self.net.hidden_dim, self.m, self.net.num_layers).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        X = torch.stack([x for (x, _) in cluster_data]).to(self.device)
        Y = torch.stack([r for (_, r) in cluster_data]).to(self.device)

        for _ in range(steps):
            optimizer.zero_grad()
            pred = model(X)
            loss = 0.5 * F.mse_loss(pred, Y)
            loss.backward()
            optimizer.step()

        return model
    
    def _compute_cluster_ntk_diag(self, cluster_data) -> torch.Tensor:
        """
        Approximate NTK information with only the diagonal of the Gram matrix.

        Returns
        -------
        torch.Tensor of shape (p,)
        """
        p = self.net.get_ntk_dim()
        m_diag = self.lamda * torch.ones(p, device=self.device)
        for x, _ in cluster_data:
            phi = self.net.get_ntk_feature(x.unsqueeze(0), 0)
            m_diag += phi ** 2  # accumulate φ_j^2
        return m_diag

    def take_action(self, user_id: int, context: np.ndarray) -> int:
    
        context_tensor = torch.from_numpy(context).float().to(self.device)
        K = context_tensor.shape[0]
        ucb = torch.zeros((K, self.m), device=self.device)

        for i in range(K):
            x_i = context_tensor[i]
            pred = self.net_users[user_id](x_i)
            phi = self.net_users[user_id].get_ntk_feature(x_i)
            unc = torch.sqrt(torch.sum(phi ** 2 / self.M[user_id]))
            ucb[i] = pred + torch.tensor(self.beta_c) * unc

        self.opt_ind = par_non_dominated_sorting(ucb.cpu().detach().numpy())
        return int(np.random.choice(self.opt_ind))

    def update(self, info: tuple):
        user_id, action, context_vec, reward_vec = info
        self.t += 1
        x = torch.from_numpy(context_vec).float().to(self.device)
        r = torch.from_numpy(reward_vec).float().to(self.device)
        self.user_data[user_id].append((x, r))
        self.T[user_id] += 1

        self._fit_nn(cluster_data=self.user_data[user_id], 
                     lr=5e-3, 
                     steps=10, 
                     model=self.net_users[user_id])
        
        g = self.net_users[user_id].get_ntk_feature(x)
        self.M[user_id] += g ** 2

        self._action_list.append((user_id, action))
        self._reward_list.append(reward_vec)
        self.X_list.append(context_vec)


class COMONB(moContextMABAgent):
    def __init__(self,
                 num_user: int,
                 num_dim: int,
                 num_arm: int,
                 num_obj: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 lamda: float = 1.0,
                 delta: float = 0.05,
                 alpha_1: float = 1.0,
                 beta_c: float = None,
                 device='cpu'):
        super().__init__(num_dim, num_arm, num_obj, lamda, delta)
        self.hidden_dim = hidden_dim
        self.num_user = num_user
        self.alpha_1 = alpha_1
        self.device = device
        self.beta_c = beta_c

        self.net = SharedFeatureMultiObjectiveNN(num_dim, hidden_dim, num_obj, num_layers).to(device)
        initial_state = self.net.state_dict()

        self.net_users = [SharedFeatureMultiObjectiveNN(num_dim, hidden_dim, num_obj, num_layers).to(device) for _ in range(self.num_user)]
        for net_u in self.net_users:
            net_u.load_state_dict(initial_state)

    def _get_flat_params(self):
        return torch.cat([p.view(-1) for p in self.net.parameters()])

    def reset(self,
              num_arm: int = None,
              num_obj: int = None,
              num_dim: int = None,
              num_user: int = None):
        super().reset(num_arm, num_obj, num_dim)
        if num_user is not None:
            self.num_user = num_user

        self.user_data = [[] for _ in range(self.num_user)]
        self.T = np.zeros(self.num_user, dtype=int)
        self.graph = nx.complete_graph(self.num_user)
        self.M = [self.lamda * torch.ones((self.net.get_ntk_dim(), ), device=self.device) for _ in range(self.num_user)]
        # self.b = [torch.zeros((self.m, self.net.get_ntk_dim()), device=self.device) for _ in range(self.num_user)]

    def _beta(self, T, K):
        if self.beta_c is not None:
            return self.beta_c
        d_ntk = self.net.get_ntk_dim()
        return 2 * np.sqrt(d_ntk * np.log(1 + T * K / self.lamda) + 1 + 2 * np.log(self.num_user * self.m / self.delta))

    def _f(self, T_val):
        beta_T = 2 * np.sqrt(2 * np.log(1+1000*self.K/self.lamda)+1+2*np.log(self.num_user*self.num_obj/self.delta))
        return (beta_T + 2 * np.sqrt(self.lamda) + 2) / np.sqrt(2 * max(T_val, 1))

    def _form_cluster(self, user_id):
        return list(nx.node_connected_component(self.graph, user_id))

    def _fit_nn(self, cluster_data, lr=1e-2, steps=10, model = None):
        if model is None: 
            model = SharedFeatureMultiObjectiveNN(self.d, self.net.hidden_dim, self.m, self.net.num_layers).to(self.device)
        # model.load_state_dict(self.net.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        X = torch.stack([x for (x, _) in cluster_data]).to(self.device)
        Y = torch.stack([r for (_, r) in cluster_data]).to(self.device)

        for _ in range(steps):
            optimizer.zero_grad()
            pred = model(X)
            loss = 0.5 * F.mse_loss(pred, Y)
            loss.backward()
            optimizer.step()

        return model
    
    def _compute_cluster_ntk_diag(self, cluster_data) -> torch.Tensor:
        """
        Approximate NTK information with only the diagonal of the Gram matrix.

        Returns
        -------
        torch.Tensor of shape (p,)
        """
        p = self.net.get_ntk_dim()
        m_diag = self.lamda * torch.ones(p, device=self.device)
        for x, _ in cluster_data:
            phi = self.net.get_ntk_feature(x.unsqueeze(0), 0)
            m_diag += phi ** 2  # accumulate φ_j^2
        return m_diag

    def take_action(self, user_id: int, context: np.ndarray) -> int:
        cluster = self._form_cluster(user_id)
        cluster_data = []
        M_bar = self.lamda * torch.ones((self.net.get_ntk_dim(), ), device=self.device)
        for uid in cluster: 
            cluster_data += self.user_data[uid]
            M_bar += self.M[uid] - self.lamda * torch.ones((self.net.get_ntk_dim(), ), device=self.device)
        # cluster_data = [item for uid in cluster for item in self.user_data[uid]]

        if len(cluster_data) == 0:
            # Cold start: randomly select
            return np.random.randint(context.shape[0])
    
        context_tensor = torch.from_numpy(context).float().to(self.device)
        K = context_tensor.shape[0]
        ucb = torch.zeros((K, self.m), device=self.device)

        model_cluster = self._fit_nn(cluster_data, steps=50)

        for i in range(K):
            x_i = context_tensor[i]
            pred = model_cluster(x_i)
            phi = model_cluster.get_ntk_feature(x_i)
            unc = torch.sqrt(torch.sum(phi ** 2 / M_bar))
            ucb[i] = pred + torch.tensor(self.beta_c) * unc
            # for k in range(self.m):
                # phi = model_cluster.get_ntk_feature_objective(x_i, k)
                # unc = torch.sqrt(torch.sum(phi ** 2 / M_bar))
                # ucb[i, k] = pred[k].item() + self.beta_c * unc.item()
        # print(f"uncertainty: {unc:.3f}")

        self.opt_ind = par_non_dominated_sorting(ucb.cpu().detach().numpy())
        return int(np.random.choice(self.opt_ind))

    def update(self, info: tuple):
        user_id, action, context_vec, reward_vec = info
        self.t += 1
        x = torch.from_numpy(context_vec).float().to(self.device)
        r = torch.from_numpy(reward_vec).float().to(self.device)
        self.user_data[user_id].append((x, r))
        self.T[user_id] += 1

        self._fit_nn(cluster_data=self.user_data[user_id], 
                     lr=5e-3, 
                     steps=10, 
                     model=self.net_users[user_id])
        
        g = self.net_users[user_id].get_ntk_feature(x)
        self.M[user_id] += g ** 2

        for neighbor in list(self.graph.neighbors(user_id)):
            # if len(self.user_data[user_id]) == 0 or len(self.user_data[neighbor]) == 0:
            #     continue  # Skip edge deletion if insufficient data

            # Form vectors
            theta_i = torch.cat([p.view(-1) for p in self.net_users[user_id].parameters()])
            theta_l = torch.cat([p.view(-1) for p in self.net_users[neighbor].parameters()])

            diff = torch.norm(theta_i - theta_l).item()
            threshold = (self._f(self.T[user_id]) + self._f(self.T[neighbor]))
            if diff > self.alpha_1 * threshold:
                self.graph.remove_edge(user_id, neighbor)
        # print(f"difference of NN: {diff:.3f}, threshold: {threshold:.3f}, alpha_1: {self.alpha_1:.3f}")

        self._action_list.append((user_id, action))
        self._reward_list.append(reward_vec)
        self.X_list.append(context_vec)