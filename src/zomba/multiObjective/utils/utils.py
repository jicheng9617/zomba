import copy 

import numpy as np  


def runif_in_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)


def max_filter(opt_ind: np.ndarray, 
               mean: np.ndarray, 
               uncertainty_scale: float, 
               uncertainty: np.ndarray, 
               ): 
    max_mean = np.max(mean[opt_ind]) 
    sub_opt_ind = [i for i in opt_ind if (max_mean - mean[i]) <= uncertainty_scale * uncertainty[i]]
    return np.array(sub_opt_ind)


def prior_free_lexi_filter(ucb: np.ndarray, lcb: np.ndarray) -> np.ndarray: 
    """
    filter the optimal arms based on transitive closure relation of the linked relation  

    Parameters
    ----------
    ucb : np.ndarray
        upper confidence bound of the arms
    lcb : np.ndarray
        lower confidence bound of the arms

    Returns
    -------
    np.ndarray
        index of the optimal arms
    """
    K,mc = ucb.shape
    opt_ind = [np.arange(K)]

    for i in range(mc): 
        x_i = opt_ind[i][np.argmax(ucb[opt_ind[i], i])]
        opt_ind.append(
            chain_filter(x_i,opt_ind[i],ucb[:, i],lcb[:, i])
        )
    return opt_ind[-1]

def chain_filter(arm, D1, u_t, l_t):
    pre_results = []
    results = [arm]
    lowest = l_t[arm]

    while len(results) != len(pre_results):
        pre_results = copy.deepcopy(results)
        for i in D1:
            if u_t[i] >= lowest and i not in results:
                results.append(i)
                lowest = np.min([lowest, l_t[i]])
    return np.array(results)