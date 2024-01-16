import numpy as np 

from zomba.mab import epsilonGreedy 

def test_mabParaChange(): 
    n_arm = np.random.randint(2, 21)
    agent = epsilonGreedy(num_arm=n_arm)
    assert agent.epsilon == .01 
    assert agent.num_arm == n_arm 

    n_arm = np.random.randint(2, 21)
    agent.reset(num_arm=n_arm, 
                epsilon=.1)
    assert agent.epsilon == .1 
    assert agent.num_arm == n_arm 

# from zomba
# def test_BernoulliBandit(): 

