import numpy as np 

class experimenter: 
    def __init__(self, 
                 banditAlgrithm, 
                 environment, 
                 ) -> None:
        self.alg = banditAlgrithm 
        self.env = environment 

    def reset(self, 
              num_rep: int, 
              banditAlgrithm: any=None, 
              environment: any=None, 
              **kwargs, 
              ) -> None: 
        self.num_rep = num_rep 
        self.cum_regret = np.zeros()

        if banditAlgrithm is not None: self.alg = banditAlgrithm(**kwargs)
        if environment is not None: self.env = environment(**kwargs) 

    def run(self, 
            
            **kwargs, 
            ): 
        self.alg.reset(**kwargs)
        self.env.reset(**kwargs) 