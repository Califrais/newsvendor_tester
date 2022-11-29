from algorithms.algorithm import Algorithm
import numpy as np

class OSD2_algorithm(Algorithm) :
    """
    Projected OSD on hyperectangle with learning rate gamma*D*(min_sales)/(G*sqrt(t))
    """
    def __init__(self, nb_products, y_min, y_max, G, gamma) :
        self.nb_products = nb_products
        self.gamma = gamma
        self.y_min = y_min
        self.y_max = y_max
        self.G = G

        self.D = np.max(y_max-y_min)
        self.decision = np.array(y_min)
        self.min_sales = np.inf

    def next_decision(self, t,state,  subgradient, sales, demands) :
        if(t==1) :
            self.decision = np.array(self.y_min+self.y_max)/2
            return self.decision
        else :
            self.min_sales =  np.minimum(np.min(sales), self.min_sales)
            learning_rate = self.gamma*self.D*self.min_sales/(self.G*np.sqrt(t-1))
            self.decision = np.clip(self.decision-learning_rate*subgradient,self.y_min,self.y_max)
            return np.array(self.decision)

    def __str__(self) :
        return "OSD2 gamma*min_sales="+str(self.gamma*self.min_sales)