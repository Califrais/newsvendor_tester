from algorithms.algorithm import Algorithm

class constant_algorithm(Algorithm) :
    def __init__(self, constant_decision) :
        self.constant_decision = constant_decision

    def next_decision(self, t, state, subgradient, sales, demands) :
        return self.constant_decision
    
    def __str__(self) :
        return "Constant"