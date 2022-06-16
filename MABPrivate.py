# Run this code to generate Figure 1 in main paper and Figure 2, 3 in the appendix 

import numpy as np
import time

import os

from scipy.stats import norm, uniform, truncnorm
from numpy.random import default_rng
rng = default_rng()

# Generate sample from polya distribution
def polya(r,p):
        lamda = rng.gamma(shape = r, scale = (1-p)/p)
        return rng.poisson(lamda)
# Generate sample from skellam distribution    
def skellam(mu, sigma_sq):
        return (mu + rng.poisson(sigma_sq/2) -  rng.poisson(sigma_sq/2))  

# Bandit classes are taken from https://github.com/vaswanis/randucb
class Bandit:
    def __init__(self, mu, epsilon, seed):
        self.mu = np.copy(mu)
        self.K = self.mu.size
        self.best_arm = np.argmax(self.mu)
        print("Best arm = ",self.best_arm)
        
        self.seed = seed
        self.epsilon = epsilon
    
    def reset_random(self):
        self.random = np.random.RandomState(self.seed)
    
    def reward(self, arm):
        # instantaneous reward of the arm
        return self.rt[arm]

    def regret(self, arm):
        # instantaneous regret of the arm
        return self.rt[self.best_arm] - self.rt[arm]

    def pregret(self, arm):
        # expected regret of the arm
        return self.mu[self.best_arm] - self.mu[arm]

class BerBandit(Bandit):
    """Bernoulli bandit."""
    def __init__(self, mu, epsilon, seed=None):
        super().__init__(mu, epsilon, seed)

    def randomize(self):
        # generate random rewards
        self.rt = (self.random.rand() < self.mu).astype(float)

    def print(self):
        return "Bernoulli bandit with arms (%s)" % ", ".join("%.3f" % s for s in self.mu)

class GaussBandit(Bandit):
    """Gaussian bandit."""
    def __init__(self, mu, epsilon, seed=None):
        super().__init__(mu, epsilon, seed)

    def randomize(self):
        # generate random rewards
        self.rt = (np.minimum(np.maximum(self.random.normal(self.mu, 0.1), 0), 1)).astype(float)  

    def print(self):
        return "Gaussian bandit with arms (%s)" % ", ".join("%.3f" % s for s in self.mu)

class BetaBandit(Bandit):
    """Beta bandit."""

    def __init__(self, mu,  epsilon, a_plus_b=4, seed=None):
        super().__init__(mu, epsilon, seed)
        self.a_plus_b = a_plus_b

    def randomize(self):
        # generate random rewards
        self.rt = self.random.beta(self.a_plus_b * self.mu, self.a_plus_b * (1 - self.mu))

    def print(self):
        return "Beta bandit with arms (%s)" % ", ".join("%.3f" % s for s in self.mu)
    
def evaluate_one(Alg, params, env, n, period_size):
    """One run of a bandit algorithm."""
    alg = Alg(env, n, params)

    regret = np.zeros(n // period_size)
    for t in range(n):
        # generate state
        env.randomize()
        # print("episode", t, "-- rewards:", env.rt)

        # take action
        arm = alg.get_arm(t)

        # update model and regret
        alg.update(t, arm, env.reward(arm))
        regret_at_t = env.regret(arm)
        regret[t // period_size] += regret_at_t

    return regret, alg

def evaluate(Alg, params, envs, n, period_size=1, printout=True):
    """Multiple runs of a bandit algorithm."""
    if printout:
        print("Evaluating %s" % Alg.print(), end="")
    
    start = time.time()

    num_exps = len(envs)
    regret = np.zeros((n // period_size, num_exps))
    alg = num_exps * [None]

    dots = np.linspace(0, num_exps - 1, 100).astype(int)
    for i, env in enumerate(envs):
        env.reset_random()
        
        output = evaluate_one(Alg, params, env, n, period_size)
        regret[:, i] = output[0]
        alg[i] = output[1]

        if i in dots and printout:
            print(".", end="")

    if printout:
        print(" %.1f seconds" % (time.time() - start))
        
        total_regret = regret.sum(axis=0)
        print("Regret: %.2f +/- %.2f (median: %.2f, max: %.2f)" %
            (total_regret.mean(), total_regret.std() / np.sqrt(num_exps),
            np.median(total_regret), total_regret.max()))

    return regret, alg

# Batch based successive arm elimination algorithm (parent class)

class BatchSE:
    def __init__(self, env, n, params):
        self.K = env.K
        self.n = n
        
        for attr, val in params.items():
            setattr(self, attr, val)
        
        self.BatchNo = 1
        self.BatchSize = self.dc ** self.BatchNo
        self.ActiveSet = np.arange(self.K) 
        self.TotalRoundsDone = 0
        self.RoundCount = 0
        
    def confidence_interval(self, t):
        ct = np.sqrt(np.log(len(self.ActiveSet) * (self.BatchNo**2)) / self.BatchSize)
        return ct * self.crs
    
    def local_rand(self, x):
        return x
    
    def central_rand(self):
        return self.rewards
    
    def get_arm(self, t):
        armId = (t - self.TotalRoundsDone) // self.BatchSize
        arm = self.ActiveSet[armId]
        return arm
    
    def update(self, t, arm, r):
        
        if self.RoundCount == 0:
            self.rewards = np.zeros(len(self.ActiveSet))
        
        armId = np.where(self.ActiveSet == arm)
        self.rewards[armId] += self.local_rand(r) 
        self.RoundCount += 1
        
        if self.RoundCount == (self.BatchSize * len(self.ActiveSet)):
            mu_hat = self.central_rand() / self.BatchSize
            ucb = mu_hat + self.confidence_interval(t)  
            lcb = mu_hat - self.confidence_interval(t)
            BadArmsId = np.where(ucb < np.max(lcb))
            self.ActiveSet = np.delete(self.ActiveSet, BadArmsId)
        
            self.BatchNo += 1
            self.BatchSize = self.dc ** self.BatchNo
            self.TotalRoundsDone += self.RoundCount
            self.RoundCount = 0      
            
    @staticmethod
    def print():
        return "BatchSE"
    
# DP-SE algorithm of Sajet and Sheffet (2019)    
    
class DPSE(BatchSE):
    def __init__(self, env, n, params):
        BatchSE.__init__(self, env, n, params)
        self.eps = env.epsilon
        
        self.Del = 1 / (self.dc ** self.BatchNo)
        self.BatchSize = 1 + int(max(32*np.log(8*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/(self.Del**2),\
                                8*np.log(4*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/(self.Del*self.eps))) 
        
    def confidence_interval(self, t):
        ct = np.sqrt(np.log(8*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/(2*self.BatchSize)) + \
                     np.log(4*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/(self.BatchSize*self.eps)   
        return ct * self.crs
    
    def local_rand(self, x):
        return x
    
    def central_rand(self):
        PrivRewards = self.rewards + rng.laplace(0, 1 / self.eps)
        return PrivRewards
    
    def update(self, t, arm, r):
        
        if self.RoundCount == 0:
            self.rewards = np.zeros(len(self.ActiveSet))
            
        armId = np.where(self.ActiveSet == arm)
        self.rewards[armId] += self.local_rand(r) 
        self.RoundCount += 1
        
        if self.RoundCount == (self.BatchSize * len(self.ActiveSet)):
            mu_hat = self.central_rand() / self.BatchSize
            ucb = mu_hat + self.confidence_interval(t)  
            lcb = mu_hat - self.confidence_interval(t)
            BadArmsId = np.where(ucb < np.max(lcb))
            self.ActiveSet = np.delete(self.ActiveSet, BadArmsId)
        
            self.BatchNo += 1
            self.Del = 1 / (self.dc ** self.BatchNo)
            self.BatchSize = 1 + int(max(32*np.log(8*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/(self.Del**2),\
                                8*np.log(4*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/(self.Del*self.eps))) 
            self.TotalRoundsDone += self.RoundCount
            self.RoundCount = 0
           
        
    @staticmethod
    def print():
        return "DPSE" 
    
# Our Dist-DP-SE algorithm with discrte noise   
    
class BatchSEDistDP(BatchSE):
    def __init__(self, env, n, params):
        BatchSE.__init__(self, env, n, params)
        self.eps = env.epsilon
        
        
    def confidence_interval(self, t):
        sigma = np.sqrt(2) / self.eps
        h = 1 / self.eps
        ct = np.sqrt(np.log(4*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/(2*self.BatchSize)) + \
                 sigma*np.sqrt(np.log(2*len(self.ActiveSet)*(self.BatchNo**2)/self.delta))/self.BatchSize + \
                 h*np.log(2*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/self.BatchSize
        return ct * self.crs
    
    def local_rand(self, x):
        if self.RoundCount == 0:
            self.precision = np.ceil(self.eps * np.sqrt(self.BatchSize))
            self.accuracy = np.ceil(self.precision * np.log(2/self.delta) / self.eps)
            self.modulo = self.BatchSize * self.precision + 2 * self.accuracy + 1
        
        x_bar  = np.floor(x*self.precision)
        x_hat = x_bar + rng.binomial(1, x*self.precision - x_bar)
        eta = self.gen_rand()
        y = (x_hat + eta) % self.modulo
        return y
    
    def central_rand(self):
        SecAggRewards = self.rewards % self.modulo
        PrivRewards = SecAggRewards / self.precision
        idx = np.where(SecAggRewards > (self.BatchSize * self.precision + self.accuracy))
        PrivRewards[idx] = PrivRewards[idx] - self.modulo / self.precision
        return PrivRewards
    
    def gen_rand(self):
        gamma_p = polya(1 / self.BatchSize, np.exp(-self.eps / self.precision))
        gamma_m = polya(1 / self.BatchSize, np.exp(-self.eps / self.precision))
        return (gamma_p - gamma_m)
     
    @staticmethod
    def print():
        return "BatchSEDistDP" 
    
# Our Dist-RDP-SE algorithm   
    
class BatchSEDistRDP(BatchSE):
    def __init__(self, env, n, params):
        BatchSE.__init__(self, env, n, params)
        self.eps = env.epsilon
        
    def confidence_interval(self, t):
        sigma = 2 / self.eps + np.sqrt(2) / (self.scale * self.eps)
        h = 1 / (self.scale * self.eps)
        ct = np.sqrt(np.log(4*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/(2*self.BatchSize)) + \
                 sigma*np.sqrt(np.log(2*len(self.ActiveSet)*(self.BatchNo**2)/self.delta))/self.BatchSize + \
                 h*np.log(2*len(self.ActiveSet)*(self.BatchNo**2)/self.delta)/self.BatchSize
        return ct * self.crs   
        
    def local_rand(self, x):
        if self.RoundCount == 0:
            self.precision = np.ceil(self.scale*self.eps * np.sqrt(self.BatchSize))
            self.accuracy = np.ceil(2*self.precision*np.log(2/self.delta)/self.eps + np.sqrt(2)*np.log(2/self.delta))
            self.modulo = self.BatchSize * self.precision + 2 * self.accuracy + 1
        
        x_bar  = np.floor(x*self.precision)
        x_hat = x_bar + rng.binomial(1, x*self.precision - x_bar)
        eta = self.gen_rand()
        y = (x_hat + eta) % self.modulo
        return y
    
    def central_rand(self):
        SecAggRewards = self.rewards % self.modulo
        PrivRewards = SecAggRewards / self.precision
        idx = np.where(SecAggRewards > (self.BatchSize * self.precision + self.accuracy))
        PrivRewards[idx] = PrivRewards[idx] - self.modulo / self.precision
        return PrivRewards
    
    def gen_rand(self):
        sigma_sq = (self.precision**2)/(self.BatchSize*(self.eps**2))
        return skellam(0, sigma_sq)
     
    @staticmethod
    def print():
        return "BatchSEDistRDP"  
 

    
if __name__ == "__main__":   # main function taken from https://github.com/vaswanis/randucb
    base_dir = os.path.join(".", "Results", "MAB")


    algorithms = [
        # baselines DP-SE, Dist-DP-SE, Dist-RDP-SE (with differenet scales). 
        # We run our algorithms with batchsizes 4^b
        # we take confidence level = 0.1 
        # we do not tune the confidence widths
        
        (DPSE, {"crs": 1, "delta": 0.1, "dc": 2}, "DPSE(dc=2)"),
        (BatchSEDistDP, {"crs": 1, "delta": 0.1, "dc": 4}, "BatchSEDistDP(dc=4)"),
        (BatchSEDistRDP, {"crs": 1, "delta": 0.1, "scale": 10, "dc": 4}, "BatchSEDistRDP(s=10,dc=4)"),
        (BatchSEDistRDP, {"crs": 1, "delta": 0.1, "scale": 100, "dc": 4}, "BatchSEDistRDP(s=100,dc=4)"),

    ]
    num_runs = 20 
    n = 10**6  # change to 10**7 for hard instance
    K = 10    # number of arms
    
    time_idx=np.array([np.arange(n)+1,]*num_runs).transpose()

    environments = [    
        
        
        # easy instance (gap = 0.5) with epsilon = 0.1, 0.5, 1 
        # comment out for hard instance
        
        (GaussBandit, {}, 0.1, 0.5, "Gaussian (easy, K = 10, eps = 0.1)"),
        (GaussBandit, {}, 0.5, 0.5, "Gaussian (easy, K = 10, eps = 0.5)"),
        (GaussBandit, {}, 1, 0.5, "Gaussian (easy, K = 10, eps = 1)"),
        
        # hard instance (gap = 0.1) with epsilon = 0.1, 0.5, 1
        #comment out for easy instance
        
        (GaussBandit, {}, 0.1, 0.1, "Gaussian (hard, K = 10, eps = 0.1)"),
        (GaussBandit, {}, 0.5, 0.1, "Gaussian (hard, K = 10, eps = 0.5)"),
        (GaussBandit, {}, 1, 0.1, "Gaussian (hard, K = 10, eps = 1)"),
        

    ]

    for env_def in environments:
        env_class, env_params, epsilon, max_gap, env_name = env_def[0], env_def[1], env_def[2], env_def[3], env_def[-1]
        print("================== running environment", env_name, "==================")
    
        envs = []
        for run in range(num_runs):

            np.random.seed(run)

            mu = max_gap * np.random.rand(K) + (0.5 - max_gap/2)
            envs.append(env_class(mu, epsilon, seed=run, **env_params))
    
            res_dir = os.path.join(base_dir, env_name)
            os.makedirs(res_dir, exist_ok=True)

        for alg_def in algorithms:
    
            alg_class, alg_params, alg_name = alg_def[0], alg_def[1], alg_def[-1]        

            fname = os.path.join(res_dir, alg_name)
            if os.path.exists(fname):
                print('File exists. Will load saved file. Moving on to the next algorithm')
            else:
                regret, _ = evaluate(alg_class, alg_params, envs, n)
                cum_regret = regret.cumsum(axis=0)
                avg_regret=cum_regret/time_idx
                np.savetxt(fname, avg_regret, delimiter=",")   # save results in file