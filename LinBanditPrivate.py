# Script to generate figures of the paper "Shuffle Private Linear Contextual Bandits" (ICML 2022)
# Code is built on https://github.com/vaswanis/randucb

import itertools
import numpy as np
import pandas as pd
import time

import os

from scipy.stats import norm
import scipy

class LinBandit:
    """Linear bandit."""

    def __init__(self, X, theta, epsilon, delta, BatchSize, noise="normal", sigma=0.5, seed=None):
        self.X = np.copy(X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.theta = np.copy(theta)
        self.noise = noise
        if self.noise == "normal":
            self.sigma = sigma

        self.mu = self.X.dot(self.theta)
        self.best_arm = np.argmax(self.mu)

        self.seed = seed
        self.random = np.random.RandomState(seed)
        
        self.epsilon = epsilon
        self.delta = delta
        self.BatchSize = BatchSize
    
    def reset_random(self):
        self.random = np.random.RandomState(self.seed)
        
    def randomize(self): # generate random rewards
        if self.noise == "normal":
            self.rt = self.mu + self.sigma * self.random.randn(self.K)
        elif self.noise == "bernoulli":
            self.rt = (self.random.rand(self.K) < self.mu).astype(float)
        elif self.noise == "beta":
            self.rt = self.random.beta(4 * self.mu, 4 * (1 - self.mu))

    def reward(self, arm): # instantaneous reward of the arm
        return self.rt[arm]

    def regret(self, arm): # instantaneous regret of the arm
        return self.rt[self.best_arm] - self.rt[arm]

    def pregret(self, arm): # expected regret of the arm
        return self.mu[self.best_arm] - self.mu[arm]

    def print(self):
        if self.noise == "normal":
            return "Linear bandit: %d dimensions, %d arms" % (self.d, self.K)
        elif self.noise == "bernoulli":
            return "Bernoulli linear bandit: %d dimensions, %d arms" % (self.d, self.K)
        elif self.noise == "beta":
            return "Beta linear bandit: %d dimensions, %d arms" % (self.d, self.K)


def evaluate_one(Alg, params, env, n, period_size):
    """One run of a bandit algorithm."""
    alg = Alg(env, n, params)
    regret = np.zeros(n // period_size)
    for t in range(n):   
        env.randomize() # generate state
        arm = alg.get_arm(t) # take action
        alg.update(t, arm, env.reward(arm)) # update model and regret
        regret_at_t = env.regret(arm)        
        regret[t // period_size] += regret_at_t       
    return regret, alg

def evaluate(Alg, params, envs, n=1000, period_size=1, printout=True):
    """Multiple runs of a bandit algorithm."""
    if printout:
        print("Evaluating %s" % Alg.print(), end="")
    start = time.time()
    num_exps = len(envs)
    regret = np.zeros((n // period_size, num_exps))
    alg = num_exps * [None]
    dots = np.linspace(0, num_exps - 1, 100).astype(int)
    for i, env in enumerate(envs):
        print('Env number:', i)
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

class LinBanditAlg:
    def __init__(self, env, n, params):
        self.env = env
        self.X = np.copy(env.X)
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.n = n
        self.sigma = 0.5
        self.crs = 1.0 # confidence region scaling
        self.BatchNo = 0
        self.epsilon = env.epsilon
        self.delta = env.delta
        self.BatchSize = env.BatchSize
        for attr, val in params.items():
            setattr(self, attr, val)
        self.Gram = np.zeros((self.d, self.d)) # sufficient statistics
        self.B = np.zeros(self.d)
        
class LinUCB(LinBanditAlg): # Abassi-Yadkori et al (2011)
    def __init__(self, env, n, params):
        LinBanditAlg.__init__(self, env, n, params)
        self.L = np.amax(np.linalg.norm(self.X, axis = 1))
        self.Gamma = 1.0
        self.cew = self.crs * self.confidence_ellipsoid_width(n)
    
    def confidence_ellipsoid_width(self, t):
        alpha = 1 / self.n
        S = np.sqrt(self.d)
        R = self.sigma
        width = np.sqrt(self.Gamma) * S + R * np.sqrt(self.d * np.log((1 + t * np.square(self.L) / self.Gamma) / alpha))
        return width
    
    def update(self, t, arm, r):
        x = self.X[arm, :]
        self.Gram += np.outer(x, x) 
        self.B += x * r 
        
    def get_arm(self, t):
        Lambda = 1e-4 * self.Gamma
        Gram_inv = np.linalg.inv(self.Gram + Lambda * np.eye(self.d))
        theta = Gram_inv.dot(self.B)
        self.mu = self.X.dot(theta) + self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1)) #UCBs
        return np.argmax(self.mu)
    
    @staticmethod
    def print():
        return "LinUCB"        
    
class LDPLinUCB(LinBanditAlg): # Zheng et al (2020)
    
    def __init__(self, env, n, params):
        LinBanditAlg.__init__(self, env, n, params)
        self.L = np.amax(np.linalg.norm(self.X, axis = 1))
        self.noise = np.sqrt(np.log(2.5 / self.delta)) / self.epsilon
        self.Gamma = self.noise * np.sqrt(self.n) * np.sqrt(self.d)
        self.cew = self.crs * self.confidence_ellipsoid_width(n)        
        
    def confidence_ellipsoid_width(self, t):
        alpha = 1 / self.n
        S = np.sqrt(self.d)
        R = self.sigma
        width = np.sqrt(self.Gamma) * S + R * np.sqrt(self.d * np.log((1 + t * np.square(self.L) / self.Gamma) / alpha))
        return width 
    
    def update(self, t, arm, r): # add local noise
        noiseVec = np.random.normal(0,self.noise,self.d)
        noiseMat = np.random.normal(0,self.noise,[self.d,self.d])
        noiseMat = (noiseMat + np.transpose(noiseMat)) / 2.0
        x = self.X[arm, :]
        self.Gram = self.Gram + np.outer(x, x) + noiseMat
        self.B = self.B + x * r + noiseVec
    
    def get_arm(self, t):
        Lambda = 2 * self.Gamma
        Gram_inv = np.linalg.inv(self.Gram + Lambda * np.eye(d))
        theta = Gram_inv.dot(self.B)
        temp = (self.X.dot(Gram_inv) * self.X).sum(axis = 1)
        self.mu = self.X.dot(theta) + self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))
        return np.argmax(self.mu)
        
    @staticmethod
    def print():
        return "LDPLinUCB"
    
class JDPLinUCB(LinBanditAlg): # Sharif and Sheffet (2018)
    def __init__(self, env, n, params):
        LinBanditAlg.__init__(self, env, n, params)
        self.L = np.amax(np.linalg.norm(self.X, axis = 1))
        self.noise = np.sqrt(np.log(self.n)) * np.log(2.5 / self.delta) / self.epsilon
        self.Gamma = self.noise * np.sqrt(np.log(self.n)) * np.sqrt(self.d)
        self.cew = self.crs * self.confidence_ellipsoid_width(n)
        
        self.N = int(np.floor(np.log(self.n))) + 1
        self.noiseVecAll = np.random.normal(0,self.noise,(self.d,self.N))
        self.noiseMatAll = np.random.normal(0,self.noise,(self.d,self.d,self.N))
                    
    def confidence_ellipsoid_width(self, t):
        alpha = 1 / self.n
        S = np.sqrt(self.d)
        R = self.sigma
        width = np.sqrt(self.Gamma) * S + R * np.sqrt(self.d * np.log((1 + t * np.square(self.L) / self.Gamma) / alpha))
        return width 
    
    def update(self, t, arm, r):  
        x = self.X[arm, :]
        self.Gram = self.Gram + np.outer(x, x) 
        self.B = self.B + x * r 
    
    def get_arm(self, t): #simulates total noise in tree based algorithm
        Lambda = 2 * self.Gamma
        noiseVec = np.zeros(self.d)
        noiseMat = np.zeros([self.d, self.d])
        noiseCount = int(np.floor(np.log(1+t))) + 1
        for i in range(noiseCount):
            noiseVec += self.noiseVecAll[:,i]
            noiseMat += self.noiseMatAll[:,:,i]
        noiseMat = (noiseMat + np.transpose(noiseMat)) / 2.0  
        Gram_inv = np.linalg.inv(self.Gram + noiseMat + Lambda * np.eye(d))
        theta = Gram_inv.dot(self.B + noiseVec)
        self.mu = self.X.dot(theta) + self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))
        return np.argmax(self.mu)    
    
    @staticmethod
    def print():
        return "JDPLinUCB"   
    
class SDPLinUCBPAmp(LinBanditAlg):  # Shuffle model with privacy amplification
    def __init__(self, env, n, params):
        LinBanditAlg.__init__(self, env, n, params)
        for attr, val in params.items():
            setattr(self, attr, val)  
        self.L = np.amax(np.linalg.norm(self.X, axis = 1))
        self.noise = np.sqrt(np.log(2.5 * self.BatchSize / self.delta)) / (self.epsilon * np.sqrt(self.BatchSize))
        self.Gamma = self.noise * np.sqrt(self.n) * np.sqrt(self.d)
        self.cew = self.crs * self.confidence_ellipsoid_width(n)
                  
    def confidence_ellipsoid_width(self, t):
        alpha = 1 / self.n
        S = np.sqrt(self.d)
        R = self.sigma
        width = np.sqrt(self.Gamma) * S + R * np.sqrt(self.d * np.log((1 + t * np.square(self.L) / self.Gamma) / alpha)) 
        return width 
    
    def update(self, t, arm, r): # add privacy amplified local noise
        noiseVec = np.random.normal(0,self.noise,self.d)
        noiseMat = np.random.normal(0,self.noise,[self.d,self.d])
        noiseMat = (noiseMat + np.transpose(noiseMat)) / 2.0
        x = self.X[arm, :]
        self.Gram = self.Gram + np.outer(x, x) + noiseMat
        self.B = self.B + x * r + noiseVec
    
    def get_arm(self, t):
        Lambda = 2 * self.Gamma
        if t == (self.BatchNo * self.BatchSize):
            Gram_inv = np.linalg.inv(self.Gram + Lambda * np.eye(d))
            theta = Gram_inv.dot(self.B)
            self.mu = self.X.dot(theta) + self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))
            self.BatchNo += 1
        return np.argmax(self.mu)
        
    @staticmethod
    def print():
        return "SDPLinUCBPAmp"
    
class SDPLinUCBPVec(LinBanditAlg): # Shuffle model with vector sum 
    def __init__(self, env, n, params):
        LinBanditAlg.__init__(self, env, n, params)
        for attr, val in params.items():
            setattr(self, attr, val)          
        self.g = max(2*np.sqrt(self.BatchSize), d, 4)
        self.scale = 1 # tuned scale = 10 for eps=1,10, scale = 1 for eps = 0.2
        self.b = self.scale*self.g**2*(np.log(self.d/self.delta))**2/(self.epsilon**2 * self.BatchSize)
        self.p = 0.25        
        self.noise = np.log(self.d / self.delta) / (self.epsilon * np.sqrt(self.BatchSize))
        self.Gamma = 1/4*self.noise * np.sqrt(self.n) * np.sqrt(self.d) # scale a little bit
        self.L = np.amax(np.linalg.norm(self.X, axis = 1))
        self.cew = self.crs * self.confidence_ellipsoid_width(n)        
        self.vector_input = np.zeros([self.BatchSize,self.d])
        self.matrix_input = np.zeros([self.BatchSize,self.d,self.d])

    def confidence_ellipsoid_width(self, t):
        alpha = 1 / self.n
        S = np.sqrt(self.d)
        R = self.sigma
        width = np.sqrt(self.Gamma) * S + R * np.sqrt(self.d * np.log((1 + t * np.square(self.L) / self.Gamma) / alpha))
        return width 
                
    def update(self, t, arm, r):
        x = self.X[arm, :]
        vector = x * r
        matrix = np.outer(x, x)
        i = t % self.BatchSize
        self.vector_input[i,:] = self.R_vector(vector)
        self.matrix_input[i,:,:] = self.R_matrix(matrix)

    def get_arm(self, t):
        Lambda = 2 * self.Gamma
        if t == 0:
            Gram_inv = np.linalg.inv(self.Gram + Lambda * np.eye(d))
            theta = Gram_inv.dot(self.B)
            self.mu = self.X.dot(theta) + self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))
            self.BatchNo += 1            
        if t == (self.BatchNo * self.BatchSize):            
            vector_Y = self.S_vector(self.vector_input)
            matrix_Y = self.S_matrix(self.matrix_input)            
            self.vector_output = self.A_vector(vector_Y)
            self.matrix_output = self.A_matrix(matrix_Y)
            self.B = self.B + self.vector_output 
            self.Gram = self.Gram + self.matrix_output             
            Gram_inv = np.linalg.inv(self.Gram + Lambda * np.eye(d))
            theta = Gram_inv.dot(self.B)
            self.mu = self.X.dot(theta) + self.cew * np.sqrt((self.X.dot(Gram_inv) * self.X).sum(axis = 1))
            self.BatchNo += 1            
        return np.argmax(self.mu)
    
    def R1d(self, x):
        x_bar  = np.floor(x*self.g/self.L)
        gamma1 = np.random.binomial(1, x*self.g/self.L-x_bar)
        gamma2 = np.random.binomial(self.b,self.p)
        res = x_bar + gamma1 + gamma2 
        return res

    def A1d(self, y_sum):
        out = self.L/self.g*(y_sum - self.p * self.b * self.BatchSize)
        return out

    def R_vector(self, v): # randomizer (vector)
        M = np.zeros(self.d)
        for k in range(self.d):
            wk = v[k] + self.L 
            mk = self.R1d(wk)
            M[k] = mk
        return M

    def R_matrix(self, mat): # randomizer (matrix)
        M = np.zeros([self.d,self.d])
        for i in range(self.d):
            for j in range(i,self.d):
                wij = mat[i,j] + self.L
                mij = self.R1d(wij)
                M[i,j] = mij
        return M
    
    def S_vector(self, S_input): # shuffler (vector)
        Y = np.zeros(self.d)
        for k in range(self.d):
            for b in range(self.BatchSize):
                Y[k] += S_input[b,k]
        return Y

    def S_matrix(self, S_input): # shuffler (matrix)
        Y = np.zeros([self.d,self.d])
        for i in range(self.d):
            for j in range(i,self.d):
                for b in range(self.BatchSize):
                    Y[i,j] += S_input[b,i,j]
        return Y

    def A_vector(self, Y): # analyzer (vector)
        out = np.zeros(self.d)
        for k in range(self.d):
            zk = self.A1d(Y[k])
            ok = zk -  self.BatchSize * self.L 
            out[k] = ok
        return out

    def A_matrix(self, Y): # analyzer (matrix)
        out = np.zeros([self.d,self.d])
        for i in range(self.d):
            for j in range(i,self.d):
                zij = self.A1d(Y[i,j])
                oij = zij - self.BatchSize * self.L 
                out[i,j] = oij
                out[j,i] = out[i,j]
        return out
    
    @staticmethod
    def print():
        return "SDPLinUCBPVec"
    
if __name__ == "__main__":
    base_dir = os.path.join(".", "Results", "Lin")

    num_runs = 50 # number of parallel runs
    n = 20000 # number of rounds
    K = 100  #  number of arms
    
    np.random.seed(2)
    
    # Algorithms to compare
    algorithms = [
          (LinUCB, {}, "LinUCB"),
          (LDPLinUCB, {}, "LDPLinUCB"),
          (JDPLinUCB, {}, "JDPLinUCB"),
          (SDPLinUCBPAmp, {}, "SDPLinUCBPAmp"),
          (SDPLinUCBPVec, {}, "SDPLinUCBPVec"),    
     ]
   
    # Bernoulli bandit with privacy level 0.2, confidence level 0.1, feature dimension 5, shuffle Batchsize 20
    # Change here for different settings
    environments = [
        
        (LinBandit, {"noise": "bernoulli", "sigma": 0}, 5, 0.2, 0.1, 20, "Bernoulli (d=5,eps=0.2,delta=0.1,B=20)"),
        
    ]

    for env_def in environments:
        env_class, env_params, d, epsilon, delta, BatchSize, env_name = env_def[0], env_def[1],\
                                    env_def[2], env_def[3], env_def[4], env_def[5], env_def[-1]
        print("================== running environment", env_name, "==================")
        
        envs = []
        for run in range(num_runs):           

            # standard d-dimensional basis (with a bias term)
            basis = np.eye(d)
            basis[:, -1] = 1

            # arm features in a unit (d - 2)-sphere
            X = np.random.randn(K, d - 1)
            X /= np.sqrt(np.square(X).sum(axis=1))[:, np.newaxis]
            X = np.hstack((X, np.ones((K, 1))))  # bias term
            X[: basis.shape[0], :] = basis

            # parameter vector in a (d - 2)-sphere with radius 0.5
            theta = np.random.randn(d - 1)
            theta *= 0.5 / np.sqrt(np.square(theta).sum())
            theta = np.append(theta, [0.5])
            
            # create environment
            envs.append(env_class(X, theta, epsilon, delta, BatchSize, seed=run, **env_params))
            print("%3d: %.2f %.2f | " % (envs[-1].best_arm,
                envs[-1].mu.min(), envs[-1].mu.max()), end="")
            if (run + 1) % 10 == 0:
                print()
        
        res_dir = os.path.join(base_dir, env_name)
        os.makedirs(res_dir, exist_ok=True)
        
        for alg_def in algorithms:
            alg_class, alg_params,alg_name = alg_def[0], alg_def[1], alg_def[-1] 
            
            fname = os.path.join(res_dir, alg_name)        
            if os.path.exists(fname):
                print('File exists. Will load saved file. Moving on to the next algorithm')
            else:
                regret, _ = evaluate(alg_class, alg_params, envs, n)                
                cum_regret = regret.cumsum(axis=0)                           
                np.savetxt(fname, cum_regret, delimiter=",")    