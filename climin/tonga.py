# -*- coding: utf-8 -*-

import scipy
import numpy as np
import scipy.linalg
import scipy.optimize
import itertools
import math
from scipy.sparse.linalg import eigsh

from base import Minimizer, repeat_or_iter


class tonga(Minimizer):

    def __init__(self, wrt, fprime, damping, blocksizes,
                 gamma=0.995, b=50, k=5, nb_estimates=50, 
                 args=None, cov_args=None, logfunc=None):
        
        super(tonga, self).__init__(
            wrt, args=args, logfunc=logfunc)

        self.cov_args = cov_args if cov_args is not None else self.args

        self.fprime = fprime
        self.damping = damping
        self.blocksizes = blocksizes
        self.gamma = gamma
        self.gamma_sqrt = math.sqrt(self.gamma)
        self.b = b #TODO find a meaningful name
        self.k = k
        self.nb_estimates = nb_estimates

          

    #@profile   
    def __iter__(self):
        X_m1 = np.zeros((1, self.wrt.size))
        oldGrad = np.zeros((self.b -1, self.wrt.size))
        step = scipy.empty(self.wrt.size)        
                
        for i, (args, kwargs) in enumerate(self.args):            
            offset = 0
            if (i==0):
                gradient_mean = self.fprime(self.wrt, *args, **kwargs)
                X = scipy.empty((self.wrt.size,1)) 
                X[:,0] = gradient_mean

                ## batches = [self.cov_args.next() for _ in range(self.nb_estimates)]
                ## gradient = [self.fprime(self.wrt, *cov_args, **cov_kwargs) for (cov_args, cov_kwargs) in batches]
                ## gradient = np.array(gradient)
                ## gradient_mean = gradient.mean(axis=0)
                
                ## X = scipy.empty((self.wrt.size, self.k + self.b)) 


            elif ((i%self.b)!=0):
                gradient_mean = self.fprime(self.wrt, *args, **kwargs)
                X = scipy.empty((self.wrt.size, X_m1[0].size + 1))
                X[:,:-1] = X_m1 * self.gamma_sqrt
                X[:,-1] = gradient_mean
            else:
                batches = [self.cov_args.next() for _ in range(self.nb_estimates)]
                gradient = [self.fprime(self.wrt, *cov_args, **cov_kwargs) for (cov_args, cov_kwargs) in batches]
                gradient = np.array(gradient)
                gradient_mean = gradient.mean(axis=0)
                
                X = scipy.empty((self.wrt.size, self.k + self.b)) 

           
            length = len(X[0])
            for size in self.blocksizes:
                
                                
                if (i%self.b==0) and (i>0):
                    #computing gradients
                    grad = gradient[:, offset:offset+size]                    
                    
                    # using old gradients to estimate the new X
                    factor = [self.gamma_sqrt**power for power in range(self.b-1,0,-1)]
                    X[offset:offset+size, self.k:self.k+self.b-1] = factor * oldGrad.T[offset:offset+size]
                    
                    X[offset:offset+size, self.k+self.b-1] = self.gamma_sqrt**(self.b+i) * gradient_mean[offset:offset+size]

                    #eigenvalues decomposition
                    covariance = scipy.dot(grad.T, grad)
                    try:
                        _,  X[offset:offset+size, :self.k] = eigsh(covariance, k=self.k, ncv=(2*self.k+2))
                    except scipy.sparse.linalg.ArpackError :
                        print ("ARPACK error 3: No shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV")
                        fileLog = open('logError', 'a')
                        for j in range(grad.len):
                            for l in range(grad.len):
                                fileLog.write(str(covariance[i,j]))
                                fileLog.write('\n')
                        fileLog.close()     
                        X[offset:offset+size, :self.k] = np.zeros((size, self.k))

                x = X[offset:offset+size]
                
                
                #step[offset:offset+size] = scipy.dot(x, scipy.linalg.inv(scipy.dot(x.T,x)+ self.damping*scipy.eye(length)))[:,-1]
                step[offset:offset+size] = (scipy.linalg.solve((scipy.dot(x.T,x)+ self.damping*scipy.eye(length)).T, x.T, sym_pos=True, overwrite_a=True))[-1]
                offset += size

                

            #storing the old gradients
            if ((i%self.b)==0) and (i>0):
                oldGrad = np.zeros((self.b -1, self.wrt.size))
            else:
                oldGrad[i%self.b-1] = gradient_mean
                
            self.wrt -= step
            X_m1 = X

            
            
            yield {
                'gradient':gradient_mean,
                'args':args,
                'kwargs':kwargs,
                'n_iter':i,
                'step':step,
            }
        

            