class reinsurance_price:
    def __init__(self,S=100, vol=0.3, r=0, q=0, ts=np.linspace(0, 10, int(np.round(10*12))+1), npaths=5000):
        self.S=S
        self.vol=vol
        self.r=r
        self.q=q
        self.ts=ts
        self.npaths=npaths
        self.paths=self.blackscholes_mc(S,vol,r,q,ts,npaths)                #generate mc paths upon initialization

    def blackscholes_mc(self,S,vol,r,q,ts,npaths):
        nsteps = len(ts) - 1
        ts = np.asfarray(ts)[:, np.newaxis]
        W = np.cumsum(np.vstack((np.zeros((1, npaths), dtype=np.float64),
                                np.random.randn(nsteps, npaths) * np.sqrt(np.diff(ts, axis=0)))),axis=0)
        paths = np.exp(-0.5*vol**2*ts + vol*W)*S*np.exp((r-q)*ts)
        return paths

    def exponential_samples(self,lambd):
        return -np.log(np.random.rand(self.npaths))/lambd

    def compare_tau_and_T(self,tau,T):
        return (tau>T,tau<=T)
    
    def u_M(self,K_mat,X):
        return np.maximum(K_mat-X,0)

    def u_D(self,K_D,X):
        return np.maximum(K_D-X,0)
    
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def direct_simulation(self,lambd=0.025,T=10,K_mat=90,K_D=100,alpha=3):
        tau_D=self.exponential_samples(lambd)
        ind,other_ind=self.compare_tau_and_T(tau_D,T)                                       #returns two lists of length npaths, boolean.
        payoff_M=np.sum(self.u_M(K_mat,self.paths[-1][ind])-alpha*np.array(self.ts[-1]))    #payoff if no default event until maturity, undiscounted
        other_ind=np.argwhere(other_ind)                                                    #pick index of True values
        ts=[self.find_nearest(np.arange(len(self.ts)),tau_D[x]) for x in other_ind]         #find nearest time step since generated tau_D are not discrete.
        price_list=self.paths[ts][np.arange(len(self.paths[ts])),ts]                        #find monte carlo price based on time steps found above
        tau_list=np.array([self.ts[t] for t in ts])                                         #a list of default times.
        payoff_D=np.sum((self.u_D(K_D,price_list)-alpha*tau_list)**np.exp(-self.r*ts[-1]))  #payoff if default occur before T, discounted
        return (payoff_M**np.exp(-self.r*ts[-1])+payoff_D)/self.npaths
    
    def averaging_simulation(self,lambd=0.025,K_mat=90,K_D=100,alpha=3):
        delta=self.ts[1]-self.ts[0]                                                         #constant time step
        cashflow=self.u_M(K_mat,self.paths[-1])
        for i in range(len(self.ts)-2,-1,-1):
            cashflow=np.exp(-lambd*delta)*cashflow-alpha*delta+(1-np.exp(-lambd*delta))*self.u_D(K_D,self.paths[i+1])
        return np.mean(cashflow)
