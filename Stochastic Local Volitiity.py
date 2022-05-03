class SLV:
    def __init__(self,K_list=[100],S_0=100,timestep=100,T=1,sigma_0=0.15,Y_0=0,rho=-0.5,gamma=0.5,n_paths=10000,nnewpaths=100000,kappa=1,get_price=False,get_vol=False,get_l=False):
        self.K_list=K_list
        self.S_0=S_0
        self.sigma_0=sigma_0
        self.Y_0=Y_0
        self.rho=rho
        self.gamma=gamma
        self.kappa=kappa
        self.n_paths=n_paths
        self.ts=np.linspace(0,T,int(np.round(T*timestep))+1)
        self.nnewpaths=nnewpaths
        self.get_price=get_price
        self.get_vol=get_vol
        self.get_l=get_l
        #self.l_full=np.empty((len(self.ts)-1,self.nnewpaths))
        self.l_full=[]
        self.a_grid=self._mc()
        if self.get_price:                                                                                       #independent run
            self.fullpaths,self.l=self._simulate()
            self.path=self.fullpaths[-1]
            self.price=[np.mean(np.maximum(0,self.path-strike)) for strike in K_list]                            #get the call prices
            self.implied_vol=list(map(lambda x,y: self.blackscholes_impv_scalar(x,1,100,y),self.K_list,self.price))     #get implied vol from call prices

        
    def blackscholes_impv_scalar(self,K, T, S, value, r=0, q=0, callput='call', tol=1e-6, maxiter=500):
        if (K <= 0) or (T <= 0):
            return np.nan
        F = S*np.exp((r-q)*T)
        K = K/F
        value = value*np.exp(r*T)/F
        callput = callput.lower()
        if callput not in ['call', 'put']:
            raise ValueError('The value of "callput" must be either "call" or "put"')
        opttype = 1 if callput == 'call' else -1
        value -= max(opttype * (1 - K), 0)
        if value < 0:
            return np.nan
        if (value == 0):
            return 0
        j = 1
        p = np.log(K)
        if K >= 1:
            x0 = np.sqrt(2 * p)
            x1 = x0 - (0.5 - K * norm.cdf(-x0) - value) * np.sqrt(2*np.pi)
            while (abs(x0 - x1) > tol*np.sqrt(T)) and (j < maxiter):
                x0 = x1
                d1 = -p/x1+0.5*x1
                x1 = x1 - (norm.cdf(d1) - K*norm.cdf(d1-x1)-value)*np.sqrt(2*np.pi)*np.exp(0.5*d1**2)
                j += 1
            return x1 / np.sqrt(T)
        else:
            x0 = np.sqrt(-2 * p)
            x1 = x0 - (0.5*K-norm.cdf(-x0)-value)*np.sqrt(2*np.pi)/K
            while (abs(x0-x1) > tol*np.sqrt(T)) and (j < maxiter):
                x0 = x1
                d1 = -p/x1+0.5*x1
                x1 = x1-(K*norm.cdf(x1-d1)-norm.cdf(-d1)-value)*np.sqrt(2*np.pi)*np.exp(0.5*d1**2)
                j += 1
            return x1/np.sqrt(T)

    def quartic_kernel(self,x,bandwidth):
        x/=bandwidth
        x = np.clip(x, -1, 1)
        return (x+1)**2*(1-x)**2

    def _a(self,i,j,a,paths,paths_knots,bandwidth):
        numerator=np.sum(a**2*self.quartic_kernel(paths[i+1]-paths_knots[j],bandwidth))
        denominator=np.sum(self.quartic_kernel(paths[i+1]-paths_knots[j],bandwidth))
        return np.sqrt(numerator/denominator) if denominator!=0 else 0

    def _mc(self,paths_knots=np.linspace(63,159,41)):
        paths,Y = np.full((len(self.ts), self.n_paths), np.nan, dtype=np.float64),np.full((len(self.ts), self.n_paths), np.nan, dtype=np.float64)
        paths[0],Y[0] = self.S_0,self.Y_0
        a_grid=np.full((len(self.ts),len(paths_knots)),0, dtype=np.float64)
        l=np.ones(self.n_paths)
        for i in range(len(self.ts)-1):
            dt = self.ts[i+1] - self.ts[i]
            Z1,Z2 = np.random.randn(self.n_paths),np.random.randn(self.n_paths)
            Y[i+1]=Y[i]*np.exp(-self.kappa*dt)+self.gamma*np.sqrt((1-np.exp(-2*self.kappa*dt))/(2*self.kappa))*Z2
            a=self.sigma_0*np.exp(Y[i+1])
            rho_bar=self.rho*np.sqrt(2*(1-np.exp(-self.kappa*dt))/(self.kappa*dt*(1+np.exp(-self.kappa*dt))))
            paths[i+1]=paths[i]*np.exp(-0.5*(self.sigma_0**2)*np.exp(2*Y[i])*(l**2)*dt+self.sigma_0*np.exp(Y[i])*l*np.sqrt(dt)*(np.sqrt(1-rho_bar**2)*Z1+rho_bar*Z2))
            bandwidth=1.5*self.S_0*self.sigma_0*np.sqrt(np.maximum(self.ts[i+1],0.15))*self.n_paths**(-0.2)
            for j in range(len(paths_knots)):
                a_grid[i+1,j]=self._a(i,j,a,paths,paths_knots,bandwidth)
            interpolate_a=interp1d(paths_knots,a_grid[i+1],kind="cubic",fill_value="extrapolate")
            l=self.sigma_0/interpolate_a(paths[i+1])
        return a_grid
    
    def _simulate(self,paths_knots=np.linspace(63,159,41)):
        paths,Y = np.full((len(self.ts), self.nnewpaths), np.nan, dtype=np.float64),np.full((len(self.ts), self.nnewpaths), np.nan, dtype=np.float64)
        paths[0],Y[0] = self.S_0,self.Y_0
        l=np.ones(self.nnewpaths)
        for i in range(len(self.ts)-1):
            dt = self.ts[i+1] - self.ts[i]
            Z1,Z2 = np.random.randn(self.nnewpaths),np.random.randn(self.nnewpaths)
            Y[i+1]=Y[i]*np.exp(-self.kappa*dt)+self.gamma*np.sqrt((1-np.exp(-2*self.kappa*dt))/(2*self.kappa))*Z2
            rho_bar=self.rho*np.sqrt(2*(1-np.exp(-self.kappa*dt))/(self.kappa*dt*(1+np.exp(-self.kappa*dt))))
            paths[i+1]=paths[i]*np.exp(-0.5*(self.sigma_0**2)*np.exp(2*Y[i])*(l**2)*dt+self.sigma_0*np.exp(Y[i])*l*np.sqrt(dt)*(np.sqrt(1-rho_bar**2)*Z1+rho_bar*Z2))
            interpolate_a=interp1d(paths_knots,self.a_grid[i+1],kind="cubic",fill_value="extrapolate")
            l=self.sigma_0/interpolate_a(paths[i+1])
            if self.get_l:
                self.l_full.append(interp1d(paths_knots,self.a_grid[i+1],kind="cubic",fill_value="extrapolate"))
        return paths,interpolate_a(paths[i+1])
    
    def get_leverage(self,at_i=2):
        if at_i<=1:
            raise ValueError("at_i must be greater than 0")
        elif not self.get_l:
            raise ValueError("need to set get_l=True")
        return (self.fullpaths[at_i],self.l_full[at_i-1])      #leverage at t=0 is not stored
    
    def get_vol(self):
        return self.implied_vol

    def __repr__(self):
        if self.get_price:
            return "Option values are {}".format(self.price)+"\n"+"The volitilities are {}".format(self.implied_vol)
               
