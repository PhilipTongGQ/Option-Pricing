class UVM:
    def __init__(self,S=100,vol=0.15,r=0,q=0,T=1,timestep=12,npaths=5000,nnewpaths=50000,forward_timestep=360):
        self.vol=vol
        self.S=S
        self.vol=vol
        self.r=r
        self.T=T
        self.q=q
        self.ts=np.linspace(0,T,int(np.round(T*timestep))+1)
        self.f_ts=np.linspace(0,T,int(np.round(T*forward_timestep))+1)
        self.npaths=npaths
        self.nnewpaths=nnewpaths
        self.W_diff=None
        self.paths=self.blackscholes_mc(S,vol,r,q,self.ts,npaths)                #generate mc paths upon initialization
        self.gamma_list=[]

    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx                              #returns index of the nearest value in the array

    def blackscholes_mc(self,S,vol,r,q,ts,npaths):
        nsteps = len(ts) - 1
        ts = np.asfarray(ts)[:, np.newaxis]
        W = np.cumsum(np.vstack((np.zeros((1, npaths), dtype=np.float64),np.random.randn(nsteps, npaths) * np.sqrt(np.diff(ts, axis=0)))),axis=0)
        self.W_diff=W[1:]-W[:-1]
        paths = np.exp(-0.5*vol**2*ts + vol*W)*S*np.exp((r-q)*ts)
        return paths
    
    def blackscholes_gamma_mc(self,S,r,q,ts,n_paths,vol_h,vol_l):
        paths = np.full((len(ts), n_paths), np.nan, dtype=np.float64)
        paths[0] = S
        W=np.full((len(ts)-1,self.nnewpaths),0,dtype=np.float64)
        for i in range(len(ts)-1):
            dt = ts[i+1] - ts[i]
            dW = np.sqrt(dt)*np.random.randn(n_paths)
            time=self.find_nearest(self.ts[:-1],ts[i])          #last in self.ts is not considered
            if self.scheme=="1":
                gamma=self.piece_wise_pred(None,paths[i],None,ps_fs_combo=self.gamma_list[time],pred=True)/(dt*self.vol*paths[i])
            else:
                gamma=self.piece_wise_pred(None,paths[i],None,ps_fs_combo=self.gamma_list[time],pred=True)/(dt*self.vol*paths[i])**2
            sigma=np.where(gamma>=0,vol_h,vol_l)
            W[i]=dW
            paths[i+1] = paths[i] * np.exp((r-q-0.5*sigma**2)*dt + sigma*dW)
        self.W_diff=W
        return paths

    def pwlinear_basis(self,xknots):
        fs = [lambda x: np.ones_like(x, dtype=np.float64), lambda x: x-xknots[0]]
        fs.extend([lambda x, a=xknots[i]: np.maximum(x-a, 0) for i in range(len(xknots))])
        return fs

    def pwlinear_fit(self,xdata, ydata, xknots):
        fs = self.pwlinear_basis(xknots)
        A = np.vstack([f(xdata) for f in fs]).T
        ps = np.linalg.lstsq(A, ydata, rcond=None)[0]
        return ps, fs

    def piece_wise_pred(self,xknots,xdata,ydata,ps_fs_combo=None,save_param=False,pred=False):                  #we have Y at t_i and X at t_(i-1)
        if pred:                                                                    #for prediction
            return sum([f(xdata)*p for (f, p) in zip(ps_fs_combo[1],ps_fs_combo[0])])
        ps,fs=self.pwlinear_fit(xdata,ydata,xknots)
        if not ps_fs_combo and not save_param:                                      
            return sum([f(xdata)*p for (f, p) in zip(fs, ps)])
        return (sum([f(xdata)*p for (f, p) in zip(fs, ps)]),(ps,fs))                    #also return fitted coefficients

    def get_spread(self,X,K1,K2):
        return 100*(np.maximum(X-K1,0)-np.maximum(X-K2,0))/(K2-K1)
    
    def delta(self,X,K1,K2):
        return 100*(np.where(X-K1>0,1,0)-np.where(X-K2>0,1,0))/(K2-K1)
    
    def sigma(self,gamma,vol_l,vol_h):
        return np.where(gamma<0,vol_l,vol_h)

#we have to use non-parametric methods, otherwise would lead to terrible estimation, this is because?
    def call_spread(self,K1=90,K2=110,vol_l=0.1,vol_h=0.2,scheme="1"):
        self.scheme=scheme
        path=self.paths[-1]
        Y=self.get_spread(path,K1,K2)
        Z=self.delta(path,K1,K2)
        for i in range(len(self.ts)-2,-1,-1):       #start from T-1
            delta_t=self.ts[i+1]-self.ts[i]
            W_diff=self.W_diff[i]
            path=self.paths[i]
            if scheme=="1":
                Z=self.piece_wise_pred(np.linspace(np.percentile(path, 1), np.percentile(path, 99), 10),path,W_diff*Y)/(delta_t*self.vol*path)
                gamma,coef=self.piece_wise_pred(np.linspace(np.percentile(path, 1), np.percentile(path, 99), 10),path,W_diff*Z,save_param=True)
                gamma/=(delta_t*self.vol*path)
            elif scheme=="2":
                target=Y*((W_diff**2)-delta_t*(1+self.vol*W_diff))
                gamma,coef=self.piece_wise_pred(np.linspace(np.percentile(path, 1), np.percentile(path, 99), 10),path,target,save_param=True)
                gamma/=(delta_t*self.vol*path)**2
            elif scheme=="2a":
                EY=self.piece_wise_pred(np.linspace(np.percentile(path, 1), np.percentile(path, 99), 10),path,Y)
                target=(Y-EY)*((W_diff**2)-delta_t*(1+self.vol*W_diff))
                gamma,coef=self.piece_wise_pred(np.linspace(np.percentile(path, 1), np.percentile(path, 99), 10),path,target,save_param=True)
                gamma/=(delta_t*self.vol*path)**2
            else:
                raise ValueError("wrong scheme")
            self.gamma_list.append(coef)
            Y=self.piece_wise_pred(np.linspace(np.percentile(path, 1), np.percentile(path, 99), 10),path,Y)+0.5*gamma*delta_t*(path**2)*(self.sigma(gamma,vol_l,vol_h)**2-self.vol**2)
        self.gamma_list=self.gamma_list[::-1]
        self.paths=self.blackscholes_gamma_mc(self.S,self.r,self.q,self.f_ts,self.nnewpaths,vol_h,vol_l)
        return np.mean(self.get_spread(self.paths[-1],K1=90,K2=110))*np.exp(-self.r*self.T)
