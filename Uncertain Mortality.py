class uncertain_mortality_price(reinsurance_price):
    def __init__(self,npaths=50000):
        self.npaths=npaths
        super().__init__(npaths=self.npaths)
        self.coef_list=[]
    
    def pwlinear_basis(self,xknots):
        fs = [lambda x: np.ones_like(x, dtype=np.float64), lambda x: x-xknots[0]]
        fs.extend([lambda x, a=xknots[i]: np.maximum(x-a, 0) for i in range(len(xknots))])
        return fs

    def pwlinear_fit(self,xdata, ydata, xknots):
        fs = self.pwlinear_basis(xknots)
        A = np.vstack([f(xdata) for f in fs]).T
        ps = np.linalg.lstsq(A, ydata, rcond=None)[0]
        return ps, fs

    def piece_wise_pred(self,xknots,xdata,ydata,lower_bound,ps_fs_combo=None):                  #we have Y at t_i and X at t_(i-1)
        if lower_bound:
            ps,fs=self.pwlinear_fit(xdata,ydata,xknots)
            return (sum([f(xdata)*p for (f, p) in zip(fs, ps)]),(ps,fs))
        else:
            return sum([f(xdata)*p for (f, p) in zip(ps_fs_combo[1],ps_fs_combo[0])])  #note the order here is reversed

    def basic_BSDE(self,lambdD_min=0.005,lambdD_max=0.04,K_mat=90,K_D=100,alpha=3,nnewpaths=100000):  #a BSDE implementation
        Y=self.u_M(K_mat,self.paths[-1])
        delta=self.ts[1]-self.ts[0]                                                  #constant time step
        for i in range(len(self.ts)-2,-1,-1):                                        #loop backwards    
            E,coef=self.piece_wise_pred(np.linspace(np.percentile(self.paths[i], 1), np.percentile(self.paths[i], 99), 10),self.paths[i],Y,lower_bound=True)
            self.coef_list.append(coef)
            u_d=self.u_D(K_D,self.paths[i])
            ind=u_d>=E-alpha*delta
            not_ind=list(~np.array(ind))
            Y[ind]=(E[ind]-alpha*delta+u_d[ind]*lambdD_max*delta)/(1+lambdD_max*delta)
            Y[not_ind]=(E[not_ind]-alpha*delta+u_d[not_ind]*lambdD_min*delta)/(1+lambdD_min*delta)
        self.coef_list=self.coef_list[::-1]                                     #coefficients saved in previous loop is in reverse order
        new_paths=self.blackscholes_mc(self.S,self.vol,self.r,self.q,self.ts,nnewpaths)  #simulate independent paths
        Y=self.u_M(K_mat,new_paths[-1])
        for i in range(len(self.ts)-2,-1,-1):
            E=self.piece_wise_pred(None,new_paths[i],Y,lower_bound=False,ps_fs_combo=self.coef_list[i])
            u_d=self.u_D(K_D,new_paths[i])
            ind=u_d>=E-alpha*delta
            not_ind=list(~np.array(ind))
            Y[ind]=(E[ind]-alpha*delta+u_d[ind]*lambdD_max*delta)/(1+lambdD_max*delta)
            Y[not_ind]=(E[not_ind]-alpha*delta+u_d[not_ind]*lambdD_min*delta)/(1+lambdD_min*delta)
        return np.mean(Y)
    
    def LS_price(self,lambdD_min=0.005,lambdD_max=0.04,K_mat=90,K_D=100,alpha=3,nnewpaths=100000):       #a Longstaff-Schwartz implementation
        self.coef_list=[]                           #reset to default
        V = self.u_M(K_mat,self.paths[-1])
        lambd = np.where(self.u_D(K_D,self.paths[-1]) >= V, lambdD_max, lambdD_min)
        for i in range(len(self.ts)-2, -1, -1):                                        #range(len(self.ts)-2, 0, -1) also works, has no effect on results since last coefficients
            #estimated at t=0 won't be used in calculating final np.mean(V) anyway
            dt = self.ts[i+1]-self.ts[i]
            p = 1-np.exp(-lambd*dt)
            V = p*self.u_D(K_D,self.paths[i+1]) + (1-p)*V - alpha*dt
            E,coef=self.piece_wise_pred(np.linspace(np.percentile(self.paths[i], 1), np.percentile(self.paths[i], 99), 10),self.paths[i],V,lower_bound=True)
            self.coef_list.append(coef)
            lambd = np.where(self.u_D(K_D,self.paths[i]) >= E, lambdD_max, lambdD_min)
        paths = self.blackscholes_mc(self.S,self.vol,self.r,self.q,self.ts,nnewpaths)  #simulate independent paths
        self.coef_list=self.coef_list[::-1]                                              #coefficients saved in previous loops are in reverse order
        V = self.u_M(K_mat,paths[-1])
        lambd = np.where(self.u_D(K_D,paths[-1]) >= V, lambdD_max, lambdD_min)
        for i in range(len(self.ts)-2, -1, -1):
            dt = self.ts[i+1]-self.ts[i]
            p = 1-np.exp(-lambd*dt)
            V = p*self.u_D(K_D,paths[i+1]) + (1-p)*V - alpha*dt
            E=self.piece_wise_pred(None,paths[i],V,lower_bound=False,ps_fs_combo=self.coef_list[i])   #third parameter is not used
            lambd = np.where(self.u_D(K_D,paths[i]) >= E, lambdD_max, lambdD_min)
        return np.mean(V)
