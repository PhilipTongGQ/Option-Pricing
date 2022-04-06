class Bermuda_pricing:
    def __init__(self,T=1,K=100,S0=100,ts=np.linspace(0, 1, 13),r=0.1,type="put",Bermudan_Asian=False,vol=0.2): #note the linspace setting has implicit T=1
        self.K=K
        self.T=T
        self.S0=S0
        self.ts=ts
        self.r=r
        self.type=type
        self.vol=vol
        self.coef_list=[]
        self.Bermudan_Asian=Bermudan_Asian
        self.mc_paths=self.generate_mc_paths(Bermudan_Asian=self.Bermudan_Asian)

    def Z(self,n,S): #n should be the index of ts(time step) starting from 1
        A=sum(S[1:(n+1)])   #exclude S0
        return (A+((len(self.ts)-1)-n)*S[n])/(len(self.ts)-1)

    def generate_mc_paths(self,n_paths=10000,q=0.02,Bermudan_Asian=False):
        vol=self.vol
        paths = np.full((len(self.ts), n_paths), np.nan, dtype=np.float64)
        paths[0] = self.S0
        for i in range(len(self.ts)-1):
            dt = self.ts[i+1] - self.ts[i]
            dW = np.sqrt(dt)*np.random.randn(n_paths)
            paths[i+1] = paths[i] * np.exp((self.r-q-1/2*vol**2)*dt + vol*dW)
        back_up=paths.copy()   #create back up paths so that we get noncumulative Bermudan Asian prices for each path.
        if Bermudan_Asian is True:
            for i in range(1,len(self.ts)):
                paths[i]=self.Z(i,back_up)
        if Bermudan_Asian is False:
            paths=back_up
        return paths
    
    def LS_price(self,t=0,mode="bs",method="LS",enhanced=False,plot=False,lower_bound=False):
        if isinstance(t,float):
            raise ValueError("t must be integer position in ts")
        paths=self.mc_paths
        ts=self.ts
        K=self.K
        if self.type=="put":
            payoff = np.maximum(K-paths[-1], 0)
        if self.type=="call":
            payoff = np.maximum(paths[-1]-K, 0)
        for i in range(len(ts)-2, t, -1):
            discount = np.exp(-self.r*(ts[i+1]-ts[i]))
            payoff = payoff*discount
            exerval = np.maximum(K-paths[i], 0)
            if mode=="bs":
                linear=np.polyfit(self.blackscholes_price(S0=paths[i],t=ts[i]),payoff,deg=1)  #the constant 1 is built in as intercept.
                if not enhanced:
                    contval=np.polyval(linear,self.blackscholes_price(S0=paths[i],t=ts[i]))
                else:
                    contval=[np.polyval(linear,self.blackscholes_price(S0=paths[i],t=ts[i])[j]) if exerval[j]!=0 else payoff[j] for j in range(len(exerval))]
                if lower_bound is True:
                    self.coef_list.append(linear)
            if mode=="piece_wise":
                contval=self.piece_wise(paths[i],payoff)
            if mode=="quad_poly":
                p = np.polyfit(paths[i], payoff, deg=2)  #fit all paths regardless enhanced or not
                if not enhanced:
                    contval = np.polyval(p, paths[i])
                else: #if enhanced, not estimate where exerval==0
                    contval = [np.polyval(p,paths[i][j]) if exerval[j]!=0 else payoff[j] for j in range(len(exerval))]
                if lower_bound==True:
                    self.coef_list.append(p)
            if mode=="kernel":
                contval=self.kernel(paths[i],payoff)
            if method=="LS":
                ind = exerval > contval
                payoff[ind] = exerval[ind]
            if method=="TVR":
                payoff = np.maximum(exerval, contval)
        if plot is True:
            return (paths[t],payoff*np.exp(-self.r*(ts[t+1]-ts[t])),p,exerval) if mode=="quad_poly" else (paths[t],payoff*np.exp(-self.r*(ts[t+1]-ts[t])),linear,exerval)
        elif lower_bound is True:
            return self.coef_list
        return np.mean(payoff*np.exp(-self.r*(ts[t+1]-ts[t])))

    def piece_wise(self,X,Y,nknots=5):   #5 knots is good.
        xknots = np.linspace(np.percentile(X, 2.5), np.percentile(X, 97.5), nknots)
        ps, fs = pwlin_fit(X, Y, xknots)
        return sum([f(X)*p for (f, p) in zip(fs, ps)])
    
    def kernel(self,X,Y,nknots=5,bw=0.05): #bandwidth of 0.05 is good.
        xknots0 = np.linspace(np.percentile(X, 2.5), np.percentile(X, 97.5), nknots)
        yknots0=kern_reg(xknots0, X, Y, bw, gauss_kern)
        f0 = interp1d(xknots0, yknots0, kind='linear', fill_value='extrapolate')
        return f0(X)

    def blackscholes_price(self,S0,vol=0.2,t=0, r=0, q=0):
        callput=self.type
        F = S0*np.exp((r-q)*(self.T-t))
        v = vol*np.sqrt(self.T-t)
        d1 = np.log(F/self.K)/v + 0.5*v
        d2 = d1 - v
        try:
            opttype = {'call':1, 'put':-1}[callput.lower()]
        except:
            raise ValueError('The value of callput must be either "call" or "put".')
        price = opttype*(F*norm.cdf(opttype*d1)-self.K*norm.cdf(opttype*d2))*np.exp(-r*(self.T-t))
        return price
    
    def plot(self,t=0,alpha=0.5,mode="bs",method="LS",enhanced=False,plot=True): #can only plot LS algo 
        if mode=="quad_poly":
            X,Y,p,exerval=self.LS_price(t,mode,method,enhanced,plot)
        if mode=="bs":
            X,Y,linear,exerval=self.LS_price(t,mode,method,enhanced,plot)
        fig, ax = plt.subplots()
        ax.scatter(X,Y,color="grey")
        ax.plot(np.sort(X),np.maximum(self.K-np.sort(X), 0),color="red",label="exercise_value")
        if mode=="quad_poly":
            ax.plot(np.sort(X),np.polyval(p,np.sort(X)),label="quad_basis")
            ax.fill_between(np.sort(X),min(Y),max(Y),where=np.polyval(p,np.sort(X))<np.maximum(self.K-np.sort(X), 0),alpha=alpha,color="orange")
            ax.legend()
        if mode=="bs":
            ax.plot(np.sort(X),np.polyval(linear,self.blackscholes_price(S0=np.sort(X),t=ts[t])),label="black_scholes_basis")
            ax.fill_between(np.sort(X),min(Y),max(Y),where=np.polyval(linear,self.blackscholes_price(S0=np.sort(X),t=ts[t]))<np.maximum(self.K-np.sort(X), 0),alpha=alpha,color="pink")
            ax.legend()
    
    def lb_price(self,coef_list,n_paths=100000,vol=0.2,q=0.02,mode="quad_poly"): #compatible with quadratic basis
        exercised=[False]*n_paths
        memory_list=[0]*n_paths
        exercise_time=[self.ts[-1]]*n_paths
        paths = np.full((len(self.ts), n_paths), np.nan, dtype=np.float64)
        paths[0] = self.S0
        back_up=paths.copy()
        for i in range(len(self.ts)-1):
            dt = self.ts[i+1] - self.ts[i]
            dW = np.sqrt(dt)*np.random.randn(n_paths)
            back_up[i+1] = back_up[i] * np.exp((self.r-q-1/2*vol**2)*dt + vol*dW)
            paths[i+1]=self.Z(i+1,back_up) if self.Bermudan_Asian is True else back_up[i+1]
            if i==len(self.ts)-2:
                final_exe=np.maximum(self.K-paths[i+1],0)  if self.type=="put" else np.maximum(paths[i+1]-self.K,0)
                final_list=[memory_list[i] if exercised[i] is True else final_exe[i] for i in range(n_paths)]
                return np.mean([final_list[i]*np.exp(-r*(exercise_time[i])) for i in range(n_paths)])
            payoff=np.polyval(coef_list[i],paths[i+1]) if mode=="quad_poly" else np.polyval(coef_list[i],self.blackscholes_price(S0=paths[i+1],t=ts[i+1]))
            exercise=np.maximum(self.K-paths[i+1],0) if self.type=="put" else np.maximum(paths[i+1]-self.K,0)
            exe=payoff<=exercise  #exercise
            for j in range(n_paths):
                if (exercised[j] is False and exe[j] is True):
                    memory_list[j]=exercise[j]
                    exercise_time[j]=self.ts[i+1]
                    exercised[j]=True
