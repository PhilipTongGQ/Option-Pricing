import tensorflow as tf
from tensorflow import keras

class neural_fit(Bermuda_pricing):
    def __init__(self):
        super().__init__(Bermudan_Asian=True,type="call")
        self.fitted=[]
    
    def backward(self):  
        ts=self.ts
        for j in range(len(ts)-2, 0, -1):
            X_train_full = paths[j]
            # normalize the inputs
            mX = np.mean(X_train_full)
            sX = np.std(X_train_full)
            X_train_full = ((X_train_full - mX) / sX)[:, np.newaxis]
            Y_train_full = (np.maximum(paths[-1]-K, 0)*np.exp(-r*(ts[-1]-ts[j])))[:, np.newaxis]
            train_size = int(len(X_train_full)*0.75)
            X_train = X_train_full[:train_size]
            X_valid = X_train_full[train_size:]
            Y_train = Y_train_full[:train_size]
            Y_valid = Y_train_full[train_size:]
            model = keras.models.Sequential([
                keras.layers.Dense(20, activation='relu', input_shape=[1]),
                keras.layers.Dense(20, activation='relu'),
                keras.layers.Dense(20, activation='relu'),
                keras.layers.Dense(1)])
            model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.001))
            early_stopping_cb = keras.callbacks.EarlyStopping(patience=4, min_delta=1e-3, restore_best_weights=True)
            model.fit(X_train, Y_train, epochs=50, batch_size=128, validation_data=(X_valid, Y_valid), verbose=False, callbacks=[early_stopping_cb])
            self.fitted.append(model)

    def forward(self,n_paths=50000,vol=0.2,q=0.02):
        coef_list=self.fitted.reverse()
        self.backward()
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
                final_list=[memory_list[i] if exercise[i] is True else final_exe[i] for i in range(n_paths)]
                return np.mean([final_list[i]*np.exp(-r*(exercise_time[i])) for i in range(n_paths)])
            mX=np.mean(paths[i+1])
            sX=np.std(paths[i+1])
            payoff=coef_list[i].predict((paths[i+1]-mX)/sX)[:, 0]
            exercise=np.maximum(self.K-paths[i+1],0) if self.type=="put" else np.maximum(paths[i+1]-self.K,0)
            exe=payoff<=exercise  #exercise
            for j in range(n_paths):
                if (exercised[j] is False and exe[j] is True):
                    memory_list[j]=exercise[j]
                    exercise_time[j]=self.ts[i+1]
                    exercised[j]=True
