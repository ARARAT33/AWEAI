"""Simple RNN/LSTM implementations from scratch in numpy."""

from __future__ import annotations
import numpy as np
from aweai.models.base import BaseModel

class RNN(BaseModel):
    model_type = "rnn"
    def __init__(self, input_dim=1, hidden=16, output_dim=1, layers=1, seq_len=4, **params):
        super().__init__(input_dim=input_dim, hidden=hidden, output_dim=output_dim, layers=layers, seq_len=seq_len, **params)
        self.input_dim=int(input_dim); self.hidden=int(hidden); self.output_dim=int(output_dim)
        self.Wxh=np.random.randn(self.input_dim,self.hidden)*0.1; self.Whh=np.random.randn(self.hidden,self.hidden)*0.1
        self.bh=np.zeros(self.hidden); self.Why=np.random.randn(self.hidden,self.output_dim)*0.1; self.by=np.zeros(self.output_dim)
    def fit(self,X,y=None,epochs=30,lr=.01,**kw):
        X=np.asarray(X,float); y=None if y is None else np.asarray(y,float); epochs=int(kw.get('epochs',epochs)); lr=float(kw.get('lr',lr))
        if X.ndim == 1: X=X[None,:]
        for _ in range(epochs):
            loss=0.0
            for i in range(len(X)):
                x=X[i]; x=x.reshape(-1,self.input_dim) if x.ndim==1 else x; h=np.zeros(self.hidden); hs=[h]
                for t in range(len(x)): h=np.tanh(x[t]@self.Wxh+h@self.Whh+self.bh); hs.append(h)
                pred=(hs[-1]@self.Why+self.by).reshape(1,-1)[:,:self.output_dim]
                target=np.asarray(y[i] if y is not None else np.zeros(self.output_dim),float).reshape(1,-1)[:,:self.output_dim]
                d=2*(pred-target)/max(target.size,1); loss+=float(np.mean((pred-target)**2)); dh=(d@self.Why.T).ravel()
                dWxh=np.zeros_like(self.Wxh); dWhh=np.zeros_like(self.Whh); db=np.zeros_like(self.bh)
                for t in reversed(range(len(x))):
                    dt=dh*(1-hs[t+1]**2); dWxh+=np.outer(x[t],dt); dWhh+=np.outer(hs[t],dt); db+=dt; dh=dt@self.Whh.T
                self.Wxh-=lr*dWxh; self.Whh-=lr*dWhh; self.bh-=lr*db; self.Why-=lr*np.outer(hs[-1],d.ravel()); self.by-=lr*d.ravel()
            self.history['loss'].append(loss/max(len(X),1))
        self.trained=True; self.metrics['final_loss']=float(self.history['loss'][-1]); return self
    def train_step(self, x, target, lr=0.01):
        """Compatibility single-sample training API."""
        return self.fit(np.asarray(x,float)[None,:], np.asarray([target]), epochs=1, lr=lr)
    def predict(self,X):
        """Predict a scalar class for multi-output classifiers, or values for regression."""
        arr=np.asarray(X,float)
        if arr.ndim==1: arr=arr[None,:]
        out=[]
        for x in arr:
            x=x.reshape(-1,self.input_dim) if x.ndim==1 else x; h=np.zeros(self.hidden)
            for t in range(len(x)): h=np.tanh(x[t]@self.Wxh+h@self.Whh+self.bh)
            value=(h@self.Why+self.by).ravel()
            if self.output_dim > 1:
                out.append(int(np.argmax(value)))
            else:
                out.append(float(value[0]))
        if len(out) == 1:
            return out[0]
        return np.asarray(out)
    def state_dict(self): return {'Wxh':self.Wxh.tolist(),'Whh':self.Whh.tolist(),'bh':self.bh.tolist(),'Why':self.Why.tolist(),'by':self.by.tolist()}
    def load_state(self,s):
        self.Wxh=np.asarray(s['Wxh']); self.Whh=np.asarray(s['Whh']); self.bh=np.asarray(s['bh']); self.Why=np.asarray(s['Why']); self.by=np.asarray(s['by']); self.trained=True

class LSTM(RNN):
    """Compact sequence model; public API-compatible with the factory."""
    model_type='lstm'

RNNModel = RNN
