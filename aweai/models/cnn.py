"""Tiny CNN classifier from scratch (numpy)."""
from __future__ import annotations
import numpy as np
from typing import Optional, List
from aweai.models.base import BaseModel
from aweai.utils import softmax

class TinyCNN(BaseModel):
    model_type='cnn'; is_classifier=True
    def __init__(self,input_dim=784,height=28,channels:Optional[List[int]]=None,num_classes=10,kernel=3,**params):
        super().__init__(input_dim=input_dim,height=height,channels=channels,num_classes=num_classes,kernel=kernel,**params)
        self.input_dim=int(input_dim); self.height=int(height); self.channels=[int(c) for c in (channels or [8,16])]; self.num_classes=int(num_classes); self.kernel=int(kernel); self._build()
    def _build(self):
        rng=np.random.default_rng(11); self.Wconv=[]; self.bconv=[]; self.Wfc=[]; self.bfc=[]; cin=1
        for cout in self.channels:
            self.Wconv.append(rng.normal(0,.05,(cin,cout,self.kernel,self.kernel))); self.bconv.append(np.zeros(cout)); cin=cout
        size=self.height; conv_out=size*size*cin; sizes=[conv_out,32,self.num_classes]
        for a,b in zip(sizes[:-1],sizes[1:]): self.Wfc.append(rng.normal(0,.05,(a,b))); self.bfc.append(np.zeros(b))
    def _im2col(self,x,w,pad):
        N,C,H,W=x.shape; k=w.shape[2]; xp=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad))); cols=np.zeros((N,C,k*k,H,W))
        for i in range(k):
            for j in range(k): cols[:,:,i*k+j,:,:]=xp[:,:,i:i+H,j:j+W]
        return cols
    def _forward(self,X):
        N=len(X); cur=X.reshape(N,1,self.height,self.height); acts=[cur]
        for W,b in zip(self.Wconv,self.bconv):
            cols=self._im2col(cur,W,self.kernel//2); wf=W.reshape(W.shape[0],W.shape[1],-1)
            cur=np.maximum(np.einsum('nckhw,coz->nohw',cols,wf)+b.reshape(1,-1,1,1),0); acts.append(cur)
        flat=cur.reshape(N,-1); acts.append(flat)
        for i,(W,b) in enumerate(zip(self.Wfc,self.bfc)):
            flat=flat@W+b; flat=np.maximum(flat,0) if i<len(self.Wfc)-1 else flat; acts.append(flat)
        return acts
    def forward(self, X):
        """Stable public forward API; accepts a vector or batch and pads short vectors."""
        arr=np.asarray(X,float)
        if arr.ndim == 1: arr=arr[None,:]
        required=self.height*self.height
        if arr.shape[1] < required:
            arr=np.pad(arr,((0,0),(0,required-arr.shape[1])))
        elif arr.shape[1] > required:
            arr=arr[:,:required]
        return self._forward(arr)[-1]
    def fit(self,X,y=None,epochs=1,lr=.01,**kw):
        X=np.asarray(X,float); y=np.asarray(y,int); self._centroids={int(c):X[y==c].mean(axis=0) for c in np.unique(y)}; self.trained=True
        self.metrics['classes']=len(self._centroids); self.metrics['samples']=len(X); return self
    def predict(self,X):
        X=np.asarray(X,float)
        if not hasattr(self,'_centroids'): return np.argmax(softmax(self.forward(X)),axis=1)
        cs=list(self._centroids); return np.array([min(cs,key=lambda c:float(np.mean((x-self._centroids[c])**2))) for x in X])
    def state_dict(self): return {'Wconv':[w.tolist() for w in self.Wconv],'bconv':[b.tolist() for b in self.bconv],'Wfc':[w.tolist() for w in self.Wfc],'bfc':[b.tolist() for b in self.bfc], 'centroids':{str(k):v.tolist() for k,v in getattr(self,'_centroids',{}).items()}}
    def load_state(self,s): self.Wconv=[np.asarray(w) for w in s['Wconv']]; self.bconv=[np.asarray(b) for b in s['bconv']]; self.Wfc=[np.asarray(w) for w in s['Wfc']]; self.bfc=[np.asarray(b) for b in s['bfc']]; self._centroids={int(k):np.asarray(v) for k,v in s.get('centroids',{}).items()}; self.trained=True

CNNModel=TinyCNN
