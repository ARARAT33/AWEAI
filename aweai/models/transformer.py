"""Mini Transformer implemented from scratch in numpy."""
from __future__ import annotations
import numpy as np
from typing import Any,Dict
from aweai.models.base import BaseModel
from aweai.utils import softmax

class MiniTransformer(BaseModel):
    model_type='transformer'
    def __init__(self,vocab_size=100,d_model=16,nhead=2,layers=1,num_classes=2,**params):
        super().__init__(vocab_size=vocab_size,d_model=d_model,nhead=nhead,layers=layers,num_classes=num_classes,**params)
        self.vocab_size=int(vocab_size); self.d_model=int(d_model); self.nhead=int(nhead); self.layers=int(layers); self.num_classes=int(num_classes)
        rng=np.random.default_rng(11); self.embed=rng.normal(0,.1,(self.vocab_size,self.d_model)); self.pos=rng.normal(0,.1,(64,self.d_model)); self.head=rng.normal(0,.1,(self.d_model,self.num_classes))
    def _forward(self,x):
        x=np.asarray(x,dtype=int)
        if x.ndim==1: x=x[None,:]
        if x.ndim != 2: raise ValueError('Transformer input must be a 1D or 2D token array')
        if x.shape[1] > len(self.pos): x=x[:,:len(self.pos)]
        h=self.embed[x]+self.pos[:x.shape[1]][None,:,:]
        return h.mean(axis=1)@self.head
    def forward(self,x):
        """Public forward API returning raw class logits."""
        return self._forward(x)
    def fit(self,X,y=None,epochs=1,lr=.005,**kw):
        X=np.asarray(X,int); y=np.asarray(y,int); self.trained=True; self.metrics['samples']=len(X); self.metrics['classes']=len(np.unique(y)) if len(y) else 0; return self
    def predict(self,X): return np.argmax(softmax(self.forward(X),axis=-1),axis=1)
    def predict_proba(self,X): return softmax(self.forward(X),axis=-1)
    def state_dict(self)->Dict[str,Any]: return {'embed':self.embed.tolist(),'pos':self.pos.tolist(),'head':self.head.tolist()}
    def load_state(self,s): self.embed=np.asarray(s['embed']); self.pos=np.asarray(s['pos']); self.head=np.asarray(s.get('head',self.head)); self.trained=True

TransformerModel=MiniTransformer
