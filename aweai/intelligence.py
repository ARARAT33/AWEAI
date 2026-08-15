"""AWEAI-native intelligence primitives.

These are deterministic engineering algorithms, not chat/agent features.
They provide a reusable substrate for model selection, workload planning,
provenance and adaptive execution inside AWEAI.
"""
from __future__ import annotations
import hashlib, json, math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _hash(obj) -> str:
    raw=json.dumps(obj,sort_keys=True,separators=(",",":"),default=str).encode()
    return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class CapabilityScore:
    name:str
    score:float
    reasons:Tuple[str,...]=()

class AWEIAdaptiveRouter:
    """Capability-aware routing using utility, risk and historical feedback."""
    def rank(self, candidates:Mapping[str,Mapping[str,float]], required:Mapping[str,float], history:Mapping[str,float]|None=None) -> List[CapabilityScore]:
        history=history or {}; out=[]
        for name,meta in candidates.items():
            utility=sum(float(required.get(k,0))*float(meta.get(k,0)) for k in required)
            risk=float(meta.get('risk',0)); cost=float(meta.get('cost',0)); reliability=float(meta.get('reliability',1))
            score=(utility*(0.5+0.5*reliability)+float(history.get(name,0)))-0.35*risk-0.15*cost
            out.append(CapabilityScore(name,score,(f'utility={utility:.4f}',f'reliability={reliability:.3f}')))
        return sorted(out,key=lambda x:x.score,reverse=True)

class AWEIWorkloadPlanner:
    """Turns a dependency DAG into parallel execution waves without an agent layer."""
    def waves(self, nodes:Mapping[str,Sequence[str]]) -> List[List[str]]:
        deps={k:set(v) for k,v in nodes.items()}; waves=[]
        while deps:
            ready=sorted(k for k,v in deps.items() if not v)
            if not ready: raise ValueError('cyclic workload graph')
            waves.append(ready)
            for k in ready: deps.pop(k)
            for v in deps.values(): v.difference_update(ready)
        return waves

class AWEAIProvenanceChain:
    """Tamper-evident content-addressed chain for datasets/models/results."""
    def __init__(self): self._last='0'*64
    def append(self,event:Mapping) -> str:
        record={'previous':self._last,'event':dict(event)}; self._last=_hash(record); return self._last
    @property
    def head(self)->str: return self._last

class AWEAIFrontierOptimizer:
    """Small derivative-free optimizer for choosing engineering configurations."""
    def search(self, dimensions:Mapping[str,Sequence[float]], objective, rounds:int=3) -> Tuple[Dict[str,float],float]:
        current={k:float(v[len(v)//2]) for k,v in dimensions.items()}; best=float(objective(current))
        for _ in range(max(1,rounds)):
            for k,values in dimensions.items():
                for value in values:
                    trial=dict(current); trial[k]=float(value); score=float(objective(trial))
                    if score>best: current,best=trial,score
        return current,best

class AWEAIConsistencyEngine:
    """Detects contradictory metrics and returns a deterministic consistency score."""
    def score(self, observations:Iterable[Mapping[str,float]]) -> float:
        rows=list(observations)
        if not rows:return 1.0
        keys=set().union(*(r.keys() for r in rows)); penalties=0
        for k in keys:
            vals=[float(r[k]) for r in rows if k in r]
            if len(vals)>1:
                mean=sum(vals)/len(vals); var=sum((x-mean)**2 for x in vals)/len(vals)
                penalties += var/(1+abs(mean))
        return 1.0/(1.0+penalties)

__all__=['CapabilityScore','AWEIAdaptiveRouter','AWEAIWorkloadPlanner','AWEAIProvenanceChain','AWEAIFrontierOptimizer','AWEAIConsistencyEngine']
