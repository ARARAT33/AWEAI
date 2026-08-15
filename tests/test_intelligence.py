from aweai.intelligence import (
    AWEIAdaptiveRouter, AWEAIConsistencyEngine, AWEAIFrontierOptimizer,
    AWEAIProvenanceChain, AWEAIWorkloadPlanner,
)


def test_intelligence_primitives():
    ranked=AWEIAdaptiveRouter().rank({'fast':{'quality':.8,'reliability':.9},'slow':{'quality':1.0,'reliability':.7}}, {'quality':1})
    assert ranked[0].name in {'fast','slow'}
    assert AWEAIWorkloadPlanner().waves({'a':[],'b':['a'],'c':['a']}) == [['a'],['b','c']]
    chain=AWEAIProvenanceChain(); h=chain.append({'artifact':'model','version':1}); assert len(h)==64 and chain.head==h
    best,score=AWEAIFrontierOptimizer().search({'x':[1,2,3]},lambda p: -(p['x']-3)**2)
    assert best['x']==3 and score==0
    assert 0 < AWEAIConsistencyEngine().score([{'loss':1},{'loss':1}]) <= 1
