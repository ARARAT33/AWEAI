"""
CognitiveProcessor - Advanced reasoning and decision-making engine
Handles natural language understanding, planning, and intelligent responses
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..utils.logger import setup_logger


@dataclass
class ThoughtNode:
    """Represents a node in the reasoning graph"""
    id: str
    content: str
    confidence: float
    parent_ids: List[str]
    children_ids: List[str]
    timestamp: datetime


@dataclass
class ReasoningChain:
    """Complete reasoning chain for a decision"""
    nodes: List[ThoughtNode]
    conclusion: str
    confidence: float
    execution_path: List[str]


class CognitiveProcessor:
    """
    Advanced cognitive processing engine
    
    Capabilities:
    - Natural language understanding
    - Multi-step reasoning
    - Context-aware decision making
    - Intent recognition
    - Semantic analysis
    - Knowledge retrieval
    """
    
    def __init__(self, config):
        self.logger = setup_logger("CognitiveProcessor")
        self.config = config
        
        # Knowledge base (set by engine)
        self.knowledge_base = None
        
        # Reasoning state
        self.thought_graph: Dict[str, ThoughtNode] = {}
        self.active_chains: List[ReasoningChain] = []
        self.context_window: List[Dict] = []
        self.max_context_length = 10
        
        # NLP models (lazy loading)
        self.nlp_model = None
        self.embedding_model = None
        
        # Reasoning parameters
        self.reasoning_depth = 5
        self.confidence_threshold = 0.7
        self.creativity_factor = 0.3
        
        # Cache for frequent queries
        self.query_cache: Dict[str, Any] = {}
        self.cache_max_size = 1000
        
        self.logger.info("Cognitive Processor initialized")
    
    async def _load_models(self):
        """Load NLP and embedding models"""
        if self.nlp_model is None:
            try:
                # Try to load spaCy or transformers model
                import spacy
                self.nlp_model = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spaCy NLP model")
            except Exception:
                self.logger.warning("spaCy model not available, using basic processing")
        
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Loaded embedding model")
            except Exception:
                self.logger.warning("Embedding model not available")
    
    async def analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a task to understand requirements and plan execution
        
        Args:
            task: Task dictionary
            
        Returns:
            Analysis results with confidence scores and execution plan
        """
        await self._load_models()
        
        task_description = task.get('description', '')
        task_type = task.get('type', 'generic')
        
        # Extract key information
        entities = await self._extract_entities(task_description)
        intent = await self._classify_intent(task_description)
        complexity = await self._assess_complexity(task_description)
        
        # Generate execution plan
        plan = await self._generate_plan(intent, entities, task_type)
        
        # Calculate confidence
        confidence = self._calculate_confidence(entities, intent, plan)
        
        analysis = {
            'intent': intent,
            'entities': entities,
            'complexity': complexity,
            'plan': plan,
            'confidence': confidence,
            'estimated_time': self._estimate_execution_time(plan),
            'required_skills': self._identify_required_skills(intent)
        }
        
        self.logger.debug(f"Task analysis complete: confidence={confidence:.2f}")
        return analysis
    
    async def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        entities = []
        
        if self.nlp_model:
            doc = self.nlp_model(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        else:
            # Basic entity extraction fallback
            import re
            patterns = {
                'FILE': r'[\w\.\-/]+(?:txt|pdf|docx|py|json|csv)',
                'PATH': r'(?:/[\\w/.-]+)|(?:[A-Z]:\\\\[\\w/.-]+)',
                'NUMBER': r'\\b\\d+(?:\\.\\d+)?\\b',
                'EMAIL': r'[\\w.-]+@[\\w.-]+\\.[\\w]+'
            }
            
            for label, pattern in patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': label,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return entities
    
    async def _classify_intent(self, text: str) -> str:
        """Classify the intent of the input text"""
        text_lower = text.lower()
        
        # Intent classification rules
        intent_patterns = {
            'automation': ['automate', 'script', 'batch', 'repeat', 'schedule'],
            'analysis': ['analyze', 'examine', 'inspect', 'review', 'check'],
            'creation': ['create', 'make', 'generate', 'build', 'write'],
            'modification': ['modify', 'change', 'update', 'edit', 'fix'],
            'deletion': ['delete', 'remove', 'clear', 'erase'],
            'search': ['find', 'search', 'locate', 'look for'],
            'learning': ['learn', 'train', 'improve', 'optimize'],
            'query': ['what', 'how', 'when', 'where', 'why', 'explain']
        }
        
        best_intent = 'generic'
        best_score = 0
        
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent
    
    async def _assess_complexity(self, text: str) -> float:
        """Assess task complexity (0.0 to 1.0)"""
        # Simple complexity metrics
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
        
        # Complexity factors
        avg_sentence_length = word_count / max(sentence_count, 1)
        has_conditions = any(word in text.lower() for word in ['if', 'when', 'unless', 'except'])
        has_iterations = any(word in text.lower() for word in ['loop', 'repeat', 'each', 'all'])
        
        complexity = min(1.0, (
            (avg_sentence_length / 50) * 0.3 +
            (1 if has_conditions else 0) * 0.3 +
            (1 if has_iterations else 0) * 0.3 +
            (word_count / 200) * 0.1
        ))
        
        return complexity
    
    async def _generate_plan(self, intent: str, entities: List[Dict], task_type: str) -> List[Dict]:
        """Generate execution plan based on intent and entities"""
        plan = []
        
        # Standard plan templates
        if intent == 'automation':
            plan = [
                {'step': 1, 'action': 'analyze_requirements', 'description': 'Analyze automation requirements'},
                {'step': 2, 'action': 'identify_patterns', 'description': 'Identify repetitive patterns'},
                {'step': 3, 'action': 'design_workflow', 'description': 'Design automation workflow'},
                {'step': 4, 'action': 'implement_script', 'description': 'Implement automation script'},
                {'step': 5, 'action': 'test_validate', 'description': 'Test and validate automation'}
            ]
        elif intent == 'analysis':
            plan = [
                {'step': 1, 'action': 'collect_data', 'description': 'Collect relevant data'},
                {'step': 2, 'action': 'preprocess', 'description': 'Preprocess and clean data'},
                {'step': 3, 'action': 'apply_methods', 'description': 'Apply analysis methods'},
                {'step': 4, 'action': 'interpret_results', 'description': 'Interpret results'},
                {'step': 5, 'action': 'generate_report', 'description': 'Generate findings report'}
            ]
        elif intent == 'creation':
            plan = [
                {'step': 1, 'action': 'define_specifications', 'description': 'Define specifications'},
                {'step': 2, 'action': 'gather_resources', 'description': 'Gather required resources'},
                {'step': 3, 'action': 'create_content', 'description': 'Create content/structure'},
                {'step': 4, 'action': 'review_refine', 'description': 'Review and refine'},
                {'step': 5, 'action': 'finalize', 'description': 'Finalize output'}
            ]
        else:
            plan = [
                {'step': 1, 'action': 'understand_task', 'description': 'Understand task requirements'},
                {'step': 2, 'action': 'plan_approach', 'description': 'Plan approach'},
                {'step': 3, 'action': 'execute', 'description': 'Execute task'},
                {'step': 4, 'action': 'verify', 'description': 'Verify results'}
            ]
        
        return plan
    
    def _calculate_confidence(self, entities: List, intent: str, plan: List) -> float:
        """Calculate confidence score for the analysis"""
        base_confidence = 0.5
        
        # Entity bonus
        entity_bonus = min(0.2, len(entities) * 0.05)
        
        # Intent clarity bonus
        intent_bonus = 0.15 if intent != 'generic' else 0.0
        
        # Plan completeness bonus
        plan_bonus = min(0.15, len(plan) * 0.03)
        
        confidence = base_confidence + entity_bonus + intent_bonus + plan_bonus
        return min(1.0, confidence)
    
    def _estimate_execution_time(self, plan: List[Dict]) -> float:
        """Estimate execution time in seconds"""
        base_time_per_step = 2.0
        return len(plan) * base_time_per_step
    
    def _identify_required_skills(self, intent: str) -> List[str]:
        """Identify skills required for the task"""
        skill_map = {
            'automation': ['scripting', 'workflow_design', 'error_handling'],
            'analysis': ['data_processing', 'pattern_recognition', 'critical_thinking'],
            'creation': ['content_generation', 'design', 'quality_assurance'],
            'modification': ['editing', 'version_control', 'testing'],
            'search': ['information_retrieval', 'filtering', 'ranking'],
            'learning': ['optimization', 'adaptation', 'evaluation']
        }
        return skill_map.get(intent, ['general_problem_solving'])
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query
        
        Args:
            query: User's question or command
            
        Returns:
            Response with answer and metadata
        """
        await self._load_models()
        
        # Check cache
        cache_key = hash(query)
        if cache_key in self.query_cache:
            self.logger.debug("Cache hit for query")
            return self.query_cache[cache_key]
        
        # Update context
        self.context_window.append({
            'type': 'query',
            'content': query,
            'timestamp': datetime.now()
        })
        if len(self.context_window) > self.max_context_length:
            self.context_window.pop(0)
        
        # Build reasoning chain
        reasoning_chain = await self._build_reasoning_chain(query)
        
        # Retrieve relevant knowledge
        knowledge = await self._retrieve_knowledge(query)
        
        # Generate response
        response = await self._generate_response(query, reasoning_chain, knowledge)
        
        # Cache response
        if len(self.query_cache) < self.cache_max_size:
            self.query_cache[cache_key] = response
        
        return response
    
    async def _build_reasoning_chain(self, query: str) -> ReasoningChain:
        """Build a chain of reasoning for the query"""
        nodes = []
        
        # Create root node
        root_node = ThoughtNode(
            id="root",
            content=query,
            confidence=1.0,
            parent_ids=[],
            children_ids=[],
            timestamp=datetime.now()
        )
        nodes.append(root_node)
        self.thought_graph["root"] = root_node
        
        # Build reasoning steps
        current_id = "root"
        for depth in range(1, min(self.reasoning_depth, 5)):
            child_content = await self._generate_next_thought(nodes[-1].content)
            child_node = ThoughtNode(
                id=f"node_{depth}",
                content=child_content,
                confidence=0.9 ** depth,
                parent_ids=[current_id],
                children_ids=[],
                timestamp=datetime.now()
            )
            
            nodes.append(child_node)
            self.thought_graph[child_node.id] = child_node
            
            # Update parent
            nodes[-2].children_ids.append(child_node.id)
            current_id = child_node.id
        
        # Create conclusion
        conclusion = await self._derive_conclusion(nodes)
        
        chain = ReasoningChain(
            nodes=nodes,
            conclusion=conclusion,
            confidence=nodes[-1].confidence,
            execution_path=[n.id for n in nodes]
        )
        
        self.active_chains.append(chain)
        if len(self.active_chains) > 100:
            self.active_chains.pop(0)
        
        return chain
    
    async def _generate_next_thought(self, current_thought: str) -> str:
        """Generate next thought in reasoning chain"""
        # Simple heuristic-based thought generation
        # In production, this would use a language model
        thoughts = [
            f"Analyzing: {current_thought}",
            f"Breaking down into components",
            f"Identifying key relationships",
            f"Evaluating alternatives",
            f"Synthesizing information"
        ]
        return thoughts[np.random.randint(len(thoughts))]
    
    async def _derive_conclusion(self, nodes: List[ThoughtNode]) -> str:
        """Derive conclusion from reasoning chain"""
        if not nodes:
            return "Unable to derive conclusion"
        
        # Aggregate information from all nodes
        all_content = " ".join([node.content for node in nodes])
        
        # Generate summary conclusion
        conclusion = f"Based on analysis: {all_content[:200]}..."
        return conclusion
    
    async def _retrieve_knowledge(self, query: str) -> List[Dict]:
        """Retrieve relevant knowledge from knowledge base"""
        if not self.knowledge_base:
            return []
        
        # Search knowledge base
        results = await self.knowledge_base.search(query, top_k=5)
        return results
    
    async def _generate_response(self, query: str, chain: ReasoningChain, 
                                  knowledge: List[Dict]) -> Dict[str, Any]:
        """Generate final response"""
        response = {
            'query': query,
            'answer': chain.conclusion,
            'confidence': chain.confidence,
            'reasoning_steps': len(chain.nodes),
            'knowledge_sources': len(knowledge),
            'timestamp': datetime.now().isoformat(),
            'context_used': len(self.context_window)
        }
        
        if knowledge:
            response['supporting_info'] = [k.get('summary', '') for k in knowledge[:3]]
        
        return response
    
    async def process(self, task: Dict[str, Any]) -> Any:
        """Generic task processing"""
        analysis = await self.analyze_task(task)
        
        # Execute based on analysis
        result = {
            'task_id': task.get('id'),
            'analysis': analysis,
            'status': 'processed',
            'output': f"Processed task: {task.get('description', 'No description')}"
        }
        
        return result
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        self.logger.info("Query cache cleared")
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state"""
        return {
            'active_chains': len(self.active_chains),
            'thought_graph_size': len(self.thought_graph),
            'context_length': len(self.context_window),
            'cache_size': len(self.query_cache),
            'parameters': {
                'reasoning_depth': self.reasoning_depth,
                'confidence_threshold': self.confidence_threshold,
                'creativity_factor': self.creativity_factor
            }
        }
