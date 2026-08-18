"""
Memory Manager - Cognitive Buffer and Memory Systems
Implements advanced memory mechanisms for AI systems including
working memory, long-term memory, attention-based retrieval, and memory consolidation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import OrderedDict, deque
import json
import hashlib


class CognitiveBuffer:
    """
    Advanced working memory buffer with:
    - Limited capacity with intelligent eviction
    - Priority-based retention
    - Temporal decay
    - Associative retrieval
    - Multi-modal storage
    """
    
    def __init__(
        self,
        capacity: int = 100,
        embedding_dim: int = 512,
        decay_rate: float = 0.01,
        priority_boost: float = 0.1
    ):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.priority_boost = priority_boost
        
        # Memory storage
        self.memory_store = OrderedDict()
        self.memory_embeddings = {}
        self.memory_priorities = {}
        self.memory_timestamps = {}
        self.memory_access_count = {}
        
        # Index structures for fast retrieval
        self.semantic_index = {}
        self.temporal_index = deque(maxlen=capacity)
        
        # Memory statistics
        self.total_writes = 0
        self.total_reads = 0
        self.eviction_count = 0
        
    def _compute_embedding(self, content: Union[str, np.ndarray]) -> np.ndarray:
        """Compute or retrieve embedding for memory item."""
        if isinstance(content, np.ndarray):
            if len(content) != self.embedding_dim:
                # Project to embedding dimension
                if len(content.shape) == 1:
                    projection = np.random.randn(len(content), self.embedding_dim) * 0.01
                    return np.dot(content, projection)
                else:
                    return np.mean(content, axis=0)[:self.embedding_dim]
            return content
        
        # String content - use hash-based pseudo-embedding
        if isinstance(content, str):
            hash_bytes = hashlib.md5(content.encode()).digest()
            hash_array = np.frombuffer(hash_bytes, dtype=np.uint8).astype(float)
            
            # Expand to embedding dimension
            expanded = np.tile(hash_array, (self.embedding_dim // len(hash_array) + 1))[:self.embedding_dim]
            return (expanded - 128) / 128  # Normalize to [-1, 1]
        
        return np.random.randn(self.embedding_dim) * 0.1
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def write(
        self,
        key: str,
        content: Any,
        priority: float = 0.5,
        metadata: Dict = None
    ) -> bool:
        """
        Write item to memory buffer.
        
        Args:
            key: Unique identifier for memory item
            content: Content to store
            priority: Importance priority (0-1)
            metadata: Optional metadata
            
        Returns:
            True if write successful
        """
        # Check if need to evict
        if len(self.memory_store) >= self.capacity and key not in self.memory_store:
            self._evict_lowest_priority()
        
        # Store content
        self.memory_store[key] = {
            'content': content,
            'metadata': metadata or {},
            'created_at': self.total_writes
        }
        
        # Compute and store embedding
        if isinstance(content, (str, np.ndarray)):
            self.memory_embeddings[key] = self._compute_embedding(content)
        else:
            # For other types, create embedding from string representation
            self.memory_embeddings[key] = self._compute_embedding(str(content))
        
        # Initialize priority and timestamps
        self.memory_priorities[key] = priority
        self.memory_timestamps[key] = self.total_writes
        self.memory_access_count[key] = 0
        
        # Update indices
        self.temporal_index.append(key)
        
        self.total_writes += 1
        
        return True
    
    def read(self, key: str) -> Optional[Any]:
        """Read item from memory by key."""
        if key not in self.memory_store:
            return None
        
        # Update access statistics
        self.memory_access_count[key] += 1
        
        # Boost priority on access
        self.memory_priorities[key] = min(1.0, 
            self.memory_priorities[key] + self.priority_boost)
        
        # Move to end of OrderedDict (most recently used)
        self.memory_store.move_to_end(key)
        
        self.total_reads += 1
        
        return self.memory_store[key]['content']
    
    def search_by_similarity(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Tuple[str, Any, float]]:
        """
        Search memory by semantic similarity.
        
        Args:
            query: Query string or embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (key, content, similarity_score) tuples
        """
        query_embedding = self._compute_embedding(query)
        
        results = []
        for key, memory_embedding in self.memory_embeddings.items():
            similarity = self._compute_similarity(query_embedding, memory_embedding)
            
            if similarity >= threshold:
                # Apply temporal decay to score
                age = self.total_writes - self.memory_timestamps[key]
                decay_factor = np.exp(-self.decay_rate * age)
                adjusted_score = similarity * decay_factor
                
                # Apply priority boost
                adjusted_score *= (0.5 + 0.5 * self.memory_priorities[key])
                
                results.append((key, self.memory_store[key]['content'], adjusted_score))
        
        # Sort by adjusted score
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:top_k]
    
    def search_by_temporal_proximity(
        self,
        reference_key: str,
        window_size: int = 10
    ) -> List[Tuple[str, Any]]:
        """Search for memories temporally close to a reference."""
        if reference_key not in self.memory_store:
            return []
        
        ref_timestamp = self.memory_timestamps[reference_key]
        results = []
        
        for key in self.memory_store:
            timestamp = self.memory_timestamps[key]
            if abs(timestamp - ref_timestamp) <= window_size and key != reference_key:
                results.append((key, self.memory_store[key]['content']))
        
        results.sort(key=lambda x: abs(self.memory_timestamps[x[0]] - ref_timestamp))
        
        return results
    
    def _evict_lowest_priority(self):
        """Evict the lowest priority memory item."""
        if not self.memory_store:
            return
        
        # Compute eviction scores (lower is better for eviction)
        eviction_scores = {}
        for key in self.memory_store:
            priority = self.memory_priorities[key]
            age = self.total_writes - self.memory_timestamps[key]
            access_count = self.memory_access_count[key]
            
            # Score: low priority, old, and rarely accessed items should be evicted
            score = priority * 0.4 - (age / max(1, self.total_writes)) * 0.3 - \
                    min(1, access_count / 10) * 0.3
            
            eviction_scores[key] = score
        
        # Evict item with lowest score
        evict_key = min(eviction_scores, key=eviction_scores.get)
        
        del self.memory_store[evict_key]
        if evict_key in self.memory_embeddings:
            del self.memory_embeddings[evict_key]
        if evict_key in self.memory_priorities:
            del self.memory_priorities[evict_key]
        if evict_key in self.memory_timestamps:
            del self.memory_timestamps[evict_key]
        if evict_key in self.memory_access_count:
            del self.memory_access_count[evict_key]
        
        self.eviction_count += 1
    
    def apply_decay(self):
        """Apply temporal decay to all memory priorities."""
        for key in self.memory_priorities:
            age = self.total_writes - self.memory_timestamps[key]
            decay = np.exp(-self.decay_rate * age)
            self.memory_priorities[key] *= decay
    
    def consolidate_memories(self, threshold: float = 0.8) -> List[str]:
        """
        Consolidate similar memories by merging them.
        
        Args:
            threshold: Similarity threshold for consolidation
            
        Returns:
            List of consolidated memory keys
        """
        consolidated = []
        keys_to_remove = set()
        
        keys = list(self.memory_embeddings.keys())
        for i, key1 in enumerate(keys):
            if key1 in keys_to_remove:
                continue
            
            for key2 in keys[i+1:]:
                if key2 in keys_to_remove:
                    continue
                
                similarity = self._compute_similarity(
                    self.memory_embeddings[key1],
                    self.memory_embeddings[key2]
                )
                
                if similarity >= threshold:
                    # Merge key2 into key1
                    content1 = self.memory_store[key1]['content']
                    content2 = self.memory_store[key2]['content']
                    
                    # Simple merge strategy - can be customized
                    if isinstance(content1, str) and isinstance(content2, str):
                        merged_content = f"{content1} | {content2}"
                    elif isinstance(content1, np.ndarray) and isinstance(content2, np.ndarray):
                        merged_content = (content1 + content2) / 2
                    else:
                        merged_content = content1
                    
                    # Update key1 with merged content
                    self.memory_store[key1]['content'] = merged_content
                    self.memory_priorities[key1] = max(
                        self.memory_priorities[key1],
                        self.memory_priorities[key2]
                    )
                    
                    keys_to_remove.add(key2)
                    consolidated.append(key2)
        
        # Remove consolidated memories
        for key in keys_to_remove:
            del self.memory_store[key]
            if key in self.memory_embeddings:
                del self.memory_embeddings[key]
            if key in self.memory_priorities:
                del self.memory_priorities[key]
            if key in self.memory_timestamps:
                del self.memory_timestamps[key]
            if key in self.memory_access_count:
                del self.memory_access_count[key]
        
        return consolidated
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memory_store:
            return {'status': 'empty'}
        
        priorities = list(self.memory_priorities.values())
        access_counts = list(self.memory_access_count.values())
        ages = [self.total_writes - ts for ts in self.memory_timestamps.values()]
        
        return {
            'total_memories': len(self.memory_store),
            'capacity': self.capacity,
            'utilization': len(self.memory_store) / self.capacity,
            'total_writes': self.total_writes,
            'total_reads': self.total_reads,
            'eviction_count': self.eviction_count,
            'avg_priority': np.mean(priorities),
            'avg_access_count': np.mean(access_counts),
            'avg_age': np.mean(ages),
            'max_age': max(ages),
            'read_write_ratio': self.total_reads / max(1, self.total_writes)
        }
    
    def clear(self):
        """Clear all memories."""
        self.memory_store.clear()
        self.memory_embeddings.clear()
        self.memory_priorities.clear()
        self.memory_timestamps.clear()
        self.memory_access_count.clear()
        self.temporal_index.clear()


class MemoryManager:
    """
    Comprehensive memory management system with multiple memory types:
    - Working memory (CognitiveBuffer)
    - Long-term memory storage
    - Episodic memory
    - Semantic memory
    - Procedural memory
    """
    
    def __init__(
        self,
        working_memory_capacity: int = 100,
        long_term_memory_capacity: int = 10000,
        consolidation_threshold: float = 0.85,
        retrieval_temperature: float = 0.7
    ):
        # Working memory
        self.working_memory = CognitiveBuffer(capacity=working_memory_capacity)
        
        # Long-term memory
        self.long_term_memory = CognitiveBuffer(capacity=long_term_memory_capacity)
        
        # Memory type specific stores
        self.episodic_memory = []  # Sequential episodes
        self.semantic_memory = {}  # Fact knowledge
        self.procedural_memory = {}  # Skills and procedures
        
        # Consolidation parameters
        self.consolidation_threshold = consolidation_threshold
        self.retrieval_temperature = retrieval_temperature
        
        # Memory operation counters
        self.consolidation_count = 0
        self.retrieval_count = 0
        
    def store_episode(self, episode: Dict) -> int:
        """Store an episodic memory."""
        episode_id = len(self.episodic_memory)
        episode['id'] = episode_id
        episode['timestamp'] = len(self.episodic_memory)
        
        self.episodic_memory.append(episode)
        
        # Also store in working memory for quick access
        self.working_memory.write(
            f"episode_{episode_id}",
            episode,
            priority=0.7,
            metadata={'type': 'episodic'}
        )
        
        return episode_id
    
    def store_semantic_fact(self, concept: str, information: Any, relations: List[str] = None):
        """Store a semantic fact."""
        self.semantic_memory[concept] = {
            'information': information,
            'relations': relations or [],
            'access_count': 0
        }
        
        self.working_memory.write(
            f"semantic_{concept}",
            information,
            priority=0.6,
            metadata={'type': 'semantic', 'concept': concept}
        )
    
    def store_procedure(self, skill_name: str, procedure: Callable or List, context: Dict = None):
        """Store a procedural memory (skill or procedure)."""
        self.procedural_memory[skill_name] = {
            'procedure': procedure,
            'context': context or {},
            'execution_count': 0
        }
    
    def retrieve_relevant(
        self,
        query: str,
        memory_types: List[str] = None,
        top_k: int = 5
    ) -> Dict[str, List]:
        """
        Retrieve relevant memories across different memory types.
        
        Args:
            query: Search query
            memory_types: Types of memory to search
            top_k: Results per memory type
            
        Returns:
            Dictionary of memory type to results
        """
        memory_types = memory_types or ['working', 'long_term', 'episodic', 'semantic']
        results = {}
        
        if 'working' in memory_types:
            working_results = self.working_memory.search_by_similarity(
                query, top_k=top_k, threshold=0.2
            )
            results['working'] = working_results
        
        if 'long_term' in memory_types:
            lt_results = self.long_term_memory.search_by_similarity(
                query, top_k=top_k, threshold=0.3
            )
            results['long_term'] = lt_results
        
        if 'episodic' in memory_types:
            # Search episodic memory by content similarity
            episode_results = []
            query_embedding = self.working_memory._compute_embedding(query)
            
            for episode in self.episodic_memory:
                episode_embedding = self.working_memory._compute_embedding(str(episode))
                similarity = self.working_memory._compute_similarity(query_embedding, episode_embedding)
                
                if similarity > 0.2:
                    episode_results.append((episode['id'], episode, similarity))
            
            episode_results.sort(key=lambda x: x[2], reverse=True)
            results['episodic'] = episode_results[:top_k]
        
        if 'semantic' in memory_types:
            # Search semantic memory by concept matching
            semantic_results = []
            query_lower = query.lower()
            
            for concept, data in self.semantic_memory.items():
                if query_lower in concept.lower() or \
                   any(query_lower in rel.lower() for rel in data.get('relations', [])):
                    data['access_count'] += 1
                    semantic_results.append((concept, data['information']))
            
            results['semantic'] = semantic_results[:top_k]
        
        self.retrieval_count += 1
        
        return results
    
    def consolidate_to_long_term(self, min_priority: float = 0.7):
        """Consolidate high-priority working memories to long-term memory."""
        consolidated_count = 0
        
        for key in list(self.working_memory.memory_store.keys()):
            priority = self.working_memory.memory_priorities.get(key, 0)
            
            if priority >= min_priority:
                content = self.working_memory.read(key)
                metadata = self.working_memory.memory_store[key].get('metadata', {})
                
                self.long_term_memory.write(
                    key,
                    content,
                    priority=priority,
                    metadata=metadata
                )
                
                consolidated_count += 1
        
        self.consolidation_count += 1
        
        return consolidated_count
    
    def run_consolidation_cycle(self):
        """Run a full memory consolidation cycle."""
        # Apply decay to working memory
        self.working_memory.apply_decay()
        
        # Consolidate similar working memories
        self.working_memory.consolidate_memories(self.consolidation_threshold)
        
        # Move high-priority memories to long-term storage
        self.consolidate_to_long_term()
        
        # Consolidate long-term memories
        self.long_term_memory.consolidate_memories(self.consolidation_threshold)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        return {
            'working_memory': self.working_memory.get_memory_statistics(),
            'long_term_memory': self.long_term_memory.get_memory_statistics(),
            'episodic_memory_size': len(self.episodic_memory),
            'semantic_memory_size': len(self.semantic_memory),
            'procedural_memory_size': len(self.procedural_memory),
            'consolidation_cycles': self.consolidation_count,
            'total_retrievals': self.retrieval_count
        }
    
    def save_memory_state(self, filepath: str):
        """Save complete memory state to file."""
        # Note: This saves serializable parts; functions in procedural memory are not saved
        state = {
            'working_memory_stats': self.working_memory.get_memory_statistics(),
            'long_term_memory_stats': self.long_term_memory.get_memory_statistics(),
            'episodic_memory': self.episodic_memory,
            'semantic_memory': self.semantic_memory,
            'procedural_memory_keys': list(self.procedural_memory.keys()),
            'consolidation_count': self.consolidation_count,
            'retrieval_count': self.retrieval_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_memory_state(self, filepath: str):
        """Load memory state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.episodic_memory = state.get('episodic_memory', [])
        self.semantic_memory = state.get('semantic_memory', {})
        self.consolidation_count = state.get('consolidation_count', 0)
        self.retrieval_count = state.get('retrieval_count', 0)
