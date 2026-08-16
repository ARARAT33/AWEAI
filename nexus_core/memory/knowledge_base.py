"""
KnowledgeBase - Advanced memory and knowledge management system
Stores, organizes, and retrieves information efficiently
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np

from ..utils.logger import setup_logger


@dataclass
class KnowledgeEntry:
    """Represents a piece of stored knowledge"""
    id: str
    content: Any
    category: str
    tags: List[str]
    source: str
    confidence: float
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'category': self.category,
            'tags': self.tags,
            'source': self.source,
            'confidence': self.confidence,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class MemoryTrace:
    """Records how knowledge was acquired"""
    event_type: str
    description: str
    timestamp: datetime
    related_entries: List[str]
    context: Dict[str, Any]


class KnowledgeBase:
    """
    Advanced knowledge management system
    
    Features:
    - Hierarchical knowledge storage
    - Semantic search
    - Automatic categorization
    - Forgetting mechanism (based on usage)
    - Experience storage
    - Cross-referencing
    """
    
    def __init__(self, config):
        self.logger = setup_logger("KnowledgeBase")
        self.config = config
        
        # Storage
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.categories: Dict[str, List[str]] = {}  # category -> entry_ids
        self.tag_index: Dict[str, List[str]] = {}   # tag -> entry_ids
        
        # Semantic embeddings (for similarity search)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_model = None
        
        # Memory traces
        self.traces: List[MemoryTrace] = []
        self.max_traces = 1000
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'total_accesses': 0,
            'categories_count': 0,
            'avg_confidence': 0.0
        }
        
        # Configuration
        self.default_ttl_days = self.config.get('knowledge_ttl_days', 30)
        self.max_entries = self.config.get('max_knowledge_entries', 10000)
        self.similarity_threshold = 0.7
        
        # Persistence
        self.storage_path = Path(self.config.get('storage_path', './nexus_memory'))
        
        self.logger.info("Knowledge Base initialized")
    
    async def store(self, content: Any, category: str = 'general', 
                    tags: Optional[List[str]] = None, 
                    source: str = 'system',
                    confidence: float = 0.9,
                    ttl_days: Optional[int] = None) -> str:
        """
        Store new knowledge
        
        Args:
            content: Knowledge content
            category: Category for organization
            tags: Tags for indexing
            source: Source of the knowledge
            confidence: Confidence score (0-1)
            ttl_days: Time to live in days
            
        Returns:
            Entry ID
        """
        # Generate ID
        entry_id = f"kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.entries)}"
        
        # Calculate expiration
        expires_at = None
        if ttl_days is not None or self.default_ttl_days > 0:
            days = ttl_days if ttl_days is not None else self.default_ttl_days
            expires_at = datetime.now() + timedelta(days=days)
        
        # Create entry
        entry = KnowledgeEntry(
            id=entry_id,
            content=content,
            category=category,
            tags=tags or [],
            source=source,
            confidence=confidence,
            expires_at=expires_at
        )
        
        # Store
        self.entries[entry_id] = entry
        
        # Update indexes
        self._update_category_index(category, entry_id)
        for tag in entry.tags:
            self._update_tag_index(tag, entry_id)
        
        # Generate embedding for semantic search
        await self._generate_embedding(entry_id, str(content))
        
        # Update stats
        self.stats['total_entries'] = len(self.entries)
        self.stats['categories_count'] = len(self.categories)
        self._update_avg_confidence()
        
        # Check size limit
        if len(self.entries) > self.max_entries:
            await self._cleanup_old_entries()
        
        self.logger.debug(f"Stored knowledge: {entry_id} in category '{category}'")
        return entry_id
    
    async def retrieve(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve specific knowledge entry"""
        entry = self.entries.get(entry_id)
        
        if entry:
            # Update access stats
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self.stats['total_accesses'] += 1
        
        return entry
    
    async def search(self, query: str, top_k: int = 10, 
                     category: Optional[str] = None) -> List[Dict]:
        """
        Search for relevant knowledge
        
        Args:
            query: Search query
            top_k: Number of results to return
            category: Filter by category
            
        Returns:
            List of relevant entries with scores
        """
        results = []
        
        # Try semantic search first
        if self.embedding_model is not None and self.embeddings:
            semantic_results = await self._semantic_search(query, top_k * 2)
            results.extend(semantic_results)
        
        # Also do keyword/tag search
        keyword_results = self._keyword_search(query, top_k * 2, category)
        
        # Merge and deduplicate
        all_results = self._merge_results(results, keyword_results)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Return top k
        return all_results[:top_k]
    
    async def _semantic_search(self, query: str, limit: int) -> List[Dict]:
        """Semantic similarity search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = await self._encode_text(query)
            
            if query_embedding is None:
                return []
            
            # Calculate similarities
            similarities = []
            for entry_id, embedding in self.embeddings.items():
                if entry_id in self.entries:
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    if similarity >= self.similarity_threshold:
                        similarities.append((entry_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Build results
            results = []
            for entry_id, score in similarities[:limit]:
                entry = self.entries[entry_id]
                results.append({
                    'entry': entry.to_dict(),
                    'score': float(score),
                    'match_type': 'semantic'
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, limit: int, 
                        category: Optional[str]) -> List[Dict]:
        """Keyword and tag-based search"""
        results = []
        query_lower = query.lower()
        
        # Search in entries
        for entry_id, entry in self.entries.items():
            if category and entry.category != category:
                continue
            
            score = 0.0
            
            # Check content
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                score += 0.5
            
            # Check tags
            matching_tags = [t for t in entry.tags if query_lower in t.lower()]
            score += len(matching_tags) * 0.2
            
            # Check category
            if query_lower in entry.category.lower():
                score += 0.3
            
            if score > 0:
                results.append({
                    'entry': entry.to_dict(),
                    'score': score,
                    'match_type': 'keyword'
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def _merge_results(self, *result_lists) -> List[Dict]:
        """Merge multiple result lists, keeping highest scores"""
        merged = {}
        
        for results in result_lists:
            for result in results:
                entry_id = result['entry']['id']
                if entry_id not in merged or result['score'] > merged[entry_id]['score']:
                    merged[entry_id] = result
        
        return list(merged.values())
    
    async def store_execution_experience(self, execution_record: Dict):
        """Store experience from task execution for future learning"""
        category = f"experience_{execution_record.get('action', 'generic')}"
        tags = ['execution', 'experience', execution_record.get('action', '')]
        
        await self.store(
            content=execution_record,
            category=category,
            tags=tags,
            source='task_executor',
            confidence=1.0 if execution_record.get('success') else 0.5
        )
        
        # Record trace
        self._record_trace(
            event_type='execution',
            description=f"Executed: {execution_record.get('action')}",
            related_entries=[category],
            context=execution_record
        )
    
    async def learn_pattern(self, pattern_type: str, pattern_data: Dict, 
                           success_rate: float):
        """Learn and store recurring patterns"""
        await self.store(
            content=pattern_data,
            category=f'pattern_{pattern_type}',
            tags=['pattern', pattern_type],
            source='learning_system',
            confidence=success_rate
        )
    
    async def get_similar_experiences(self, current_situation: Dict) -> List[Dict]:
        """Find similar past experiences"""
        # Convert situation to searchable query
        query = " ".join([
            str(v) for v in current_situation.values() 
            if isinstance(v, (str, int, float))
        ])
        
        # Search for similar experiences
        results = await self.search(query, top_k=5, category='experience_*')
        return results
    
    def _update_category_index(self, category: str, entry_id: str):
        """Update category index"""
        if category not in self.categories:
            self.categories[category] = []
        if entry_id not in self.categories[category]:
            self.categories[category].append(entry_id)
    
    def _update_tag_index(self, tag: str, entry_id: str):
        """Update tag index"""
        if tag not in self.tag_index:
            self.tag_index[tag] = []
        if entry_id not in self.tag_index[tag]:
            self.tag_index[tag].append(entry_id)
    
    async def _generate_embedding(self, entry_id: str, text: str):
        """Generate embedding vector for text"""
        try:
            if self.embedding_model is None:
                # Try to load model
                try:
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                except ImportError:
                    self.logger.debug("Embedding model not available")
                    return
            
            embedding = self.embedding_model.encode(text)
            self.embeddings[entry_id] = embedding
            
        except Exception as e:
            self.logger.debug(f"Could not generate embedding: {e}")
    
    async def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding"""
        if self.embedding_model is None:
            return None
        
        try:
            return self.embedding_model.encode(text)
        except Exception:
            return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    def _update_avg_confidence(self):
        """Update average confidence statistic"""
        if self.entries:
            confidences = [e.confidence for e in self.entries.values()]
            self.stats['avg_confidence'] = np.mean(confidences)
    
    def _record_trace(self, event_type: str, description: str,
                      related_entries: List[str], context: Dict):
        """Record memory trace"""
        trace = MemoryTrace(
            event_type=event_type,
            description=description,
            timestamp=datetime.now(),
            related_entries=related_entries,
            context=context
        )
        
        self.traces.append(trace)
        if len(self.traces) > self.max_traces:
            self.traces.pop(0)
    
    async def _cleanup_old_entries(self):
        """Remove old/expired entries"""
        now = datetime.now()
        expired = [
            entry_id for entry_id, entry in self.entries.items()
            if entry.expires_at and entry.expires_at < now
        ]
        
        for entry_id in expired:
            await self._remove_entry(entry_id)
        
        if expired:
            self.logger.info(f"Cleaned up {len(expired)} expired entries")
    
    async def _remove_entry(self, entry_id: str):
        """Remove an entry from the knowledge base"""
        if entry_id not in self.entries:
            return
        
        entry = self.entries[entry_id]
        
        # Remove from indexes
        if entry.category in self.categories:
            if entry_id in self.categories[entry.category]:
                self.categories[entry.category].remove(entry_id)
        
        for tag in entry.tags:
            if tag in self.tag_index:
                if entry_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(entry_id)
        
        # Remove embedding
        if entry_id in self.embeddings:
            del self.embeddings[entry_id]
        
        # Remove entry
        del self.entries[entry_id]
        
        self.stats['total_entries'] = len(self.entries)
    
    async def cleanup_old_entries(self):
        """Public method to trigger cleanup"""
        await self._cleanup_old_entries()
    
    async def save_state(self):
        """Save knowledge base to disk"""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Save entries
            entries_data = {
                entry_id: entry.to_dict() 
                for entry_id, entry in self.entries.items()
            }
            
            with open(self.storage_path / 'entries.json', 'w') as f:
                json.dump(entries_data, f, indent=2)
            
            # Save indexes
            indexes = {
                'categories': self.categories,
                'tag_index': self.tag_index,
                'stats': self.stats
            }
            
            with open(self.storage_path / 'indexes.json', 'w') as f:
                json.dump(indexes, f, indent=2)
            
            self.logger.info(f"Knowledge base saved to {self.storage_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save knowledge base: {e}")
    
    async def load_state(self):
        """Load knowledge base from disk"""
        try:
            entries_file = self.storage_path / 'entries.json'
            indexes_file = self.storage_path / 'indexes.json'
            
            if not entries_file.exists():
                self.logger.warning("No saved knowledge base found")
                return False
            
            # Load entries
            with open(entries_file, 'r') as f:
                entries_data = json.load(f)
            
            for entry_id, data in entries_data.items():
                # Reconstruct KnowledgeEntry
                entry = KnowledgeEntry(
                    id=data['id'],
                    content=data['content'],
                    category=data['category'],
                    tags=data['tags'],
                    source=data['source'],
                    confidence=data['confidence'],
                    access_count=data['access_count'],
                    last_accessed=datetime.fromisoformat(data['last_accessed']) if data['last_accessed'] else None,
                    created_at=datetime.fromisoformat(data['created_at']),
                    expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None
                )
                self.entries[entry_id] = entry
            
            # Load indexes
            with open(indexes_file, 'r') as f:
                indexes = json.load(f)
            
            self.categories = indexes.get('categories', {})
            self.tag_index = indexes.get('tag_index', {})
            self.stats = indexes.get('stats', self.stats)
            
            self.logger.info(f"Knowledge base loaded from {self.storage_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            **self.stats,
            'storage_path': str(self.storage_path),
            'traces_count': len(self.traces),
            'embeddings_count': len(self.embeddings)
        }
    
    def get_categories(self) -> List[str]:
        """Get list of all categories"""
        return list(self.categories.keys())
    
    def get_all_tags(self) -> List[str]:
        """Get list of all tags"""
        return list(self.tag_index.keys())
