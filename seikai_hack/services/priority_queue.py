import heapq
from typing import List, Dict, Any
from datetime import datetime, timedelta
import uuid

class PriorityQueueService:
    def __init__(self):
        self.priority_queues = {}  # session_id -> priority queue
    
    async def update_priorities(self, session_id: str, question_results: List[Dict[str, Any]]):
        """Update topic priorities based on question results"""
        if session_id not in self.priority_queues:
            self.priority_queues[session_id] = []
        
        # Process each question result
        for result in question_results:
            analysis = result.get("analysis", {})
            topics = analysis.get("topics", [])
            is_correct = analysis.get("is_correct", False)
            confidence = analysis.get("confidence", 0.5)
            
            # Update priority for each topic
            for topic_name in topics:
                await self._update_topic_priority(
                    session_id, 
                    topic_name, 
                    is_correct, 
                    confidence
                )
        
        # Rebuild priority queue
        await self._rebuild_queue(session_id)
    
    async def _update_topic_priority(
        self, 
        session_id: str, 
        topic_name: str, 
        is_correct: bool, 
        confidence: float
    ):
        """Update priority for a specific topic"""
        # Find existing topic or create new one
        topic = await self._get_or_create_topic(session_id, topic_name)
        
        # Calculate new priority score
        new_score = self._calculate_priority_score(topic, is_correct, confidence)
        
        # Update topic stats
        topic["priority_score"] = new_score
        topic["questions_attempted"] += 1
        if is_correct:
            topic["questions_correct"] += 1
        topic["last_practiced"] = datetime.utcnow()
        
        # Store updated topic
        await self._store_topic(session_id, topic)
    
    def _calculate_priority_score(self, topic: Dict[str, Any], is_correct: bool, confidence: float) -> float:
        """Calculate new priority score based on performance"""
        base_score = topic.get("priority_score", 1.0)
        questions_attempted = topic.get("questions_attempted", 0)
        questions_correct = topic.get("questions_correct", 0)
        
        # If this is the first attempt, maintain base priority
        if questions_attempted == 0:
            return base_score
        
        # Calculate success rate
        success_rate = questions_correct / questions_attempted if questions_attempted > 0 else 0
        
        # Adjust priority based on performance
        if is_correct:
            # Correct answer decreases priority (less need to study)
            if success_rate > 0.8:
                # High success rate - significantly decrease priority
                new_score = base_score * 0.7
            elif success_rate > 0.6:
                # Moderate success rate - slightly decrease priority
                new_score = base_score * 0.9
            else:
                # Low success rate - maintain priority
                new_score = base_score
        else:
            # Wrong answer increases priority (more need to study)
            if success_rate < 0.3:
                # Low success rate - significantly increase priority
                new_score = base_score * 1.5
            elif success_rate < 0.6:
                # Moderate success rate - increase priority
                new_score = base_score * 1.2
            else:
                # High success rate - slight increase
                new_score = base_score * 1.1
        
        # Apply confidence adjustment
        if confidence < 0.7:
            # Low confidence in analysis - increase priority to be safe
            new_score *= 1.1
        
        # Ensure minimum priority
        return max(new_score, 0.1)
    
    async def _get_or_create_topic(self, session_id: str, topic_name: str) -> Dict[str, Any]:
        """Get existing topic or create new one"""
        # In a real implementation, this would query the database
        # For now, we'll use a simple in-memory approach
        topic_key = f"{session_id}_{topic_name}"
        
        # Check if topic exists in our in-memory storage
        if hasattr(self, '_topics') and topic_key in self._topics:
            return self._topics[topic_key]
        
        # Create new topic
        new_topic = {
            "id": str(uuid.uuid4()),
            "name": topic_name,
            "priority_score": 1.0,
            "questions_attempted": 0,
            "questions_correct": 0,
            "last_practiced": datetime.utcnow()
        }
        
        # Store in memory
        if not hasattr(self, '_topics'):
            self._topics = {}
        self._topics[topic_key] = new_topic
        
        return new_topic
    
    async def _store_topic(self, session_id: str, topic: Dict[str, Any]):
        """Store updated topic (in real implementation, this would update database)"""
        topic_key = f"{session_id}_{topic['name']}"
        if not hasattr(self, '_topics'):
            self._topics = {}
        self._topics[topic_key] = topic
    
    async def _rebuild_queue(self, session_id: str):
        """Rebuild the priority queue for a session"""
        # Get all topics for this session
        session_topics = []
        if hasattr(self, '_topics'):
            for key, topic in self._topics.items():
                if key.startswith(session_id):
                    session_topics.append(topic)
        
        # Sort by priority score (highest priority first)
        session_topics.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Store sorted queue
        self.priority_queues[session_id] = session_topics
    
    async def get_priorities(self, session_id: str) -> List[Dict[str, Any]]:
        """Get prioritized list of topics for study focus"""
        if session_id not in self.priority_queues:
            await self._rebuild_queue(session_id)
        
        priorities = self.priority_queues.get(session_id, [])
        
        # Add study recommendations
        for topic in priorities:
            topic["study_recommendation"] = self._generate_study_recommendation(topic)
        
        return priorities
    
    def _generate_study_recommendation(self, topic: Dict[str, Any]) -> str:
        """Generate study recommendation based on topic performance"""
        success_rate = topic["questions_correct"] / topic["questions_attempted"] if topic["questions_attempted"] > 0 else 0
        
        if success_rate >= 0.8:
            return "Review briefly - you're doing well!"
        elif success_rate >= 0.6:
            return "Practice more problems - you're on the right track"
        elif success_rate >= 0.4:
            return "Focus on this topic - review concepts and practice"
        else:
            return "High priority - review fundamentals and practice extensively"
    
    async def reset_priorities(self, session_id: str):
        """Reset all topic priorities to default values"""
        if hasattr(self, '_topics'):
            # Reset all topics for this session
            for key in list(self._topics.keys()):
                if key.startswith(session_id):
                    topic = self._topics[key]
                    topic["priority_score"] = 1.0
                    topic["questions_attempted"] = 0
                    topic["questions_correct"] = 0
                    topic["last_practiced"] = datetime.utcnow()
        
        # Rebuild queue
        await self._rebuild_queue(session_id)
