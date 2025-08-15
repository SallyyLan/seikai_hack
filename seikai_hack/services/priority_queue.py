import heapq
from typing import List, Dict, Any
from datetime import datetime, timedelta
import uuid

class PriorityQueueService:
    def __init__(self):
        self.priority_queues = {}  # session_id -> priority queue
    
    async def calculate_study_priorities(
        self, 
        session_id: str, 
        topics: List[Dict[str, Any]], 
        minimum_score: float,
        maximum_score: float
    ) -> List[Dict[str, Any]]:
        """Calculate study priorities using the confidence scoring formula"""
        try:
            print(f"🔍 Priority calculation: {len(topics)} topics, min_score: {minimum_score}, max_score: {maximum_score}")
            
            if not topics:
                print("⚠️ No topics provided for priority calculation")
                return []
            
            # Calculate scores for each topic using the formula: s_i = B_i * (7 - c_i)/7
            # where s_i is the score for that topic, B_i is the score worth, c_i is confidence
            for topic in topics:
                topic_score = topic.get("score_value", 0.0)
                user_confidence = topic.get("user_confidence", 1)
                
                # Apply the formula: s_i = B_i * (7 - user_confidence) / 7
                calculated_score = round(topic_score * (7 - user_confidence) / 7, 2)
                topic["calculated_score"] = calculated_score
                
                print(f"📊 Topic: {topic['name']}, Score: {topic_score:.2f}, Confidence: {user_confidence}, Calculated: {calculated_score:.2f}")
            
            # Sort topics by calculated score (highest first)
            sorted_topics = sorted(topics, key=lambda x: x["calculated_score"], reverse=True)
            print(f"📈 Sorted topics by calculated score: {[t['name'] for t in sorted_topics]}")
            
            # Build priority queue using the algorithm
            selected_topics = []
            current_sum = 0.0
            threshold = 1.0  # Threshold for similar scores
            
            # If minimum_score is 0 or very low, select all topics
            if minimum_score <= 0.1:
                print(f"🎯 Minimum score is {minimum_score}, selecting all topics")
                for i, topic in enumerate(sorted_topics):
                    topic["study_priority"] = i + 1
                    selected_topics.append(topic)
                    print(f"✅ Added all topics: {topic['name']}, priority: {topic['study_priority']}")
            else:
                # Use the original algorithm for higher minimum scores
                print(f"🎯 Using score-based selection algorithm for minimum score: {minimum_score}")
                i = 0
                while current_sum < minimum_score and i < len(sorted_topics):
                    current_topic = sorted_topics[i]
                    current_score = current_topic["calculated_score"]
                    
                    # Check if we have adjacent topics to compare
                    if i + 1 < len(sorted_topics):
                        next_topic = sorted_topics[i + 1]
                        next_score = next_topic["calculated_score"]
                        
                        # Check if scores are similar (within threshold)
                        if abs(current_score - next_score) <= threshold:
                            # Use ChatGPT to determine which topic is easier
                            easier_topic = await self._determine_easier_topic(current_topic, next_topic)
                            
                            if easier_topic["id"] == current_topic["id"]:
                                # Current topic is easier, add it
                                selected_topics.append(current_topic)
                                current_sum += current_score
                                current_topic["study_priority"] = len(selected_topics)
                                print(f"✅ Added topic: {current_topic['name']}, priority: {current_topic['study_priority']}, sum: {current_sum}")
                                i += 1
                            else:
                                # Next topic is easier, add it instead
                                selected_topics.append(next_topic)
                                current_sum += next_score
                                next_topic["study_priority"] = len(selected_topics)
                                print(f"✅ Added topic: {next_topic['name']}, priority: {next_topic['study_priority']}, sum: {current_sum}")
                                # Skip both topics since we used the next one
                                i += 2
                        else:
                            # Scores are not similar, add current topic
                            selected_topics.append(current_topic)
                            current_sum += current_score
                            current_topic["study_priority"] = len(selected_topics)
                            print(f"✅ Added topic: {current_topic['name']}, priority: {current_topic['study_priority']}, sum: {current_sum}")
                            i += 1
                    else:
                        # Last topic, just add it
                        selected_topics.append(current_topic)
                        current_sum += current_score
                        current_topic["study_priority"] = len(selected_topics)
                        print(f"✅ Added topic: {current_topic['name']}, priority: {current_topic['study_priority']}, sum: {current_sum}")
                        i += 1
                
                # If we still haven't reached minimum_score, add remaining topics
                if current_sum < minimum_score and len(selected_topics) < len(sorted_topics):
                    print(f"⚠️ Haven't reached minimum score ({current_sum}/{minimum_score}), adding remaining topics")
                    for i in range(len(selected_topics), len(sorted_topics)):
                        remaining_topic = sorted_topics[i]
                        remaining_topic["study_priority"] = len(selected_topics) + 1
                        selected_topics.append(remaining_topic)
                        print(f"✅ Added remaining topic: {remaining_topic['name']}, priority: {remaining_topic['study_priority']}")
            
            print(f"🎯 Final selection: {len(selected_topics)} topics selected")
            
            # Store the priority queue
            self.priority_queues[session_id] = selected_topics
            
            return selected_topics
            
        except Exception as e:
            raise Exception(f"Failed to calculate study priorities: {str(e)}")
    
    async def calculate_topic_based_priorities(
        self, 
        session_id: str, 
        exam_topics: List[str],
        topic_confidence: Dict[str, Any],
        practice_exam_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Calculate study priorities based on topics and user confidence levels"""
        try:
            print(f"🔍 Topic-based priority calculation for session: {session_id}")
            print(f"🔍 Exam topics: {exam_topics}")
            print(f"🔍 Topic confidence: {topic_confidence}")
            print(f"🔍 Practice exam data keys: {list(practice_exam_data.keys())}")
            
            # Get topics data from practice exam
            topics_data = practice_exam_data.get("topics", [])
            print(f"🔍 Topics data from practice exam: {len(topics_data) if topics_data else 0} topics")
            
            # If we have topics data, use it directly
            if topics_data:
                print(f"✅ Using topics data directly")
                topics_list = []
                covered_topics = set()
                practice_exam_scores = []
                
                # First, add topics that appear in the practice exam and collect their scores
                for topic_data in topics_data:
                    topic_name = topic_data["name"]
                    if topic_name in exam_topics:  # Only include topics from exam coverage
                        raw_score = float(topic_data["score"])
                        practice_exam_scores.append(raw_score)
                        
                        topics_list.append({
                            "id": topic_name,
                            "name": topic_name,
                            "score_value": raw_score,
                            "user_confidence": topic_confidence.get(topic_name, 1),
                            "questions_covering": topic_data.get("questions", []),
                            "total_questions": len(topic_data.get("questions", [])),
                            "appears_in_practice": True,
                            "raw_score": raw_score
                        })
                        covered_topics.add(topic_name)
                        print(f"📝 Added practice exam topic: {topic_name}, raw score: {raw_score:.2f}, confidence: {topic_confidence.get(topic_name, 1)}")
                    else:
                        print(f"⚠️ Topic '{topic_name}' not in exam coverage, skipping")
                
                # Calculate average raw score from practice exam topics
                if practice_exam_scores:
                    average_raw_score = round(sum(practice_exam_scores) / len(practice_exam_scores), 2)
                    print(f"📊 Average raw score from practice exam topics: {average_raw_score:.2f}")
                else:
                    # Fallback if no practice exam scores
                    total_score = practice_exam_data.get("total_score", 100.0)
                    average_raw_score = round(max(1.0, total_score / len(exam_topics)), 2)
                    print(f"⚠️ No practice exam scores, using fallback average: {average_raw_score:.2f}")
                
                # Then, add topics that are in exam coverage but NOT in practice exam
                missing_topics = []
                
                for exam_topic in exam_topics:
                    if exam_topic not in covered_topics:
                        # Topic not in practice exam - assign the average raw score
                        user_conf = topic_confidence.get(exam_topic, 1)
                        
                        # Scale the average score based on confidence: lower confidence = higher priority
                        # Formula: scaled_score = average_raw_score * (7 - confidence) / 7
                        # This makes low confidence topics get higher scores (higher priority)
                        confidence_factor = (7 - user_conf) / 7
                        scaled_score = round(average_raw_score * confidence_factor, 2)
                        
                        missing_topics.append(exam_topic)
                        
                        topics_list.append({
                            "id": exam_topic,
                            "name": exam_topic,
                            "score_value": scaled_score,
                            "user_confidence": user_conf,
                            "questions_covering": [],
                            "total_questions": 0,
                            "appears_in_practice": False,
                            "raw_score": average_raw_score,
                            "confidence_scaled": True,
                            "scaled_score": scaled_score
                        })
                        print(f"⚠️ Topic '{exam_topic}' NOT found in practice exam - average score: {average_raw_score:.2f}, confidence-scaled: {scaled_score:.2f} (confidence: {user_conf})")
                
                if missing_topics:
                    print(f"📝 Missing topics (assigned average score): {', '.join(missing_topics)}")
                    print(f"💡 These topics were assigned the average raw score ({average_raw_score:.2f}) and then scaled by confidence")
            else:
                print(f"⚠️ No topics data, using fallback method")
                # Fallback: extract from questions if topics data not available
                questions = practice_exam_data.get("questions", [])
                print(f"🔍 Questions from practice exam: {len(questions)} questions")
                
                # Calculate topic scores based on questions that cover each topic
                topic_scores = {}
                for topic in exam_topics:
                    topic_scores[topic] = {
                        "name": topic,
                        "score_value": 0.0,
                        "user_confidence": topic_confidence.get(topic, 1),
                        "questions_covering": [],
                        "total_questions": 0
                    }
                
                # Analyze which questions cover which topics
                for question in questions:
                    question_topics = question.get("topics", [])
                    question_score = question.get("score", 0)
                    
                    for topic in question_topics:
                        if topic in topic_scores:
                            topic_scores[topic]["score_value"] += question_score
                            topic_scores[topic]["questions_covering"].append({
                                "question": question.get("question_text", ""),
                                "score": question_score
                            })
                            topic_scores[topic]["total_questions"] += 1
                
                # Calculate average score from topics that have scores
                practice_scores = [data["score_value"] for data in topic_scores.values() if data["score_value"] > 0]
                if practice_scores:
                    average_raw_score = round(sum(practice_scores) / len(practice_scores), 2)
                    print(f"📊 Fallback: Average raw score from practice exam topics: {average_raw_score:.2f}")
                else:
                    total_score = practice_exam_data.get("total_score", 100.0)
                    average_raw_score = round(max(1.0, total_score / len(exam_topics)), 2)
                    print(f"⚠️ Fallback: No practice scores, using default average: {average_raw_score:.2f}")
                
                # Convert to list format for priority calculation
                topics_list = []
                for topic_name, topic_data in topic_scores.items():
                    if topic_data["score_value"] > 0:
                        # Topic appeared in practice exam
                        topics_list.append({
                            "id": topic_name,
                            "name": topic_name,
                            "score_value": topic_data["score_value"],
                            "user_confidence": topic_data["user_confidence"],
                            "questions_covering": topic_data["questions_covering"],
                            "total_questions": topic_data["total_questions"],
                            "appears_in_practice": True,
                            "raw_score": topic_data["score_value"]
                        })
                        print(f"📝 Fallback practice topic: {topic_name}, score: {topic_data['score_value']:.2f}, confidence: {topic_data['user_confidence']}")
                    else:
                        # Topic didn't appear in practice exam - assign average score
                        user_conf = topic_data["user_confidence"]
                        confidence_factor = (7 - user_conf) / 7
                        scaled_score = round(average_raw_score * confidence_factor, 2)
                        
                        topics_list.append({
                            "id": topic_name,
                            "name": topic_name,
                            "score_value": scaled_score,
                            "user_confidence": user_conf,
                            "questions_covering": [],
                            "total_questions": 0,
                            "appears_in_practice": False,
                            "raw_score": average_raw_score,
                            "confidence_scaled": True,
                            "scaled_score": scaled_score
                        })
                        print(f"⚠️ Fallback missing topic: {topic_name}, average score: {average_raw_score:.2f}, confidence-scaled: {scaled_score:.2f} (confidence: {user_conf})")
            
            print(f"🔍 Final topics list: {len(topics_list)} topics")
            for topic in topics_list:
                if topic.get("appears_in_practice"):
                    print(f"  - {topic['name']}: raw_score={topic['raw_score']:.2f}, final_score={topic['score_value']:.2f}, confidence={topic['user_confidence']}")
                else:
                    print(f"  - {topic['name']}: average_raw={topic['raw_score']:.2f}, confidence-scaled={topic['scaled_score']:.2f}, confidence={topic['user_confidence']}")
            
            # Calculate study priorities using the existing algorithm
            minimum_score = practice_exam_data.get("minimum_score", 0.0)
            maximum_score = practice_exam_data.get("total_score", 0.0)
            
            print(f"🔍 Calling calculate_study_priorities with: min={minimum_score}, max={maximum_score}")
            
            return await self.calculate_study_priorities(session_id, topics_list, minimum_score, maximum_score)
            
        except Exception as e:
            raise Exception(f"Failed to calculate topic-based priorities: {str(e)}")
    
    async def _determine_easier_topic(self, topic1: Dict[str, Any], topic2: Dict[str, Any]) -> Dict[str, Any]:
        """Use ChatGPT to determine which topic is easier to study"""
        try:
            # This would typically call the GPT service
            # For now, we'll use a simple heuristic based on name length and score
            # In production, this should call GPT to analyze the topics
            
            # Simple heuristic: shorter name and higher score might indicate easier topic
            if len(topic1["name"]) < len(topic2["name"]) and topic1["calculated_score"] > topic2["calculated_score"]:
                return topic1
            elif len(topic2["name"]) < len(topic1["name"]) and topic2["calculated_score"] > topic1["calculated_score"]:
                return topic2
            else:
                # Default to the one with higher calculated score
                return topic1 if topic1["calculated_score"] > topic2["calculated_score"] else topic2
                
        except Exception as e:
            # Fallback: return the topic with higher calculated score
            return topic1 if topic1["calculated_score"] > topic2["calculated_score"] else topic2
    
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
            "last_practiced": datetime.utcnow(),
            "user_confidence": 1,
            "calculated_score": 0.0,
            "study_priority": 0
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
                    topic["user_confidence"] = 1
                    topic["calculated_score"] = 0.0
                    topic["study_priority"] = 0
        
        # Rebuild queue
        await self._rebuild_queue(session_id)
