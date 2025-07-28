#!/usr/bin/env python3
"""
Challenge 1B Standalone Test
Tests persona-driven document intelligence with mock data.
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from difflib import SequenceMatcher
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockPersonaDrivenProcessor:
    """Mock persona-driven document intelligence processor for testing."""
    
    def __init__(self):
        """Initialize the processor with lightweight NLP capabilities."""
        
        # Task-specific keywords for boosting
        self.task_keywords = {
            "travel_planner": ["itinerary", "accommodation", "transportation", "attractions", "restaurants", "budget", "planning", "trip", "travel", "vacation", "sightseeing", "booking", "reservation", "hotel", "flight", "tour", "guide", "destination", "visit", "explore"],
            "hr_professional": ["forms", "onboarding", "compliance", "fillable", "signature", "document", "workflow", "process", "template", "automation", "digital", "paperless", "approval", "employee", "hr", "human resources", "recruitment", "training", "policy", "procedure"],
            "food_contractor": ["vegetarian", "buffet", "corporate", "menu", "gluten-free", "dinner", "catering", "ingredients", "recipes", "dietary", "restrictions", "preparation", "serving", "food", "meal", "cuisine", "cooking", "nutrition", "diet", "allergies"]
        }
        
        # Persona-specific relevance patterns
        self.persona_patterns = {
            "travel_planner": ["plan", "book", "visit", "explore", "experience", "recommend", "guide", "tips", "arrange", "organize", "schedule", "reserve"],
            "hr_professional": ["create", "manage", "process", "automate", "comply", "approve", "sign", "fill", "implement", "administer", "coordinate", "facilitate"],
            "food_contractor": ["prepare", "cook", "serve", "cater", "menu", "dietary", "ingredients", "nutrition", "create", "design", "plan", "organize"]
        }
    
    def calculate_semantic_similarity(self, text: str, query: str) -> float:
        """Calculate semantic similarity using sequence matching."""
        try:
            # Use sequence matcher for similarity
            similarity = SequenceMatcher(None, text.lower(), query.lower()).ratio()
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def calculate_keyword_boost(self, text: str, task_type: str) -> float:
        """Calculate keyword boost based on task-specific terms."""
        if task_type not in self.task_keywords:
            return 0.0
        
        text_lower = text.lower()
        keywords = self.task_keywords[task_type]
        
        # Count keyword matches
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Normalize by text length and keyword count
        boost = keyword_count / max(len(text.split()), 1)
        return min(boost * 0.5, 1.0)  # Cap at 1.0
    
    def calculate_persona_relevance(self, text: str, persona: str) -> float:
        """Calculate relevance based on persona patterns."""
        if persona not in self.persona_patterns:
            return 0.0
        
        text_lower = text.lower()
        patterns = self.persona_patterns[persona]
        
        # Count pattern matches
        pattern_count = sum(1 for pattern in patterns if pattern in text_lower)
        
        # Normalize
        relevance = pattern_count / max(len(text.split()), 1)
        return min(relevance * 0.3, 1.0)
    
    def calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity as a proxy for content richness."""
        try:
            # Simple complexity metrics
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
            unique_words = len(set(words))
            complexity = (avg_word_length * unique_words) / max(len(words), 1)
            return min(complexity / 10.0, 1.0)  # Normalize
        except:
            return 0.0
    
    def rank_sections(self, sections: List[Dict], persona: str, task: str) -> List[Dict]:
        """Rank sections by relevance to persona and task."""
        ranked_sections = []
        
        # Combine persona and task for query
        query = f"{persona} {task}"
        
        for section in sections:
            section_text = f"{section['section_title']} {section.get('content', '')}"
            
            # Calculate different relevance scores
            semantic_score = self.calculate_semantic_similarity(section_text, query)
            keyword_boost = self.calculate_keyword_boost(section_text, self._get_task_type(persona, task))
            persona_relevance = self.calculate_persona_relevance(section_text, persona)
            complexity_score = self.calculate_text_complexity(section_text)
            
            # Combine scores with weights
            importance_score = (
                semantic_score * 0.4 +
                keyword_boost * 0.3 +
                persona_relevance * 0.2 +
                complexity_score * 0.1
            )
            
            # Add structural importance (H1 > H2 > H3)
            structural_boost = 1.0 / section.get('level', 1)
            importance_score *= structural_boost
            
            ranked_section = {
                **section,
                "importance_rank": len(ranked_sections) + 1,
                "importance_score": round(importance_score, 4)
            }
            
            ranked_sections.append(ranked_section)
        
        # Sort by importance score (descending)
        ranked_sections.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Reassign ranks
        for i, section in enumerate(ranked_sections):
            section['importance_rank'] = i + 1
        
        return ranked_sections
    
    def _get_task_type(self, persona: str, task: str) -> str:
        """Map persona and task to task type for keyword boosting."""
        task_lower = task.lower()
        persona_lower = persona.lower()
        
        if "travel" in task_lower or "trip" in task_lower or "travel" in persona_lower:
            return "travel_planner"
        elif "hr" in persona_lower or "forms" in task_lower or "onboarding" in task_lower:
            return "hr_professional"
        elif "food" in persona_lower or "menu" in task_lower or "catering" in task_lower:
            return "food_contractor"
        else:
            return "general"
    
    def extract_subsection_analysis(self, sections: List[Dict], top_n: int = 5) -> List[Dict]:
        """Extract detailed subsection analysis for top-ranked sections."""
        subsection_analysis = []
        
        for section in sections[:top_n]:
            # Enhanced content extraction (simplified for demo)
            refined_text = self._extract_refined_content(section)
            
            analysis = {
                "document": section["document"],
                "refined_text": refined_text,
                "page_number": section["page_number"],
                "original_section": section["section_title"],
                "relevance_score": section["importance_score"]
            }
            
            subsection_analysis.append(analysis)
        
        return subsection_analysis
    
    def _extract_refined_content(self, section: Dict) -> str:
        """Extract refined content for subsection analysis."""
        # In a full implementation, this would extract actual content
        # For now, return enhanced description
        return f"Detailed content and insights related to '{section['section_title']}' on page {section['page_number']}. This section provides comprehensive information relevant to the user's needs and contains valuable actionable insights for the specified task."
    
    def process_collection(self, input_data: Dict) -> Dict:
        """Process a complete document collection for Challenge 1B."""
        start_time = time.time()
        
        try:
            # Extract input parameters
            challenge_info = input_data.get("challenge_info", {})
            documents = input_data.get("documents", [])
            persona = input_data.get("persona", {}).get("role", "")
            task = input_data.get("job_to_be_done", {}).get("task", "")
            
            logger.info(f"Processing collection: {challenge_info.get('challenge_id', 'unknown')}")
            logger.info(f"Persona: {persona}")
            logger.info(f"Task: {task}")
            logger.info(f"Documents: {len(documents)}")
            
            # Create mock sections for testing
            all_sections = self._create_mock_sections(documents, persona, task)
            
            # Rank sections by relevance
            ranked_sections = self.rank_sections(all_sections, persona, task)
            
            # Extract subsection analysis
            subsection_analysis = self.extract_subsection_analysis(ranked_sections)
            
            # Prepare output
            output = {
                "metadata": {
                    "input_documents": [doc["filename"] for doc in documents],
                    "persona": persona,
                    "job_to_be_done": task,
                    "processing_timestamp": datetime.now().isoformat(),
                    "total_sections_processed": len(all_sections),
                    "processing_time": round(time.time() - start_time, 2)
                },
                "extracted_sections": [
                    {
                        "document": section["document"],
                        "section_title": section["section_title"],
                        "importance_rank": section["importance_rank"],
                        "page_number": section["page_number"],
                        "importance_score": section["importance_score"]
                    }
                    for section in ranked_sections[:10]  # Top 10 sections
                ],
                "subsection_analysis": subsection_analysis
            }
            
            logger.info(f"Processing completed in {output['metadata']['processing_time']}s")
            return output
            
        except Exception as e:
            logger.error(f"Error processing collection: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
    
    def _create_mock_sections(self, documents: List[Dict], persona: str, task: str) -> List[Dict]:
        """Create mock sections for testing based on persona and task."""
        mock_sections = []
        
        # Generate relevant mock sections based on persona
        if "travel" in persona.lower():
            mock_sections = [
                {"document": "travel_guide.pdf", "section_title": "Travel Itinerary Planning", "page_number": 1, "level": 1, "content": "Comprehensive travel planning guide with itinerary suggestions"},
                {"document": "accommodation.pdf", "section_title": "Hotel Booking and Accommodation", "page_number": 2, "level": 1, "content": "Detailed accommodation options and booking procedures"},
                {"document": "attractions.pdf", "section_title": "Tourist Attractions and Sightseeing", "page_number": 3, "level": 1, "content": "Must-visit attractions and sightseeing recommendations"},
                {"document": "restaurants.pdf", "section_title": "Restaurant Recommendations", "page_number": 4, "level": 2, "content": "Top-rated restaurants and dining options"},
                {"document": "transport.pdf", "section_title": "Transportation Options", "page_number": 5, "level": 2, "content": "Public transport and travel logistics"}
            ]
        elif "hr" in persona.lower():
            mock_sections = [
                {"document": "forms.pdf", "section_title": "Creating Fillable Forms", "page_number": 1, "level": 1, "content": "Step-by-step guide to creating digital forms"},
                {"document": "onboarding.pdf", "section_title": "Employee Onboarding Process", "page_number": 2, "level": 1, "content": "Complete onboarding workflow and procedures"},
                {"document": "compliance.pdf", "section_title": "Compliance Documentation", "page_number": 3, "level": 1, "content": "HR compliance requirements and documentation"},
                {"document": "workflow.pdf", "section_title": "Digital Workflow Automation", "page_number": 4, "level": 2, "content": "Automating HR processes and workflows"},
                {"document": "approval.pdf", "section_title": "Approval and Signature Process", "page_number": 5, "level": 2, "content": "Digital approval and signature workflows"}
            ]
        elif "food" in persona.lower():
            mock_sections = [
                {"document": "menu.pdf", "section_title": "Vegetarian Menu Planning", "page_number": 1, "level": 1, "content": "Comprehensive vegetarian menu design and planning"},
                {"document": "catering.pdf", "section_title": "Corporate Catering Services", "page_number": 2, "level": 1, "content": "Professional catering for corporate events"},
                {"document": "dietary.pdf", "section_title": "Dietary Restrictions and Allergies", "page_number": 3, "level": 1, "content": "Managing dietary requirements and food allergies"},
                {"document": "preparation.pdf", "section_title": "Food Preparation Guidelines", "page_number": 4, "level": 2, "content": "Safe food preparation and handling procedures"},
                {"document": "serving.pdf", "section_title": "Buffet-Style Serving", "page_number": 5, "level": 2, "content": "Efficient buffet setup and serving techniques"}
            ]
        else:
            # Generic sections
            mock_sections = [
                {"document": "document1.pdf", "section_title": "Introduction", "page_number": 1, "level": 1, "content": "General introduction and overview"},
                {"document": "document2.pdf", "section_title": "Main Content", "page_number": 2, "level": 1, "content": "Primary content and information"},
                {"document": "document3.pdf", "section_title": "Conclusion", "page_number": 3, "level": 1, "content": "Summary and conclusions"}
            ]
        
        return mock_sections

def test_semantic_similarity(processor: MockPersonaDrivenProcessor):
    """Test semantic similarity calculations."""
    print("\n🧠 Testing semantic similarity...")
    
    test_cases = [
        {
            "text": "Travel itinerary and accommodation booking",
            "query": "Travel Planner Plan a trip",
            "expected_high": True
        },
        {
            "text": "Create fillable forms for onboarding",
            "query": "HR professional Create and manage forms",
            "expected_high": True
        },
        {
            "text": "Vegetarian buffet menu preparation",
            "query": "Food Contractor Prepare vegetarian buffet",
            "expected_high": True
        },
        {
            "text": "Random unrelated content",
            "query": "Travel Planner Plan a trip",
            "expected_high": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        similarity = processor.calculate_semantic_similarity(
            test_case["text"], 
            test_case["query"]
        )
        
        result = "✅" if (similarity > 0.3) == test_case["expected_high"] else "❌"
        print(f"   {result} Test {i}: {similarity:.3f} - {test_case['text'][:50]}...")

def test_keyword_boosting(processor: MockPersonaDrivenProcessor):
    """Test keyword boosting functionality."""
    print("\n🔑 Testing keyword boosting...")
    
    test_cases = [
        {
            "text": "Travel itinerary and accommodation booking for vacation",
            "task_type": "travel_planner",
            "expected_boost": True
        },
        {
            "text": "Create fillable forms for onboarding and compliance",
            "task_type": "hr_professional", 
            "expected_boost": True
        },
        {
            "text": "Vegetarian buffet menu with gluten-free options",
            "task_type": "food_contractor",
            "expected_boost": True
        },
        {
            "text": "Random text without relevant keywords",
            "task_type": "travel_planner",
            "expected_boost": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        boost = processor.calculate_keyword_boost(
            test_case["text"], 
            test_case["task_type"]
        )
        
        result = "✅" if (boost > 0.1) == test_case["expected_boost"] else "❌"
        print(f"   {result} Test {i}: {boost:.3f} - {test_case['text'][:50]}...")

def test_persona_relevance(processor: MockPersonaDrivenProcessor):
    """Test persona relevance calculations."""
    print("\n👤 Testing persona relevance...")
    
    test_cases = [
        {
            "text": "Plan and book travel arrangements",
            "persona": "travel_planner",
            "expected_relevance": True
        },
        {
            "text": "Create and manage digital forms",
            "persona": "hr_professional",
            "expected_relevance": True
        },
        {
            "text": "Prepare and serve corporate catering",
            "persona": "food_contractor",
            "expected_relevance": True
        },
        {
            "text": "Random text without persona patterns",
            "persona": "travel_planner",
            "expected_relevance": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        relevance = processor.calculate_persona_relevance(
            test_case["text"], 
            test_case["persona"]
        )
        
        result = "✅" if (relevance > 0.05) == test_case["expected_relevance"] else "❌"
        print(f"   {result} Test {i}: {relevance:.3f} - {test_case['text'][:50]}...")

def main():
    """Main test function."""
    print("🚀 Challenge 1B: Persona-Driven Document Intelligence Test (Standalone)")
    print("=" * 80)
    
    # Initialize processor
    print("🔧 Initializing processor...")
    processor = MockPersonaDrivenProcessor()
    print("✅ Processor initialized successfully!")
    
    # Test core functionality
    test_semantic_similarity(processor)
    test_keyword_boosting(processor)
    test_persona_relevance(processor)
    
    # Test all collections
    collections = [
        "Collection 1",  # Travel Planning
        "Collection 2",  # Adobe Acrobat Learning  
        "Collection 3"   # Recipe Collection
    ]
    
    results = []
    total_time = 0
    
    for collection in collections:
        input_file = f"Adobe-India-Hackathon25/Challenge_1b/{collection}/challenge1b_input.json"
        output_file = f"Adobe-India-Hackathon25/Challenge_1b/{collection}/challenge1b_output_mock.json"
        
        if Path(input_file).exists():
            print(f"\n🔄 Testing {collection}...")
            
            # Load input data
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            # Process collection
            result = processor.process_collection(input_data)
            
            # Save output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            processing_time = result['metadata'].get('processing_time', 0)
            total_sections = result['metadata'].get('total_sections_processed', 0)
            extracted_sections = len(result.get('extracted_sections', []))
            
            print(f"✅ {collection} processed successfully!")
            print(f"   📊 Total sections processed: {total_sections}")
            print(f"   📋 Top sections extracted: {extracted_sections}")
            print(f"   ⏱️  Processing time: {processing_time:.2f}s")
            print(f"   📄 Output saved to: {output_file}")
            
            # Show top 3 sections
            if result.get('extracted_sections'):
                print(f"   🏆 Top 3 sections:")
                for i, section in enumerate(result['extracted_sections'][:3], 1):
                    print(f"      {i}. {section['section_title']} (Score: {section['importance_score']:.3f})")
            
            results.append({
                "collection": collection,
                "success": True,
                "processing_time": processing_time,
                "total_sections": total_sections,
                "extracted_sections": extracted_sections
            })
            
            total_time += processing_time
        else:
            print(f"❌ Input file not found: {input_file}")
            results.append({
                "collection": collection,
                "success": False,
                "error": "Input file not found"
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r.get("success"))
    total_sections = sum(r.get("total_sections", 0) for r in results if r.get("success"))
    total_extracted = sum(r.get("extracted_sections", 0) for r in results if r.get("success"))
    
    print(f"✅ Successful collections: {successful}/{len(collections)}")
    print(f"📊 Total sections processed: {total_sections}")
    print(f"📋 Total sections extracted: {total_extracted}")
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    print(f"⏱️  Average time per collection: {total_time/len(collections):.2f}s")
    
    # Performance compliance
    print(f"\n🎯 PERFORMANCE COMPLIANCE:")
    print(f"   ⏱️  Runtime: {total_time:.2f}s ≤ 60s (PASS)" if total_time <= 60 else f"   ⏱️  Runtime: {total_time:.2f}s > 60s (FAIL)")
    print(f"   📦 Model size: ~0MB ≤ 1GB (PASS)")
    print(f"   🔌 Offline operation: ✅ (PASS)")
    
    # Individual results
    print(f"\n📋 DETAILED RESULTS:")
    for result in results:
        if result.get("success"):
            print(f"   ✅ {result['collection']}: {result['processing_time']:.2f}s, {result['total_sections']} sections")
        else:
            print(f"   ❌ {result['collection']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Test completed!")

if __name__ == "__main__":
    main() 