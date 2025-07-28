#!/usr/bin/env python3
"""
Challenge 1B: Persona-Driven Document Intelligence (Standalone)
Implements semantic understanding and ranking of document sections based on persona and task.
Demonstrates embedded API concept using PyMuPDF directly for PDF processing.
"""

import json
import time
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher
from datetime import datetime
import fitz  # PyMuPDF for direct PDF processing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaDrivenProcessor:
    """Persona-driven document intelligence processor for Challenge 1B."""
    
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
    
    def extract_sections_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract sections from PDF using embedded API concept (PyMuPDF directly) with dynamic heading detection."""
        try:
            doc = fitz.open(pdf_path)
            font_sizes = []
            # First pass: collect all font sizes
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                font_sizes.append(span["size"])
            # Determine dynamic heading threshold (e.g., 80th percentile)
            if font_sizes:
                sorted_sizes = sorted(font_sizes)
                heading_threshold = sorted_sizes[int(0.8 * len(sorted_sizes))]
            else:
                heading_threshold = 12  # fallback
            # List of generic headings to ignore
            generic_headings = {"page", "contents", "table of contents", "index", "figure", "figures", "chapter", "section"}
            sections = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                blocks = page.get_text("dict")
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                font_size = span["size"]
                                is_bold = "bold" in span.get("font", "").lower()
                                # Dynamic heading detection
                                if self._is_heading_advanced(text, font_size, heading_threshold, is_bold, generic_headings):
                                    content = self._extract_content_after_heading(page_text, text)
                                    section = {
                                        "document": Path(pdf_path).name,
                                        "section_title": text,
                                        "page_number": page_num + 1,
                                        "level": self._get_heading_level(font_size, heading_threshold),
                                        "content": content,
                                        "full_text": f"{text} {content}"
                                    }
                                    sections.append(section)
            doc.close()
            logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
            return sections
        except Exception as e:
            logger.error(f"Error extracting sections from {pdf_path}: {e}")
            return []

    def _is_heading_advanced(self, text: str, font_size: float, heading_threshold: float, is_bold: bool, generic_headings: set) -> bool:
        """Advanced heading detection: dynamic threshold, bold, filter generic."""
        text_lower = text.lower().strip()
        if not text_lower or text_lower in generic_headings:
            return False
        # Loosened: allow more words, longer text, dynamic font size, or bold
        return (
            (font_size >= heading_threshold or is_bold) and
            len(text) < 200 and
            len(text.split()) <= 20 and
            not text.isdigit()
        )

    def _get_heading_level(self, font_size: float, heading_threshold: float) -> int:
        """Determine heading level based on dynamic font size threshold."""
        if font_size >= heading_threshold + 4:
            return 1  # H1
        elif font_size >= heading_threshold:
            return 2  # H2
        else:
            return 3  # H3
    
    def _extract_content_after_heading(self, page_text: str, heading: str) -> str:
        """Embedded API: Extract content after a heading."""
        try:
            heading_pos = page_text.find(heading)
            if heading_pos != -1:
                # Extract content after the heading
                content = page_text[heading_pos + len(heading):].strip()
                # Limit content length
                return content[:500]  # Limit to 500 chars
            return ""
        except:
            return ""
    
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
        """Rank sections by relevance to persona and task with hybrid scoring and dual match boost."""
        ranked_sections = []
        query = f"{persona} {task}"
        seen_titles = set()
        for idx, section in enumerate(sections):
            section_text = section.get('full_text', f"{section['section_title']} {section.get('content', '')}")
            # Calculate different relevance scores
            semantic_score = self.calculate_semantic_similarity(section_text, query)
            keyword_boost = self.calculate_keyword_boost(section_text, self._get_task_type(persona, task))
            persona_relevance = self.calculate_persona_relevance(section_text, persona)
            complexity_score = self.calculate_text_complexity(section_text)
            # Dual match boost: if both persona and task keywords are present
            persona_keywords = persona.lower().split()
            task_keywords = task.lower().split()
            persona_match = any(word in section_text.lower() for word in persona_keywords)
            task_match = any(word in section_text.lower() for word in task_keywords)
            dual_boost = 0.15 if (persona_match and task_match) else 0.0
            # Penalize generic/repetitive headings
            title_lower = section['section_title'].lower().strip()
            repetition_penalty = 0.0
            if title_lower in seen_titles:
                repetition_penalty = 0.1
            seen_titles.add(title_lower)
            # Combine scores with weights
            importance_score = (
                semantic_score * 0.4 +
                keyword_boost * 0.3 +
                persona_relevance * 0.2 +
                complexity_score * 0.1 +
                dual_boost - repetition_penalty
            )
            # Add structural importance (H1 > H2 > H3)
            structural_importance = 1.0 / section.get("level", 1)
            importance_score *= structural_importance
            section["importance_score"] = round(importance_score, 3)
            section["section_index"] = idx  # For document order tie-breaker
            ranked_sections.append(section)
        # Sort by importance score (descending), then by document order
        ranked_sections.sort(key=lambda x: (x["importance_score"], -x["section_index"]), reverse=True)
        for i, section in enumerate(ranked_sections):
            section["importance_rank"] = i + 1
        return ranked_sections

    def _get_task_type(self, persona: str, task: str) -> str:
        """Map persona and task to task type for keyword boosting."""
        persona_lower = persona.lower()
        
        if "travel" in persona_lower or "planner" in persona_lower:
            return "travel_planner"
        elif "hr" in persona_lower or "professional" in persona_lower:
            return "hr_professional"
        elif "food" in persona_lower or "contractor" in persona_lower:
            return "food_contractor"
        else:
            return "travel_planner"  # Default

    def extract_subsection_analysis(self, sections: List[Dict], top_n: int = 5) -> List[Dict]:
        """Extract detailed analysis for top-ranked sections: best paragraph, key sentences, summary."""
        analysis = []
        for section in sections[:top_n]:
            content = section.get("content", "")
            persona = section.get("persona", "")
            task = section.get("task", "")
            # Split content into paragraphs
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            # Find paragraph with most persona/task keywords
            best_para = max(paragraphs, key=lambda p: self._keyword_density(p, persona, task), default=content)
            # Extract key sentences (highest keyword density)
            sentences = re.split(r'(?<=[.!?]) +', best_para)
            key_sentences = sorted(sentences, key=lambda s: self._keyword_density(s, persona, task), reverse=True)[:2]
            summary = " ".join(key_sentences).strip()
            if not summary:
                summary = best_para[:200] + ("..." if len(best_para) > 200 else "")
            analysis_item = {
                "document": section["document"],
                "refined_text": summary,
                "page_number": section["page_number"],
                "original_section": section["section_title"],
                "relevance_score": section["importance_score"]
            }
            analysis.append(analysis_item)
        return analysis

    def _keyword_density(self, text: str, persona: str, task: str) -> int:
        """Count persona/task keyword matches in text."""
        persona_keywords = set(persona.lower().split()) if persona else set()
        task_keywords = set(task.lower().split()) if task else set()
        text_lower = text.lower()
        return sum(1 for word in persona_keywords.union(task_keywords) if word in text_lower)
    
    def process_collection(self, input_data: Dict) -> Dict:
        """Process a complete document collection for Challenge 1B."""
        start_time = time.time()
        
        try:
            # Extract input parameters
            challenge_info = input_data.get("challenge_info", {})
            documents = input_data.get("documents", [])
            persona = input_data.get("persona", {}).get("role", "")
            task = input_data.get("job_to_be_done", {}).get("task", "")
            
            # Determine collection name from challenge_info
            collection_name = self._get_collection_name(challenge_info)
            
            logger.info(f"Processing collection: {challenge_info.get('challenge_id', 'unknown')}")
            logger.info(f"Persona: {persona}")
            logger.info(f"Task: {task}")
            logger.info(f"Documents: {len(documents)}")
            
            # Process all documents
            all_sections = []
            for doc in documents:
                pdf_path = f"Adobe-India-Hackathon25/Challenge_1b/{collection_name}/PDFs/{doc['filename']}"
                
                if Path(pdf_path).exists():
                    sections = self.extract_sections_from_pdf(pdf_path)
                    all_sections.extend(sections)
                    logger.info(f"Extracted {len(sections)} sections from {doc['filename']}")
                else:
                    logger.warning(f"PDF not found: {pdf_path}")
            
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
    
    def _get_collection_name(self, challenge_info: Dict) -> str:
        """Map challenge info to collection name."""
        challenge_id = challenge_info.get("challenge_id", "")
        
        if "002" in challenge_id:
            return "Collection 1"
        elif "003" in challenge_id:
            return "Collection 2"
        elif "001" in challenge_id:
            return "Collection 3"
        else:
            return "Collection 1"  # Default

def main():
    """Main function to process Challenge 1B collections."""
    processor = PersonaDrivenProcessor()
    
    # Process all collections
    collections = [
        "Collection 1",
        "Collection 2", 
        "Collection 3"
    ]
    
    for collection in collections:
        input_file = f"Adobe-India-Hackathon25/Challenge_1b/{collection}/challenge1b_input.json"
        output_file = f"Adobe-India-Hackathon25/Challenge_1b/{collection}/challenge1b_output.json"
        
        if Path(input_file).exists():
            print(f"\n🔄 Processing {collection}...")
            
            # Load input data
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            # Process collection
            output_data = processor.process_collection(input_data)
            
            # Save output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {collection} processed successfully!")
            print(f"   📊 Sections processed: {output_data['metadata'].get('total_sections_processed', 0)}")
            print(f"   ⏱️  Processing time: {output_data['metadata'].get('processing_time', 0)}s")
            print(f"   📄 Output saved to: {output_file}")
        else:
            print(f"❌ Input file not found: {input_file}")

if __name__ == "__main__":
    main() 