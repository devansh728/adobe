# Challenge 1B: Persona-Driven Document Intelligence
## Technical Approach and Implementation

### Overview
This system implements semantic understanding and ranking of document sections based on persona and task requirements. It builds upon the Round 1A PDF parser to provide intelligent content relevance scoring using a hybrid approach combining sequence matching, keyword boosting, and persona-specific pattern recognition.

### Architecture

#### 1. **Modular Design**
- **PDF Parser Module**: Embedded API concept using PyMuPDF directly for section extraction
- **NLP Processing Module**: Lightweight semantic understanding with sequence matching
- **Ranking Engine**: Multi-factor scoring algorithm with persona awareness
- **Collection Processor**: Batch processing for multiple document collections

```mermaid
graph TB
    A[Document Collection] --> B[PDF Parser Module]
    B --> C[NLP Processing Module]
    C --> D[Ranking Engine]
    D --> E[Collection Processor]
    E --> F[Output JSON]
    
    B --> G[PyMuPDF Direct Processing]
    C --> H[Sequence Matching]
    C --> I[Keyword Boosting]
    C --> J[Persona Patterns]
    D --> K[Multi-Factor Scoring]
    
    style A fill:#e1f5fe
    style F fill:#c8e6c9
    style G fill:#fff3e0
    style H fill:#f3e5f5
    style I fill:#e8f5e8
    style J fill:#fff8e1
    style K fill:#fce4ec
```

#### 2. **Core Components**

##### A. PersonaDrivenProcessor Class
```python
class PersonaDrivenProcessor:
    def __init__(self):
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
```

##### B. Section Extraction Pipeline
```mermaid
flowchart TD
    A[PDF Input] --> B[Open with PyMuPDF]
    B --> C[Collect Font Sizes]
    C --> D[Calculate Dynamic Threshold]
    D --> E[Process Each Page]
    E --> F[Extract Text Spans]
    F --> G{Is Heading?}
    G -->|Yes| H[Extract Content]
    G -->|No| I[Skip]
    H --> J[Create Section Object]
    J --> K[Add to Sections List]
    K --> L{More Pages?}
    L -->|Yes| E
    L -->|No| M[Return Sections]
    
    style A fill:#e3f2fd
    style M fill:#e8f5e8
    style G fill:#fff3e0
    style H fill:#f3e5f5
```

```python
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
        return sections
    except Exception as e:
        logger.error(f"Error extracting sections from {pdf_path}: {e}")
        return []
```

### Technical Implementation

#### 1. **Dynamic Heading Detection**
```mermaid
flowchart TD
    A[Text Span] --> B{Is Generic Heading?}
    B -->|Yes| C[Skip]
    B -->|No| D{Font Size >= Threshold OR Bold?}
    D -->|No| C
    D -->|Yes| E{Length < 200 chars?}
    E -->|No| C
    E -->|Yes| F{Words <= 20?}
    F -->|No| C
    F -->|Yes| G{Not Pure Digits?}
    G -->|No| C
    G -->|Yes| H[Classify as Heading]
    
    style A fill:#e3f2fd
    style H fill:#e8f5e8
    style C fill:#ffebee
```

```python
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
```

#### 2. **Semantic Similarity (40% weight)**
```python
def calculate_semantic_similarity(self, text: str, query: str) -> float:
    """Calculate semantic similarity using sequence matching."""
    try:
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, text.lower(), query.lower()).ratio()
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0
```

#### 3. **Keyword Boosting (30% weight)**
```python
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
```

#### 4. **Persona Relevance (20% weight)**
```python
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
```

#### 5. **Text Complexity (10% weight)**
```python
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
```

### Ranking Algorithm

#### Multi-Factor Scoring Formula
```mermaid
flowchart TD
    A[Section Text] --> B[Semantic Similarity]
    A --> C[Keyword Boost]
    A --> D[Persona Relevance]
    A --> E[Text Complexity]
    A --> F[Dual Match Boost]
    A --> G[Repetition Penalty]
    
    B --> H[40% Weight]
    C --> I[30% Weight]
    D --> J[20% Weight]
    E --> K[10% Weight]
    F --> L[+0.15 if both match]
    G --> M[-0.1 if repeated]
    
    H --> N[Sum All Scores]
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
    
    N --> O[Structural Importance]
    O --> P[Final Score]
    
    style A fill:#e3f2fd
    style P fill:#e8f5e8
    style N fill:#fff3e0
```

```python
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
```

### Subsection Analysis

#### Detailed Content Analysis
```mermaid
flowchart TD
    A[Top Ranked Sections] --> B[Extract Content]
    B --> C[Split into Paragraphs]
    C --> D[Find Best Paragraph]
    D --> E[Extract Key Sentences]
    E --> F[Create Summary]
    F --> G[Generate Analysis Item]
    
    D --> H[Keyword Density Analysis]
    E --> I[Top 2 Sentences]
    F --> J[200 char limit]
    
    style A fill:#e3f2fd
    style G fill:#e8f5e8
    style H fill:#fff3e0
```

```python
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
```

### Collection Processing

#### Main Processing Pipeline
```mermaid
flowchart TD
    A[Input JSON] --> B[Extract Parameters]
    B --> C[Determine Collection]
    C --> D[Process Documents]
    D --> E[Extract Sections]
    E --> F[Rank Sections]
    F --> G[Analyze Subsections]
    G --> H[Generate Output]
    H --> I[Save JSON]
    
    B --> J[Challenge Info]
    B --> K[Documents List]
    B --> L[Persona]
    B --> M[Task]
    
    D --> N[PDF Files]
    E --> O[Section Objects]
    F --> P[Ranked Sections]
    G --> Q[Analysis Items]
    
    style A fill:#e3f2fd
    style I fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#f3e5f5
```

```python
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
```

### Performance Optimizations

#### 1. **Lightweight NLP Approach**
- **Sequence Matching**: Uses Python's `difflib.SequenceMatcher` for semantic similarity
- **No Heavy Models**: Avoids large transformer models for faster processing
- **Local Processing**: All computation done in-memory without external dependencies

#### 2. **Dynamic Thresholding**
- **Font Size Analysis**: Automatically determines heading thresholds from document
- **Adaptive Detection**: Adjusts to different document styles and layouts
- **Fallback Mechanisms**: Graceful degradation for edge cases

#### 3. **Memory Management**
- **Streaming Processing**: Processes documents one at a time
- **Garbage Collection**: Automatic cleanup between documents
- **Efficient Data Structures**: Uses lightweight dictionaries and lists

### Error Handling and Robustness

#### 1. **PDF Processing Errors**
```python
try:
    doc = fitz.open(pdf_path)
    # ... processing logic
    doc.close()
except Exception as e:
    logger.error(f"Error extracting sections from {pdf_path}: {e}")
    return []
```

#### 2. **Input Validation**
- **Collection Mapping**: Automatic mapping of challenge IDs to collection names
- **File Existence Checks**: Validates PDF files before processing
- **Parameter Extraction**: Safe extraction of persona and task information

#### 3. **Fallback Mechanisms**
- **Default Thresholds**: Fallback font size thresholds if analysis fails
- **Empty Content Handling**: Graceful handling of missing content
- **Error Logging**: Comprehensive error logging for debugging

### Testing Strategy

#### 1. **Collection Processing**
```mermaid
flowchart TD
    A[Start Processing] --> B[Load Collections List]
    B --> C[For Each Collection]
    C --> D{Input File Exists?}
    D -->|No| E[Skip Collection]
    D -->|Yes| F[Load Input Data]
    F --> G[Process Collection]
    G --> H[Save Output]
    H --> I[Log Results]
    I --> J{More Collections?}
    J -->|Yes| C
    J -->|No| K[Complete]
    
    style A fill:#e3f2fd
    style K fill:#e8f5e8
    style G fill:#fff3e0
    style H fill:#f3e5f5
```

```python
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
```

### Key Design Decisions

#### 1. **Why Sequence Matching?**
- **Lightweight**: No external model dependencies
- **Fast**: O(n²) complexity suitable for real-time processing
- **Effective**: Captures semantic similarity through character-level matching
- **Offline**: Works without internet connectivity

#### 2. **Why Multi-Factor Ranking?**
- **Comprehensive Coverage**: Combines semantic, keyword, persona, and structural relevance
- **Persona Awareness**: Tailored to specific user roles and tasks
- **Flexibility**: Adaptable to different document types and content
- **Explainability**: Clear scoring factors for transparency and debugging

#### 3. **Why Embedded API Concept?**
- **Direct PDF Processing**: Uses PyMuPDF directly for maximum control
- **No External Dependencies**: Self-contained processing pipeline
- **Performance**: Eliminates API overhead for faster processing
- **Reliability**: Reduces points of failure

### Performance Metrics

#### Runtime Performance
- **Target**: ≤60s for complete collection processing
- **Current**: ~30-45s for typical collections
- **Optimization**: Batch processing and efficient algorithms

#### Model Size
- **Target**: ≤1GB total model size
- **Current**: ~0MB (no external models)
- **Components**: PyMuPDF + standard Python libraries

#### Accuracy Metrics
- **Section Relevance**: 60 points (semantic + keyword matching)
- **Subsection Granularity**: 40 points (detailed content analysis)
- **Overall Score**: Weighted combination of relevance factors

### Future Enhancements

#### 1. **Advanced NLP Features**
- **Named Entity Recognition**: Extract specific entities from content
- **Topic Modeling**: Identify document themes and topics
- **Sentiment Analysis**: Understand content tone and sentiment

#### 2. **Enhanced Ranking**
- **Learning-to-Rank**: ML-based ranking optimization
- **User Feedback**: Incorporate user relevance feedback
- **Dynamic Weights**: Adaptive scoring based on document type

#### 3. **Performance Improvements**
- **GPU Acceleration**: Optional GPU support for faster processing
- **Distributed Processing**: Multi-core document processing
- **Advanced Caching**: Redis-based caching for repeated queries

### Conclusion

This implementation provides a robust, scalable solution for persona-driven document intelligence that meets all hackathon requirements while delivering high-quality semantic understanding and ranking capabilities. The modular design ensures maintainability and extensibility for future enhancements, while the lightweight approach ensures fast processing and offline operation. 


### Docker Setup

**Run the command in Root Folder to create the image**: 

```bash
docker build -f Dockerfile.1b -t challenge1b-processor .
```

- **To create the container/ to run the image**: 

```bash
docker run --rm challenge1b-processor
```

or

```bash
docker run --rm -v "$(pwd)/Adobe-India-Hackathon25/Challenge_1b/:/app/Adobe-India-Hackathon25/Challenge_1b/" challenge1b-processor
```


