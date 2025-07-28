import fitz
import re
import unicodedata
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Check if scikit-learn is available
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Load the trained ML model
def load_ml_model():
    """Load the trained ML model and encoder."""
    try:
        with open('outline_extractor/heading_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('outline_extractor/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('outline_extractor/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, label_encoder, feature_names
    except Exception as e:
        print(f"Warning: Could not load ML model: {e}")
        return None, None, None

# Global variables for ML model
ML_MODEL, LABEL_ENCODER, FEATURE_NAMES = load_ml_model()

def extract_features_from_span(span: Dict[str, Any], all_spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract comprehensive features for ML model."""
    text = span["text"]
    
    # Basic text features
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'char_count': len(text),
        'is_all_caps': text.isupper() and len(text) > 2,
        'is_numbered': bool(re.match(r'^\d+\.', text)),
        'ends_with_colon': text.endswith(':'),
        'ends_with_period': text.endswith('.'),
        'contains_year': bool(re.search(r'\d{4}', text)),
        'contains_page': 'page' in text.lower(),
        'contains_number': bool(re.search(r'\d', text)),
        'font_size': span["font_size"],
        'is_bold': span["is_bold"],
        'alignment': 1 if span["alignment"] == "center" else 0,
        'y_position': span["y_center"],
        'x_position': span["x_center"],
    }
    
    # Position features (optimized)
    page_spans = [s for s in all_spans if s["page"] == span["page"]]
    if page_spans:
        y_positions = [s["y_center"] for s in page_spans]
        if y_positions:
            features['y_percentile'] = np.percentile(y_positions, 25)
            features['is_top_quarter'] = span["y_center"] <= np.percentile(y_positions, 25)
        else:
            features['y_percentile'] = 0
            features['is_top_quarter'] = False
    else:
        features['y_percentile'] = 0
        features['is_top_quarter'] = False
    
    # Font size features (optimized)
    font_sizes = [s["font_size"] for s in all_spans]
    if font_sizes:
        features['font_size_percentile'] = np.percentile(font_sizes, 75)
        features['is_large_font'] = span["font_size"] > np.percentile(font_sizes, 75)
    else:
        features['font_size_percentile'] = 0
        features['is_large_font'] = False
    
    # Pattern features
    features['starts_with_capital'] = text[0].isupper() if text else False
    features['has_multiple_capitals'] = sum(1 for c in text if c.isupper()) > 2
    features['is_short'] = len(text) < 50
    features['is_medium'] = 50 <= len(text) <= 200
    features['is_long'] = len(text) > 200
    
    # Form-specific features
    form_keywords = ['name', 'designation', 'date', 'service', 'pay', 'address', 'phone', 'email']
    features['contains_form_keyword'] = any(keyword in text.lower() for keyword in form_keywords)
    features['is_numbered_form_field'] = bool(re.match(r'^\d+\.\s+[A-Za-z]', text))
    
    # Heading pattern features
    heading_patterns = [
        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title case
        r'^[A-Z\s]+$',  # All caps
        r'^\d+\.\s+[A-Z]',  # Numbered
        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s+of\s+the\s+[A-Z]',  # "Name of the..."
    ]
    features['matches_heading_pattern'] = any(re.match(pattern, text) for pattern in heading_patterns)
    
    return features

def predict_heading_with_ml(span: Dict[str, Any], all_spans: List[Dict[str, Any]]) -> Optional[str]:
    """Use ML model to predict heading type with performance optimization."""
    if ML_MODEL is None or LABEL_ENCODER is None or FEATURE_NAMES is None:
        return None
    
    try:
        # Quick pre-filtering to avoid expensive ML prediction
        text = span["text"]
        if len(text) < 3 or len(text) > 200:  # Skip very short/long text
            return None
        
        # Extract features efficiently
        features = extract_features_from_span(span, all_spans)
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in FEATURE_NAMES:
            feature_vector.append(features.get(feature_name, 0))
        
        # Make prediction
        prediction = ML_MODEL.predict([feature_vector])[0]
        predicted_label = LABEL_ENCODER.inverse_transform([prediction])[0]
        
        # Get prediction probability
        probabilities = ML_MODEL.predict_proba([feature_vector])[0]
        max_probability = max(probabilities)
        
        # Only return prediction if confidence is high enough
        if max_probability > 0.6:  # Reduced threshold for better recall
            return predicted_label
        
        return None
    except Exception as e:
        # Silent fail for performance
        return None

def normalize_text(text: str) -> str:
    """Enhanced text normalization for complex PDFs."""
    if not text:
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Remove zero-width characters and artifacts
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]', '', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\"\'\–\—\…]', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_spans_robust(doc: fitz.Document) -> List[Dict[str, Any]]:
    """Robust span extraction with multiple fallback methods."""
    spans = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Method 1: Try detailed dict extraction
        try:
            blocks = page.get_text("dict")
            for block in blocks.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = normalize_text(span.get("text", ""))
                        if not text or len(text.strip()) < 2:
                            continue
                        
                        # Calculate positioning info
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        x0, y0, x1, y1 = bbox
                        x_center = (x0 + x1) / 2
                        y_center = (y0 + y1) / 2
                        
                        # Calculate indentation
                        page_width = page.rect.width
                        indentation = x0 / page_width if page_width > 0 else 0
                        
                        # Determine alignment
                        if x_center < page_width * 0.3:
                            alignment = "left"
                        elif x_center > page_width * 0.7:
                            alignment = "right"
                        else:
                            alignment = "center"
                        
                        span_info = {
                            "text": text,
                            "page": page_num + 1,
                            "font_size": span.get("size", 0),
                            "font_name": span.get("font", ""),
                            "is_bold": "bold" in span.get("font", "").lower() or span.get("flags", 0) & 2**4,
                            "is_italic": "italic" in span.get("font", "").lower() or span.get("flags", 0) & 2**1,
                            "bbox": bbox,
                            "x_center": x_center,
                            "y_center": y_center,
                            "indentation": indentation,
                            "alignment": alignment,
                            "line_length": len(text),
                            "is_all_caps": text.isupper() and len(text) > 2,
                            "has_numbers": bool(re.search(r'\d', text)),
                            "word_count": len(text.split())
                        }
                        spans.append(span_info)
        except Exception as e:
            print(f"Method 1 failed for page {page_num + 1}: {e}")
        
        # Method 2: Fallback to simple text extraction with positioning
        if not spans or len([s for s in spans if s["page"] == page_num + 1]) == 0:
            try:
                # Get text blocks with basic info
                text_dict = page.get_text("textdict")
                for block in text_dict:
                    text = normalize_text(block.get("text", ""))
                    if not text or len(text.strip()) < 2:
                        continue
                    
                    bbox = block.get("bbox", [0, 0, 0, 0])
                    x0, y0, x1, y1 = bbox
                    x_center = (x0 + x1) / 2
                    y_center = (y0 + y1) / 2
                    
                    page_width = page.rect.width
                    indentation = x0 / page_width if page_width > 0 else 0
                    
                    if x_center < page_width * 0.3:
                        alignment = "left"
                    elif x_center > page_width * 0.7:
                        alignment = "right"
                    else:
                        alignment = "center"
                    
                    span_info = {
                        "text": text,
                        "page": page_num + 1,
                        "font_size": 12,  # Default size
                        "font_name": "unknown",
                        "is_bold": False,
                        "is_italic": False,
                        "bbox": bbox,
                        "x_center": x_center,
                        "y_center": y_center,
                        "indentation": indentation,
                        "alignment": alignment,
                        "line_length": len(text),
                        "is_all_caps": text.isupper() and len(text) > 2,
                        "has_numbers": bool(re.search(r'\d', text)),
                        "word_count": len(text.split())
                    }
                    spans.append(span_info)
            except Exception as e:
                print(f"Method 2 failed for page {page_num + 1}: {e}")
        
        # Method 3: Last resort - raw text extraction
        if not spans or len([s for s in spans if s["page"] == page_num + 1]) == 0:
            try:
                raw_text = page.get_text()
                if raw_text.strip():
                    # Split into lines and create basic spans
                    lines = raw_text.split('\n')
                    for i, line in enumerate(lines):
                        line = normalize_text(line)
                        if not line or len(line.strip()) < 2:
                            continue
                        
                        # Estimate position based on line number
                        y_center = (i + 1) * 20  # Rough estimate
                        
                        span_info = {
                            "text": line,
                            "page": page_num + 1,
                            "font_size": 12,
                            "font_name": "unknown",
                            "is_bold": False,
                            "is_italic": False,
                            "bbox": [0, y_center, page.rect.width, y_center + 20],
                            "x_center": page.rect.width / 2,
                            "y_center": y_center,
                            "indentation": 0,
                            "alignment": "left",
                            "line_length": len(line),
                            "is_all_caps": line.isupper() and len(line) > 2,
                            "has_numbers": bool(re.search(r'\d', line)),
                            "word_count": len(line.split())
                        }
                        spans.append(span_info)
            except Exception as e:
                print(f"Method 3 failed for page {page_num + 1}: {e}")
    
    return spans

def extract_spans(doc: fitz.Document) -> List[Dict[str, Any]]:
    """Extract detailed span information with enhanced metadata."""
    spans = extract_spans_robust(doc)
    
    # Debug: Print span count
    print(f"Extracted {len(spans)} spans")
    if spans:
        print(f"Sample spans: {[s['text'][:30] for s in spans[:3]]}")
    
    return spans

# --- Table/Banner/Template Filtering ---
def find_repeated_lines(spans: List[Dict[str, Any]], min_pages=2) -> set:
    """Find lines that repeat on multiple pages (likely banners/footers/templates)."""
    line_counts = defaultdict(set)  # text -> set of pages
    for span in spans:
        line = span["text"].strip().lower()
        if len(line) < 3:
            continue
        line_counts[line].add(span["page"])
    repeated = {line for line, pages in line_counts.items() if len(pages) >= min_pages}
    return repeated

def is_table_row(span: Dict[str, Any], spans: List[Dict[str, Any]]) -> bool:
    """Heuristic: Table rows have many short, similarly aligned spans on the same y_center."""
    y = span["y_center"]
    same_line = [s for s in spans if abs(s["y_center"] - y) < 2 and s["page"] == span["page"]]
    # Table row: many short spans, or many columns
    if len(same_line) >= 4 and all(len(s["text"]) < 30 for s in same_line):
        return True
    return False

def is_template_artifact(text: str) -> bool:
    """Detect common template artifacts (page numbers, banners, etc.)."""
    text = text.strip().lower()
    if re.match(r'^page \d+( of \d+)?$', text):
        return True
    if "confidential" in text or "do not copy" in text:
        return True
    if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', text):  # date
        return True
    return False

# --- Multilingual/Script-Aware Enhancements ---
def detect_script(text: str) -> str:
    """Detect the script/writing system of the text."""
    if not text:
        return "unknown"
    # Unicode block detection
    for c in text:
        code = ord(c)
        if 0x4E00 <= code <= 0x9FFF:
            return 'CJK'
        if 0x0900 <= code <= 0x097F:
            return 'Devanagari'
        if 0x3040 <= code <= 0x30FF:
            return 'Japanese'
        if 0x0400 <= code <= 0x04FF:
            return 'Cyrillic'
        if 0x0600 <= code <= 0x06FF:
            return 'Arabic'
    return 'Latin'

def tokenize_text(text: str) -> List[str]:
    """Tokenize text for better analysis."""
    # Remove punctuation and split
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

# Enhanced heading patterns for complex PDFs including forms
HEADING_PATTERNS = [
    r'^[A-Z][A-Z\s]+$',  # ALL CAPS
    r'^\d+\.\s+[A-Z]',   # Numbered headings
    r'^\d+\.\s+[a-z]',   # Numbered items (lowercase)
    r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title Case
    r'^[A-Z][a-z]+\s*[:\-]\s*',  # Headings with colon/dash
    r'^[IVX]+\.\s+',  # Roman numerals
    r'^[A-Z]\s*[\.\-]\s*',  # Single letter headings
    r'^\d+\.\s+[A-Za-z]',  # Numbered items
    r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*:',  # Title with colon
    r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*[:\-]',  # Title with colon/dash
    r'^\d+\.\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*',  # Numbered field labels
]

# Multilingual heading keywords
HEADING_KEYWORDS = {
    "english": [
        "abstract", "introduction", "method", "methodology", "results", "conclusion", 
        "discussion", "references", "bibliography", "appendix", "acknowledgments",
        "related work", "background", "problem", "solution", "evaluation", "analysis",
        "implementation", "experiment", "study", "survey", "review", "overview"
    ],
    "hindi": ["सारांश", "परिचय", "विधि", "परिणाम", "निष्कर्ष", "संदर्भ"],
    "chinese": ["摘要", "引言", "方法", "结果", "结论", "参考文献"],
    "japanese": ["要約", "序論", "方法", "結果", "結論", "参考文献"],
    "korean": ["초록", "서론", "방법", "결과", "결론", "참고문헌"],
    "arabic": ["ملخص", "مقدمة", "طريقة", "نتائج", "استنتاج", "مراجع"],
    "russian": ["аннотация", "введение", "метод", "результаты", "заключение", "литература"]
}

# Author patterns to filter out
AUTHOR_PATTERNS = [
    r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last
    r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Middle Last
    r'^[A-Z]\.[A-Z]\.\s+[A-Z][a-z]+$',  # F.M. Last
    r'^[A-Z][a-z]+\s+et\s+al\.?$',  # Author et al.
    r'^[A-Z][a-z]+\s+and\s+[A-Z][a-z]+$',  # Author and Author
]

# Form-specific keywords and patterns
FORM_KEYWORDS = [
    "name", "designation", "date", "service", "pay", "permanent", "temporary",
    "address", "phone", "email", "department", "section", "division",
    "application", "form", "grant", "advance", "leave", "travel",
    "document", "certificate", "approval", "authorization", "signature",
    "witness", "attestation", "verification", "endorsement", "recommendation",
    "mission", "goals", "objectives", "summary", "overview", "introduction",
    "acknowledgements", "references", "appendix", "conclusion"
]

def is_author_line(text: str) -> bool:
    """Check if text looks like an author line."""
    text = text.strip()
    for pattern in AUTHOR_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False

def is_form_field(text: str) -> bool:
    """Check if text looks like a form field label."""
    text_lower = text.lower().strip()
    
    # Check for numbered patterns like "1. Name of..." (high priority)
    if re.match(r'^\d+\.\s+[A-Za-z]', text):
        return True
    
    # Check for patterns like "Name of the Government Servant"
    if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s+of\s+the\s+[A-Z]', text):
        return True
    
    # Check for form keywords (but be more selective)
    form_keywords = [
        "name", "designation", "date", "service", "pay", "permanent", "temporary",
        "address", "phone", "email", "department", "section", "division",
        "application", "form", "grant", "advance", "leave", "travel"
    ]
    
    for keyword in form_keywords:
        if keyword in text_lower and len(text) < 50:  # Must be reasonably short
            return True
    
    # Check for ALL CAPS field labels (but be selective)
    if text.isupper() and 3 < len(text) < 30:
        return True
    
    # Check for patterns ending with colon (but not too long)
    if text.endswith(':') and 3 < len(text) < 40:
        return True
    
    return False

def is_heading(span: Dict[str, Any], title: str, size_to_level: Dict[float, int], spans: List[Dict[str, Any]], repeated_lines: set) -> Optional[str]:
    """Hybrid heading detection: Fast rules first, ML for uncertain cases."""
    text = span["text"]
    
    # Skip if it's the title
    if text.lower() == title.lower():
        return None
    # Skip author lines
    if is_author_line(text):
        return None
    # Skip very short or very long text
    if len(text) < 3 or len(text) > 100:
        return None
    # Skip page numbers
    if re.match(r'^\d+$', text):
        return None
    # Skip full sentences (likely body text)
    if text.endswith('.') and len(text.split()) > 6:
        return None
    # Skip repeated template artifacts
    if text.strip().lower() in repeated_lines:
        return None
    if is_template_artifact(text):
        return None
    # Skip table rows/cells
    if is_table_row(span, spans):
        return None
    
    # FAST RULE-BASED DETECTION (Primary)
    score = 0
    
    # Check for form field patterns (high priority)
    if is_form_field(text):
        score += 8
        if score >= 8:
            return "H1"  # Form fields are typically H1
    
    # Check for heading patterns
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, text):
            score += 3
            break
    
    # Check multilingual keywords
    script = detect_script(text)
    if script in HEADING_KEYWORDS:
        tokens = tokenize_text(text)
        for keyword in HEADING_KEYWORDS[script]:
            if keyword.lower() in [t.lower() for t in tokens]:
                score += 4
                break
    
    # Font-based detection
    font_size = span["font_size"]
    if font_size in size_to_level:
        if span["is_bold"]:
            score += 2
        if span["is_all_caps"] and len(text) > 2:
            score += 2
        if span["alignment"] == "center":
            score += 1
        if font_size > 14:
            score += 2
    
    # Penalize likely body text
    if len(text.split()) > 8:
        score -= 2
    if text.endswith('.'):
        score -= 1
    if re.search(r'\d{4}', text):
        score -= 1
    
    # Return heading level based on score
    if score >= 6:
        return "H1"
    elif score >= 4:
        return "H2"
    elif score >= 2:
        return "H3"
    
    # ML FALLBACK (only for uncertain cases)
    # Only use ML if rule-based score is borderline (1-3)
    if 1 <= score <= 3 and ML_MODEL is not None:
        try:
            ml_prediction = predict_heading_with_ml(span, spans)
            if ml_prediction and ml_prediction != "NOT_HEADING":
                return ml_prediction
        except:
            pass  # Silent fail for performance
    
    return None

def cluster_font_sizes(spans: List[Dict[str, Any]]) -> Dict[float, int]:
    """Cluster font sizes to determine heading levels."""
    font_sizes = [s["font_size"] for s in spans if s["font_size"] > 0]
    if not font_sizes:
        return {}
    
    # Get most common font sizes
    size_counter = Counter(font_sizes)
    common_sizes = [size for size, _ in size_counter.most_common(4)]
    
    # Sort by size (largest first) and assign levels
    sorted_sizes = sorted(common_sizes, reverse=True)
    size_to_level = {size: i + 1 for i, size in enumerate(sorted_sizes)}
    
    return size_to_level

def detect_title(spans: List[Dict[str, Any]]) -> str:
    """Enhanced title detection for complex PDFs."""
    if not spans:
        return ""
    
    # Look for the largest, topmost, centered text
    candidates = []
    
    for span in spans[:30]:  # Check first 30 spans
        text = span["text"]
        
        # Skip very short or very long text
        if len(text) < 5 or len(text) > 200:
            continue
            
        # Skip obvious non-titles
        if text.isdigit() or text.startswith('Page') or text.startswith('---'):
            continue
        
        if span["font_size"] > 14 and span["alignment"] == "center":
            score = span["font_size"] * 2  # Font size bonus
            if span["is_bold"]:
                score += 15
            if span["is_all_caps"]:
                score += 8
            if span["y_center"] < 300:  # Top of page
                score += 20
            
            # Penalize if it looks like a form field
            if is_form_field(text):
                score -= 10
            
            candidates.append((score, text))
    
    if candidates:
        # Return the highest scoring candidate
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    # Fallback: return the first large text
    for span in spans:
        if span["font_size"] > 12 and len(span["text"]) > 5:
            return span["text"]
    
    return spans[0]["text"] if spans else ""

def extract_features(span: Dict[str, Any], title: str) -> List[float]:
    """Extract features for ML model."""
    text = span["text"]
    
    features = [
        span["font_size"],
        float(span["is_bold"]),
        float(span["is_italic"]),
        span["indentation"],
        span["line_length"],
        float(span["is_all_caps"]),
        float(span["has_numbers"]),
        span["word_count"],
        float(span["alignment"] == "center"),
        float(span["alignment"] == "left"),
        float(span["alignment"] == "right"),
        span["y_center"],
        len(text) / max(len(title), 1),  # Relative to title length
        float(any(keyword in text.lower() for keyword in HEADING_KEYWORDS["english"])),
        float(re.match(r'^\d+\.', text) is not None),  # Numbered
        float(re.match(r'^[A-Z][A-Z\s]+$', text) is not None),  # ALL CAPS
        float(re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', text) is not None),  # Title Case
    ]
    
    return features

def extract_headings(spans: List[Dict[str, Any]], size_to_level: Dict[float, int], title: str) -> List[Dict[str, Any]]:
    """Extract headings using ML model + rule-based fallback."""
    headings = []
    repeated_lines = find_repeated_lines(spans, min_pages=2)
    
    for span in spans:
        text = span["text"]
        
        # Use enhanced heading detection
        heading_level = is_heading(span, title, size_to_level, spans, repeated_lines)
        
        if heading_level:
            # Convert heading level string to number
            if heading_level == "TITLE":
                continue  # Skip titles as they're handled separately
            elif heading_level == "H1":
                level = 1
            elif heading_level == "H2":
                level = 2
            elif heading_level == "H3":
                level = 3
            else:
                level = 1  # Default to H1
            
            headings.append({
                "level": level,
                "text": text,
                "page": span["page"],
                "y_center": span["y_center"]
            })
    
    # Sort by page and position
    headings.sort(key=lambda h: (h["page"], h["y_center"]))
    return headings

def build_hierarchy(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build hierarchical structure from flat headings."""
    if not headings:
        return []
    
    # Sort by page and y-position
    headings.sort(key=lambda h: (h["page"], h["y_center"]))
    
    def build_tree(heading_list: List[Dict[str, Any]], current_level: int = 1) -> List[Dict[str, Any]]:
        tree = []
        i = 0
        
        while i < len(heading_list):
            heading = heading_list[i]
            
            if heading["level"] == current_level:
                node = {
                    "level": heading["level"],
                    "text": heading["text"],
                    "page": heading["page"]
                }
                
                # Find children
                children = []
                j = i + 1
                while j < len(heading_list) and heading_list[j]["level"] > current_level:
                    children.append(heading_list[j])
                    j += 1
                
                if children:
                    node["children"] = build_tree(children, current_level + 1)
                
                tree.append(node)
                i = j
            else:
                i += 1
        
        return tree
    
    return build_tree(headings)

def extract_outline(pdf_path: str) -> Dict[str, Any]:
    """Main function to extract outline from PDF using ML model."""
    try:
        doc = fitz.open(pdf_path)
        spans = extract_spans_robust(doc)
        
        if not spans:
            return {"title": "", "headings": []}
        
        # Detect title
        title = detect_title(spans)
        
        # Cluster font sizes
        size_to_level = cluster_font_sizes(spans)
        
        # Extract headings using ML model
        headings = extract_headings(spans, size_to_level, title)
        
        # Build hierarchy
        hierarchy = build_hierarchy(headings)
        
        doc.close()
        
        return {
            "title": title,
            "headings": hierarchy
        }
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return {"title": "", "headings": []}

