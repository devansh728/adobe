#!/usr/bin/env python3
"""
Advanced ML Model Training for Title + H1-H3 Detection
Uses sample PDFs and synthetic data to train a robust classifier
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import fitz
import re
from typing import List, Dict, Any, Tuple

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
    
    # Position features
    page_spans = [s for s in all_spans if s["page"] == span["page"]]
    if page_spans:
        y_positions = [s["y_center"] for s in page_spans]
        features['y_percentile'] = np.percentile(y_positions, 25) if y_positions else 0
        features['is_top_quarter'] = span["y_center"] <= np.percentile(y_positions, 25) if y_positions else False
    
    # Font size features
    font_sizes = [s["font_size"] for s in all_spans]
    if font_sizes:
        features['font_size_percentile'] = np.percentile(font_sizes, 75) if font_sizes else 0
        features['is_large_font'] = span["font_size"] > np.percentile(font_sizes, 75) if font_sizes else False
    
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

def create_training_data_from_pdfs() -> Tuple[List[Dict], List[str]]:
    """Create training data from sample PDFs."""
    training_data = []
    labels = []
    
    # Sample PDFs to extract training data
    pdf_files = [
        'Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/file01.pdf',
        'Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/file02.pdf',
        'Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/file03.pdf',
        'Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/file04.pdf',
        'Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/file05.pdf'
    ]
    
    for pdf_file in pdf_files:
        if Path(pdf_file).exists():
            try:
                doc = fitz.open(pdf_file)
                spans = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    try:
                        blocks = page.get_text("dict")
                        for block in blocks["blocks"]:
                            if "lines" in block:
                                for line in block["lines"]:
                                    for span in line["spans"]:
                                        try:
                                            span_info = {
                                                "text": span["text"].strip(),
                                                "font_size": span["size"],
                                                "is_bold": "bold" in span["font"].lower(),
                                                "x_center": (span["bbox"][0] + span["bbox"][2]) / 2,
                                                "y_center": (span["bbox"][1] + span["bbox"][3]) / 2,
                                                "page": page_num + 1,
                                                "alignment": "center" if abs(span["bbox"][0] - 50) < 20 else "left"
                                            }
                                            if span_info["text"]:
                                                spans.append(span_info)
                                        except Exception as e:
                                            continue
                    except Exception as e:
                        continue
                
                # Manually label some obvious cases
                for span in spans:
                    text = span["text"]
                    features = extract_features_from_span(span, spans)
                    
                    # Label based on heuristics
                    label = "NOT_HEADING"
                    
                    # Title detection
                    if (span["font_size"] > 16 and span["y_center"] < 300 and 
                        span["alignment"] == "center" and len(text) > 5):
                        label = "TITLE"
                    
                    # H1 detection
                    elif ((span["font_size"] > 14 and span["is_bold"]) or
                          (span["font_size"] > 16) or
                          (features['is_numbered_form_field'] and span["font_size"] > 12)):
                        label = "H1"
                    
                    # H2 detection
                    elif ((span["font_size"] > 12 and span["is_bold"]) or
                          (span["font_size"] > 14) or
                          (features['contains_form_keyword'] and span["font_size"] > 10)):
                        label = "H2"
                    
                    # H3 detection
                    elif ((span["font_size"] > 11 and span["is_bold"]) or
                          (span["font_size"] > 13)):
                        label = "H3"
                    
                    # Skip very short or very long text
                    if len(text) < 3 or len(text) > 300:
                        label = "NOT_HEADING"
                    
                    training_data.append(features)
                    labels.append(label)
                
                doc.close()
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue
    
    return training_data, labels

def create_synthetic_data() -> Tuple[List[Dict], List[str]]:
    """Create synthetic training data for better model coverage."""
    synthetic_data = []
    synthetic_labels = []
    
    # Synthetic titles
    titles = [
        "Application Form for Grant of LTC Advance",
        "Research Proposal on Machine Learning",
        "Annual Report 2024",
        "Project Implementation Plan",
        "Technical Documentation"
    ]
    
    for title in titles:
        features = {
            'text_length': len(title),
            'word_count': len(title.split()),
            'char_count': len(title),
            'is_all_caps': title.isupper(),
            'is_numbered': False,
            'ends_with_colon': False,
            'ends_with_period': False,
            'contains_year': bool(re.search(r'\d{4}', title)),
            'contains_page': False,
            'contains_number': bool(re.search(r'\d', title)),
            'font_size': 18,
            'is_bold': True,
            'alignment': 1,
            'y_position': 100,
            'x_position': 300,
            'y_percentile': 10,
            'is_top_quarter': True,
            'font_size_percentile': 90,
            'is_large_font': True,
            'starts_with_capital': True,
            'has_multiple_capitals': True,
            'is_short': len(title) < 50,
            'is_medium': 50 <= len(title) <= 200,
            'is_long': len(title) > 200,
            'contains_form_keyword': False,
            'is_numbered_form_field': False,
            'matches_heading_pattern': True
        }
        synthetic_data.append(features)
        synthetic_labels.append("TITLE")
    
    # Synthetic H1 headings
    h1_headings = [
        "1. Name of the Government Servant",
        "2. Designation of the Officer",
        "3. Date of Application",
        "Introduction",
        "Methodology",
        "Results",
        "Conclusion",
        "References"
    ]
    
    for heading in h1_headings:
        features = {
            'text_length': len(heading),
            'word_count': len(heading.split()),
            'char_count': len(heading),
            'is_all_caps': heading.isupper(),
            'is_numbered': bool(re.match(r'^\d+\.', heading)),
            'ends_with_colon': heading.endswith(':'),
            'ends_with_period': heading.endswith('.'),
            'contains_year': False,
            'contains_page': False,
            'contains_number': bool(re.search(r'\d', heading)),
            'font_size': 16,
            'is_bold': True,
            'alignment': 0,
            'y_position': 200,
            'x_position': 100,
            'y_percentile': 30,
            'is_top_quarter': False,
            'font_size_percentile': 85,
            'is_large_font': True,
            'starts_with_capital': True,
            'has_multiple_capitals': True,
            'is_short': len(heading) < 50,
            'is_medium': 50 <= len(heading) <= 200,
            'is_long': len(heading) > 200,
            'contains_form_keyword': any(keyword in heading.lower() for keyword in ['name', 'designation', 'date']),
            'is_numbered_form_field': bool(re.match(r'^\d+\.\s+[A-Za-z]', heading)),
            'matches_heading_pattern': True
        }
        synthetic_data.append(features)
        synthetic_labels.append("H1")
    
    # Synthetic H2 headings
    h2_headings = [
        "Personal Information",
        "Contact Details",
        "Educational Background",
        "Work Experience",
        "Skills and Qualifications",
        "Additional Information"
    ]
    
    for heading in h2_headings:
        features = {
            'text_length': len(heading),
            'word_count': len(heading.split()),
            'char_count': len(heading),
            'is_all_caps': heading.isupper(),
            'is_numbered': False,
            'ends_with_colon': False,
            'ends_with_period': False,
            'contains_year': False,
            'contains_page': False,
            'contains_number': False,
            'font_size': 14,
            'is_bold': True,
            'alignment': 0,
            'y_position': 300,
            'x_position': 120,
            'y_percentile': 50,
            'is_top_quarter': False,
            'font_size_percentile': 75,
            'is_large_font': True,
            'starts_with_capital': True,
            'has_multiple_capitals': True,
            'is_short': len(heading) < 50,
            'is_medium': 50 <= len(heading) <= 200,
            'is_long': len(heading) > 200,
            'contains_form_keyword': False,
            'is_numbered_form_field': False,
            'matches_heading_pattern': True
        }
        synthetic_data.append(features)
        synthetic_labels.append("H2")
    
    # Synthetic H3 headings
    h3_headings = [
        "Full Name",
        "Date of Birth",
        "Phone Number",
        "Email Address",
        "Current Address",
        "Permanent Address"
    ]
    
    for heading in h3_headings:
        features = {
            'text_length': len(heading),
            'word_count': len(heading.split()),
            'char_count': len(heading),
            'is_all_caps': heading.isupper(),
            'is_numbered': False,
            'ends_with_colon': False,
            'ends_with_period': False,
            'contains_year': False,
            'contains_page': False,
            'contains_number': False,
            'font_size': 12,
            'is_bold': True,
            'alignment': 0,
            'y_position': 400,
            'x_position': 140,
            'y_percentile': 70,
            'is_top_quarter': False,
            'font_size_percentile': 60,
            'is_large_font': False,
            'starts_with_capital': True,
            'has_multiple_capitals': True,
            'is_short': len(heading) < 50,
            'is_medium': 50 <= len(heading) <= 200,
            'is_long': len(heading) > 200,
            'contains_form_keyword': any(keyword in heading.lower() for keyword in ['name', 'date', 'phone', 'email', 'address']),
            'is_numbered_form_field': False,
            'matches_heading_pattern': True
        }
        synthetic_data.append(features)
        synthetic_labels.append("H3")
    
    # Synthetic NOT_HEADING examples
    not_headings = [
        "This is a sample paragraph that contains regular text content.",
        "The document provides information about various topics.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Page 1 of 10",
        "Confidential",
        "Draft Version",
        "2024-01-15"
    ]
    
    for text in not_headings:
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text),
            'is_all_caps': text.isupper(),
            'is_numbered': False,
            'ends_with_colon': False,
            'ends_with_period': text.endswith('.'),
            'contains_year': bool(re.search(r'\d{4}', text)),
            'contains_page': 'page' in text.lower(),
            'contains_number': bool(re.search(r'\d', text)),
            'font_size': 10,
            'is_bold': False,
            'alignment': 0,
            'y_position': 500,
            'x_position': 50,
            'y_percentile': 80,
            'is_top_quarter': False,
            'font_size_percentile': 40,
            'is_large_font': False,
            'starts_with_capital': text[0].isupper() if text else False,
            'has_multiple_capitals': sum(1 for c in text if c.isupper()) > 2,
            'is_short': len(text) < 50,
            'is_medium': 50 <= len(text) <= 200,
            'is_long': len(text) > 200,
            'contains_form_keyword': False,
            'is_numbered_form_field': False,
            'matches_heading_pattern': False
        }
        synthetic_data.append(features)
        synthetic_labels.append("NOT_HEADING")
    
    return synthetic_data, synthetic_labels

def train_model():
    """Train the ML model for Title + H1-H3 detection."""
    print("🚀 Training Advanced ML Model for Title + H1-H3 Detection")
    print("=" * 60)
    
    # Create training data
    print("📊 Creating training data from sample PDFs...")
    pdf_data, pdf_labels = create_training_data_from_pdfs()
    
    print("📊 Creating synthetic training data...")
    synthetic_data, synthetic_labels = create_synthetic_data()
    
    # Combine data
    all_data = pdf_data + synthetic_data
    all_labels = pdf_labels + synthetic_labels
    
    print(f"📈 Total training samples: {len(all_data)}")
    print(f"📊 Label distribution:")
    for label in set(all_labels):
        count = all_labels.count(label)
        print(f"  {label}: {count}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    # Train model
    print("\n🤖 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\n📊 Model Evaluation:")
    y_pred = model.predict(X_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, df, encoded_labels, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': df.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save model and encoder
    print("\n💾 Saving model and encoder...")
    with open('outline_extractor/heading_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('outline_extractor/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save feature names for later use
    with open('outline_extractor/feature_names.json', 'w') as f:
        json.dump(list(df.columns), f)
    
    print("✅ Model training completed successfully!")
    print(f"📁 Model saved to: outline_extractor/heading_model.pkl")
    print(f"📁 Encoder saved to: outline_extractor/label_encoder.pkl")
    print(f"📁 Feature names saved to: outline_extractor/feature_names.json")

if __name__ == "__main__":
    train_model() 