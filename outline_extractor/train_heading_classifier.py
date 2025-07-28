import os
import fitz
import unicodedata
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# --- Feature extraction logic (same as extractor.py) ---
HEADING_KEYWORDS = [
    'chapter', 'section', 'part', 'introduction', 'conclusion', 'overview', 'summary', 'appendix', 'references', 'bibliography', 'contents', 'abstract',
    'cap[ií]tulo', 'secci[oó]n', 'parte', 'introducci[oó]n', 'conclusi[oó]n',
    'chapitre', 'section', 'partie', 'introduction', 'conclusion',
    'kapitel', 'abschnitt', 'teil', 'einleitung', 'schlussfolgerung',
    '章', '節', '部分', '引言', '結論', '概要',
    'अध्याय', 'अनुभाग', 'परिचय', 'निष्कर्ष',
]
HEADING_KEYWORDS_REGEX = '|'.join(HEADING_KEYWORDS)
HEADING_PATTERNS = [
    re.compile(r'^(第?\d+[章節])', re.UNICODE),
    re.compile(r'^(\d+[\.\-\s]*){1,4}\s*\w+', re.UNICODE),
    re.compile(r'^[A-Z][A-Z\s\-:]{2,}$', re.UNICODE),
    re.compile(rf'.*({HEADING_KEYWORDS_REGEX}).*', re.IGNORECASE|re.UNICODE),
]
AUTHOR_PATTERNS = [
    re.compile(r'^[A-Z][a-z]+( [A-Z][a-z]+)+$'),
    re.compile(r'@'),
    re.compile(r'(university|institute|college|school|department|lab|centre|center|faculty|academy|company|corporation|inc\.|ltd\.|llc)', re.I),
]
def is_author_line(text: str) -> bool:
    for pat in AUTHOR_PATTERNS:
        if pat.search(text):
            return True
    return False
def detect_script(text: str) -> str:
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
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\u200c', '').replace('\u200b', '')
    text = text.strip()
    return text
def tokenize_text(text: str) -> list:
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
def extract_features(span: dict, title: str) -> list:
    text = span["text"]
    tokens = span["tokens"]
    features = [
        span["font_size"],
        int(span["is_bold"]),
        int(span["is_italic"]),
        1 if span["alignment"] == "center" else 0,
        span["y_center"] / span["page_height"],
        len(text),
        len(tokens),
        int(any(pat.match(text) for pat in HEADING_PATTERNS)),
        int(any(re.search(kw, text, re.IGNORECASE|re.UNICODE) for kw in HEADING_KEYWORDS)),
        int(span["script"] != 'Latin'),
        int(is_author_line(text)),
        int(text.strip().lower() == title.strip().lower()),
        int(re.match(r'^(page|p\.?|\d{1,3})$', text.strip(), re.IGNORECASE) is not None),
        int(text.endswith('.') and len(tokens) > 5),
    ]
    return features
def extract_spans(doc: fitz.Document) -> list:
    spans = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = normalize_text(span["text"])
                    if not text:
                        continue
                    bbox = span["bbox"]
                    font_size = span["size"]
                    font_name = span["font"]
                    flags = span["flags"]
                    is_bold = bool(flags & 2**4) or "bold" in font_name.lower()
                    is_italic = bool(flags & 2**1) or "italic" in font_name.lower()
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    alignment = (
                        "center" if abs(x_center - page_width/2) < 50 else
                        "left" if bbox[0] < page_width * 0.1 else
                        "right" if bbox[2] > page_width * 0.9 else
                        "left"
                    )
                    script = detect_script(text)
                    spans.append({
                        "text": text,
                        "tokens": tokenize_text(text),
                        "font_size": font_size,
                        "font_name": font_name,
                        "is_bold": is_bold,
                        "is_italic": is_italic,
                        "alignment": alignment,
                        "bbox": bbox,
                        "x_center": x_center,
                        "y_center": y_center,
                        "page": page_num + 1,
                        "line_y": line["bbox"][1],
                        "block_y": block["bbox"][1],
                        "page_height": page_height,
                        "script": script
                    })
    return spans

def main():
    # For demo: use a few sample PDFs (user can expand this set)
    pdfs = [
        os.path.join(os.path.dirname(__file__), 'sample1.pdf'),
        os.path.join(os.path.dirname(__file__), 'sample2.pdf'),
    ]
    data = []
    labels = []
    for pdf_path in pdfs:
        if not os.path.exists(pdf_path):
            continue
        doc = fitz.open(pdf_path)
        spans = extract_spans(doc)
        # For demo: use font size as proxy for heading (user should label real data)
        # Label: 0 = not heading, 1 = H1, 2 = H2, 3 = H3
        font_sizes = [s["font_size"] for s in spans]
        size_counter = Counter(font_sizes)
        common = [size for size, _ in size_counter.most_common(4)]
        size_to_level = {size: i+1 for i, size in enumerate(sorted(common, reverse=True))}
        title = spans[0]["text"] if spans else ""
        for s in spans:
            features = extract_features(s, title)
            # Heuristic label: treat top 3 font sizes as H1/H2/H3, else not heading
            label = size_to_level.get(s["font_size"], 0)
            data.append(features)
            labels.append(label)
    X = np.array(data)
    y = np.array(labels)
    # Save sample data for reproducibility
    pd.DataFrame(np.column_stack([X, y]), columns=[f'f{i}' for i in range(X.shape[1])] + ['label']).to_csv(os.path.join(os.path.dirname(__file__), 'sample_heading_features.csv'), index=False)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Save model
    joblib.dump(clf, os.path.join(os.path.dirname(__file__), 'heading_classifier.pkl'))
    print('Model saved as heading_classifier.pkl')

if __name__ == '__main__':
    main() 