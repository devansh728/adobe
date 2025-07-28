#!/usr/bin/env python3
"""Debug script for form field detection"""

from outline_extractor.extractor import extract_outline, is_form_field

def debug_form_detection():
    """Debug form field detection in file01.pdf"""
    print("🔍 Debugging form field detection...")
    
    result = extract_outline('Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/file01.pdf')
    spans = result.get('spans', [])
    
    print(f"📊 Total spans extracted: {len(spans)}")
    print("\n📝 Sample spans with form field detection:")
    
    for i, span in enumerate(spans[:15]):
        text = span['text']
        is_form = is_form_field(text)
        print(f"  {i+1}. \"{text}\"")
        print(f"     Font: {span['font_size']}, Bold: {span['is_bold']}, Form: {is_form}")
    
    print(f"\n🎯 Form fields detected: {sum(1 for s in spans if is_form_field(s['text']))}")

if __name__ == "__main__":
    debug_form_detection() 