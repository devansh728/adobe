import os
import json
import time
import gc
from pathlib import Path
from outline_extractor.extractor import extract_outline
from outline_extractor.utils import validate_outline_schema

def flatten_outline_tree(tree, flat=None):
    """Flatten hierarchical outline to match schema format."""
    if flat is None:
        flat = []
    
    for node in tree:
        # Convert level to string format (H1, H2, H3, etc.)
        flat.append({
            "level": f"H{node['level']}",
            "text": node["text"],
            "page": node["page"]
        })
        
        # Recursively process children
        if "children" in node:
            flatten_outline_tree(node["children"], flat)
    
    return flat

def process_single_pdf(pdf_file: Path, output_dir: Path) -> bool:
    """Process a single PDF with enhanced error handling."""
    try:
        print(f"Processing {pdf_file.name}...")
        start_time = time.time()
        
        # Extract outline
        outline_data = extract_outline(str(pdf_file))
        
        # Flatten hierarchy to match schema
        flat_outline = flatten_outline_tree(outline_data.get("headings", []))
        
        # Prepare output in exact schema format
        output = {
            "title": outline_data.get("title", ""),
            "outline": flat_outline
        }
        
        # Validate against schema
        if not validate_outline_schema(output):
            print(f"Warning: {pdf_file.name} output does not match schema")
        
        # Save output
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        processing_time = time.time() - start_time
        print(f"✓ Processed {pdf_file.name} -> {output_file.name} ({processing_time:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {pdf_file.name}: {e}")
        return False

def process_pdfs():
    """Process all PDFs in sample dataset directory with enhanced complexity handling."""
    input_dir = Path("Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs")
    output_dir = Path("Adobe-India-Hackathon25/Challenge_1a/sample_dataset/outputs")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("file0[1-5].pdf"))
    
    if not pdf_files:
        print("No PDF files found in sample dataset directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    print("=" * 50)
    
    # Process each PDF individually to manage memory
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file.name}")
        
        # Process the PDF
        if process_single_pdf(pdf_file, output_dir):
            successful += 1
        else:
            failed += 1
        
        # Force garbage collection after each PDF to manage memory
        gc.collect()
        
        # Check if we're approaching time limit (for 50-page PDFs)
        if i == 1:  # After first PDF, estimate total time
            elapsed = time.time()
            estimated_total = elapsed * len(pdf_files)
            if estimated_total > 8:  # Leave 2s buffer
                print(f"Warning: Estimated total time {estimated_total:.1f}s exceeds 10s limit")
    
    print("\n" + "=" * 50)
    print(f"Processing complete!")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"📁 Output files saved to: {output_dir}")

if __name__ == "__main__":
    print("🚀 Starting PDF Outline Extractor")
    print("📂 Input directory: Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs")
    print("📂 Output directory: Adobe-India-Hackathon25/Challenge_1a/sample_dataset/outputs")
    print("=" * 50)
    
    start_time = time.time()
    process_pdfs()
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total processing time: {total_time:.2f}s")
    print("✅ PDF processing completed!") 