#!/usr/bin/env python3
"""
Comprehensive test script for PDF Outline Extractor
Processes all 5 sample PDFs and stores results in JSON format
"""

import json
import time
from pathlib import Path
from outline_extractor.extractor import extract_outline
from outline_extractor.utils import validate_outline_schema, save_json

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

def test_single_pdf(pdf_path: Path, output_dir: Path) -> dict:
    """Test a single PDF and return results."""
    print(f"\n🔍 Testing: {pdf_path.name}")
    
    start_time = time.time()
    
    # Extract outline
    result = extract_outline(str(pdf_path))
    
    # Flatten hierarchy to match schema
    flat_outline = flatten_outline_tree(result.get("headings", []))
    
    # Prepare output in exact schema format
    output = {
        "title": result.get("title", ""),
        "outline": flat_outline
    }
    
    # Validate against schema
    schema_valid = validate_outline_schema(output)
    
    processing_time = time.time() - start_time
    
    # Save JSON output
    output_file = output_dir / f"{pdf_path.stem}_result.json"
    save_json(output, str(output_file))
    
    # Print results
    print(f"  📄 Title: {output['title']}")
    print(f"  📊 Headings found: {len(output['outline'])}")
    print(f"  ✅ Schema valid: {schema_valid}")
    print(f"  ⏱️  Processing time: {processing_time:.3f}s")
    print(f"  💾 Saved to: {output_file}")
    
    if output['outline']:
        print(f"  📝 Sample headings:")
        for i, heading in enumerate(output['outline'][:3]):
            print(f"    {i+1}. {heading['level']}: {heading['text']} (p.{heading['page']})")
    
    return {
        "file": pdf_path.name,
        "title": output['title'],
        "headings_count": len(output['outline']),
        "schema_valid": schema_valid,
        "processing_time": processing_time,
        "output_file": str(output_file)
    }

def main():
    """Test all 5 sample PDFs and generate comprehensive results."""
    print("🚀 PDF Outline Extractor - Comprehensive Test")
    print("=" * 60)
    
    # Setup paths
    input_dir = Path("Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs")
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("file0[1-5].pdf"))
    pdf_files.sort()  # Ensure consistent order
    
    if not pdf_files:
        print("❌ No PDF files found!")
        return
    
    print(f"📂 Found {len(pdf_files)} PDF files to test")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print("=" * 60)
    
    # Test each PDF
    results = []
    total_start_time = time.time()
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file.name}")
        result = test_single_pdf(pdf_file, output_dir)
        results.append(result)
    
    total_time = time.time() - total_start_time
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['schema_valid'])
    total_headings = sum(r['headings_count'] for r in results)
    avg_time = total_time / len(results)
    
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"📊 Total headings extracted: {total_headings}")
    print(f"⏱️  Total processing time: {total_time:.3f}s")
    print(f"⏱️  Average time per PDF: {avg_time:.3f}s")
    print(f"📁 Results saved to: {output_dir}")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for result in results:
        status = "✅" if result['schema_valid'] else "❌"
        print(f"  {status} {result['file']}: {result['headings_count']} headings, {result['processing_time']:.3f}s")
    
    # Save comprehensive results
    summary = {
        "test_summary": {
            "total_pdfs": len(results),
            "successful": successful,
            "total_headings": total_headings,
            "total_time": total_time,
            "average_time": avg_time
        },
        "individual_results": results
    }
    
    summary_file = output_dir / "test_summary.json"
    save_json(summary, str(summary_file))
    print(f"\n📄 Complete summary saved to: {summary_file}")
    
    # Performance check
    if total_time <= 10:
        print(f"✅ Performance: {total_time:.3f}s ≤ 10s (PASS)")
    else:
        print(f"❌ Performance: {total_time:.3f}s > 10s (FAIL)")
    
    if successful == len(results):
        print(f"✅ Success Rate: {successful}/{len(results)} (PASS)")
    else:
        print(f"❌ Success Rate: {successful}/{len(results)} (FAIL)")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    main() 