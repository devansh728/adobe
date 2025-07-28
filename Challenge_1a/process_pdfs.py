import os
import json
from pathlib import Path
from outline_extractor.extractor import extract_outline

def flatten_outline_tree(tree, flat=None):
    if flat is None:
        flat = []
    for node in tree:
        # Convert level to string (e.g., H1, H2, H3)
        flat.append({
            "level": f"H{node['level']}",
            "text": node["text"],
            "page": node["page"]
        })
        if "children" in node:
            flatten_outline_tree(node["children"], flat)
    return flat

def process_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        outline_data = extract_outline(str(pdf_file))
        # Flatten hierarchy to match schema
        flat_outline = flatten_outline_tree(outline_data.get("headings", []))
        output = {
            "title": outline_data.get("title", ""),
            "outline": flat_outline
        }
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Processed {pdf_file.name} -> {output_file.name}")

if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("completed processing pdfs") 