from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import json
from pathlib import Path
from outline_extractor.extractor import extract_outline
from outline_extractor.utils import save_json, validate_outline_schema

app = FastAPI(title="PDF Outline Extractor API", 
              description="Extract structured outlines from PDF documents",
              version="1.0.0")

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

@app.post('/extract-outline/')
async def extract_outline_api(file: UploadFile = File(...)):
    """
    Extract structured outline from uploaded PDF file.
    
    Returns JSON with:
    - title: Document title (string)
    - outline: Array of heading objects with level (H1/H2/H3), text, and page
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    try:
        # Extract outline using the enhanced extractor
        outline_data = extract_outline(file_path)
        
        # Flatten hierarchy to match schema
        flat_outline = flatten_outline_tree(outline_data.get("headings", []))
        
        # Prepare response in exact schema format
        response = {
            "title": outline_data.get("title", ""),
            "outline": flat_outline
        }
        
        # Validate against schema
        if not validate_outline_schema(response):
            raise HTTPException(status_code=500, detail="Generated output does not match required schema")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    finally:
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass

@app.get('/health')
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "PDF Outline Extractor"}

@app.get('/')
async def root():
    """API information."""
    return {
        "service": "PDF Outline Extractor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract-outline/": "Extract outline from PDF",
            "GET /health": "Health check",
            "GET /": "API information"
        }
    }

