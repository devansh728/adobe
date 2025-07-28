#!/usr/bin/env python3
"""
Challenge 1B API: Persona-Driven Document Intelligence
Provides semantic understanding and ranking of document sections based on persona and task.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Import our Challenge 1B processor
from challenge_1b_processor import PersonaDrivenProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the processor
processor = PersonaDrivenProcessor()

app = FastAPI(
    title="Challenge 1B: Persona-Driven Document Intelligence",
    description="Semantic understanding and ranking of document sections based on persona and task",
    version="1.0.0"
)

@app.post('/process-collection/')
async def process_collection(
    input_data: Dict[str, Any]
):
    """
    Process a document collection with persona-driven intelligence.
    
    Input format:
    {
        "challenge_info": {"challenge_id": "round_1b_XXX", "test_case_name": "xxx"},
        "documents": [{"filename": "doc.pdf", "title": "Title"}],
        "persona": {"role": "User Persona"},
        "job_to_be_done": {"task": "Task description"}
    }
    
    Returns ranked sections with importance scores and subsection analysis.
    """
    try:
        start_time = time.time()
        
        # Process the collection
        result = processor.process_collection(input_data)
        
        # Add API metadata
        result["api_metadata"] = {
            "endpoint": "/process-collection/",
            "processing_time": round(time.time() - start_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing collection: {str(e)}")

@app.post('/rank-sections/')
async def rank_sections(
    sections: List[Dict[str, Any]],
    persona: str = Form(...),
    task: str = Form(...)
):
    """
    Rank sections by relevance to persona and task.
    
    Input:
    - sections: List of section objects with title, content, etc.
    - persona: User persona (e.g., "Travel Planner")
    - task: Specific task description
    
    Returns ranked sections with importance scores.
    """
    try:
        # Rank the sections
        ranked_sections = processor.rank_sections(sections, persona, task)
        
        return JSONResponse(content={
            "ranked_sections": ranked_sections,
            "metadata": {
                "persona": persona,
                "task": task,
                "total_sections": len(ranked_sections),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            }
        })
        
    except Exception as e:
        logger.error(f"Error ranking sections: {e}")
        raise HTTPException(status_code=500, detail=f"Error ranking sections: {str(e)}")

@app.post('/extract-sections/')
async def extract_sections(
    file: UploadFile = File(...)
):
    """
    Extract sections from a single PDF file.
    
    Returns extracted sections with metadata.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract sections
        sections = processor.extract_sections_from_pdf(str(file_path))
        
        return JSONResponse(content={
            "sections": sections,
            "metadata": {
                "filename": file.filename,
                "total_sections": len(sections),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            }
        })
        
    except Exception as e:
        logger.error(f"Error extracting sections: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting sections: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            if file_path.exists():
                file_path.unlink()
        except:
            pass

@app.get('/health')
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Challenge 1B: Persona-Driven Document Intelligence",
        "version": "1.0.0",
        "processor_ready": processor is not None
    }

@app.get('/')
async def root():
    """API information."""
    return {
        "service": "Challenge 1B: Persona-Driven Document Intelligence API",
        "version": "1.0.0",
        "description": "Semantic understanding and ranking of document sections based on persona and task",
        "endpoints": {
            "POST /process-collection/": "Process document collection with persona-driven intelligence",
            "POST /rank-sections/": "Rank sections by relevance to persona and task",
            "POST /extract-sections/": "Extract sections from single PDF",
            "GET /health": "Health check",
            "GET /": "API information"
        },
        "features": {
            "semantic_similarity": "Sentence transformer-based semantic understanding",
            "keyword_boosting": "Task-specific keyword relevance boosting",
            "persona_patterns": "Persona-specific relevance patterns",
            "structural_importance": "Hierarchical heading importance",
            "subsection_analysis": "Detailed content analysis for top sections"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001) 