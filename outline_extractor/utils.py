import json
from typing import Any, Dict, List

def save_json(data: Any, out_path: str):
    """Save data as JSON file."""
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def validate_outline_schema(data: Any) -> bool:
    """
    Validate that the output matches the required schema format.
    
    Required schema:
    {
        "title": string,
        "outline": [
            {
                "level": string (H1, H2, H3, etc.),
                "text": string,
                "page": integer
            }
        ]
    }
    """
    try:
        # Check if data is a dictionary
        if not isinstance(data, dict):
            return False
        
        # Check required top-level keys
        if "title" not in data or "outline" not in data:
            return False
        
        # Validate title
        if not isinstance(data["title"], str):
            return False
        
        # Validate outline is a list
        if not isinstance(data["outline"], list):
            return False
        
        # Validate each outline item
        for item in data["outline"]:
            if not isinstance(item, dict):
                return False
            
            # Check required fields
            if "level" not in item or "text" not in item or "page" not in item:
                return False
            
            # Validate level is string and starts with H
            if not isinstance(item["level"], str) or not item["level"].startswith("H"):
                return False
            
            # Validate text is string
            if not isinstance(item["text"], str):
                return False
            
            # Validate page is integer
            if not isinstance(item["page"], int) or item["page"] < 1:
                return False
        
        return True
        
    except Exception:
        return False

def validate_heading_levels(outline: List[Dict[str, Any]]) -> bool:
    """Validate that heading levels are reasonable (H1-H6)."""
    for item in outline:
        level_str = item.get("level", "")
        if not level_str.startswith("H"):
            return False
        
        try:
            level_num = int(level_str[1:])
            if level_num < 1 or level_num > 6:
                return False
        except ValueError:
            return False
    
    return True

def validate_text_content(outline: List[Dict[str, Any]]) -> bool:
    """Validate that text content is reasonable."""
    for item in outline:
        text = item.get("text", "")
        
        # Text should not be empty
        if not text or not text.strip():
            return False
        
        # Text should not be too long (likely not a heading)
        if len(text) > 500:
            return False
        
        # Text should not be too short (likely not meaningful)
        if len(text.strip()) < 2:
            return False
    
    return True

def comprehensive_validation(data: Any) -> Dict[str, bool]:
    """Perform comprehensive validation and return detailed results."""
    results = {
        "schema_valid": validate_outline_schema(data),
        "heading_levels_valid": False,
        "text_content_valid": False
    }
    
    if results["schema_valid"]:
        outline = data.get("outline", [])
        results["heading_levels_valid"] = validate_heading_levels(outline)
        results["text_content_valid"] = validate_text_content(outline)
    
    return results

