import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, List, Any


def extract_text_data(image: np.ndarray, regions: List[Dict[str, Any]], config: Dict) -> Dict[str, Any]:
    """
    Extract text data from the image, including headers, KPIs, and text blocks.
    
    Args:
        image: Preprocessed image
        regions: Layout regions identified in the image
        config: Configuration for text extraction
        
    Returns:
        Dictionary with headers, KPIs, and text blocks
    """
    text_blocks = []
    headers = []
    kpis = {}

    # Process each region based on its type
    for region in regions:
        region_type = region.get("type", "").lower()
        bbox = region.get("bbox", {})

        # Skip non-text regions
        if region_type not in ["text", "title"]:
            continue

        # Extract the region from the image
        x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get(
            "y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
        if x1 >= x2 or y1 >= y2:
            continue

        region_image = image[y1:y2, x1:x2]

        # Extract text using OCR
        text = pytesseract.image_to_string(region_image).strip()

        if not text:
            continue

        # Store the text block
        text_blocks.append({
            "text": text,
            "position": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "type": region_type
        })

        # Process headers
        if region_type == "title":
            headers.append(text)

        # Extract KPIs using regex patterns
        extract_kpis(text, kpis, config.get("kpi_patterns", []))

    return {
        "headers": headers,
        "kpis": kpis,
        "text_blocks": text_blocks
    }


def extract_kpis(text: str, kpis: Dict[str, str], patterns: List[str]) -> None:
    """
    Extract KPIs from text using regex patterns.
    Updates the kpis dictionary in place.
    """
    # If no patterns provided, use default patterns
    if not patterns:
        patterns = [
            r"([A-Za-z\s]+):\s*([^\n]+)",  # Key: Value
            r"([A-Za-z\s]+)\s*-\s*([^\n]+)",  # Key - Value
        ]

    # Apply each pattern
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            key, value = match
            key = key.strip()
            value = value.strip()

            # Look for numeric or currency values in the value
            if re.search(r'[$€£¥]|[0-9]', value):
                kpis[key] = value
