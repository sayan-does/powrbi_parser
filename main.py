from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
import yaml
import os
from typing import Dict, Any, List, Optional

# Import parsers
from structure_parser import analyze_layout
from text_parser import extract_text_data
from visual_parser import extract_visual_data

app = FastAPI(
    title="Dashboard Data Extraction API",
    description="Extract structured data from dashboard images and PDFs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


@app.post("/extract")
async def extract_dashboard_data(file: UploadFile = File(...)):
    """
    Extract all data from a dashboard image or PDF.
    Returns structured JSON with all detected elements.
    """
    # Read file content
    content = await file.read()

    # Process the input file
    try:
        # Convert file to image
        if file.filename.lower().endswith('.pdf'):
            # For PDF handling, we'd typically use pdf2image or similar library
            # For this example, we'll assume it's handled by a helper function
            image = convert_pdf_to_image(content)
        else:
            # Process as image
            image = cv2.imdecode(np.frombuffer(
                content, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400, detail="Failed to process input file")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing file: {str(e)}")

    # Extract all data
    result = extract_all_data(image)

    return result


def convert_pdf_to_image(pdf_content: bytes) -> np.ndarray:
    """
    Convert PDF content to an image.
    Note: In a real implementation, you would use pdf2image or a similar library.
    For this example, we're returning a placeholder.
    """
    # This is a placeholder. In a real implementation, you would use:
    # import pdf2image
    # images = pdf2image.convert_from_bytes(pdf_content)
    # return np.array(images[0])

    # For this placeholder, we'll just return a blank image
    return np.ones((800, 600, 3), dtype=np.uint8) * 255


def extract_all_data(image: np.ndarray) -> Dict[str, Any]:
    """
    Process a dashboard image and extract all structured data.
    This is the main pipeline function that orchestrates the extraction process.
    """
    # Step 1: Preprocess the image (e.g., resize, denoise)
    processed_image = preprocess_image(image)

    # Step 2: Analyze layout to identify different regions
    regions = analyze_layout(processed_image, config["layout"])

    # Step 3: Extract text data (includes headers, KPIs, text blocks)
    text_data = extract_text_data(processed_image, regions, config["text"])

    # Step 4: Extract visual data (tables, charts)
    visual_data = extract_visual_data(
        processed_image, regions, config["visual"])

    # Step 5: Combine all data into structured format
    result = {
        "headers": text_data.get("headers", []),
        "KPIs": text_data.get("kpis", {}),
        "text_blocks": text_data.get("text_blocks", []),
        "tables": visual_data.get("tables", []),
        "charts": visual_data.get("charts", [])
    }

    return result


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image to improve extraction accuracy.
    """
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Basic noise removal
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    return denoised


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
