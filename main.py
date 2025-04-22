import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import zipfile
import shutil
import os
import json
import uuid
from pathlib import Path

app = FastAPI()


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

TEMP_DIR = config["extraction"]["temp_dir"]
MAX_SIZE_MB = config["extraction"]["max_file_size_mb"]
UPLOAD_DIR = "uploads"
EXTRACT_DIR = "extracted"

Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path(EXTRACT_DIR).mkdir(exist_ok=True)


@app.post("/upload-pbix/")
async def upload_pbix(file: UploadFile = File(...)):
    if not file.filename.endswith(".pbix"):
        return JSONResponse(content={"error": "File must be a .pbix"}, status_code=400)

    # Save uploaded file
    uid = str(uuid.uuid4())
    saved_path = os.path.join(UPLOAD_DIR, f"{uid}.pbix")
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Rename to zip and extract
    zip_path = saved_path.replace(".pbix", ".zip")
    os.rename(saved_path, zip_path)

    extract_path = os.path.join(EXTRACT_DIR, uid)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Parse layout.json
    layout_json_path = os.path.join(extract_path, "Report", "Layout")
    if not os.path.exists(layout_json_path):
        return JSONResponse(content={"error": "Layout file not found"}, status_code=500)

    with open(layout_json_path, "r", encoding="utf-8") as f:
        layout_data = json.load(f)

    visuals_info = []

    try:
        for page in layout_data.get("sections", []):
            page_name = page.get("displayName", "Unnamed Page")
            for visual in page.get("visualContainers", []):
                vis_info = {
                    "page": page_name,
                    "visual_type": visual.get("config", {}).get("singleVisual", {}).get("visualType", "unknown"),
                    "title": visual.get("config", {}).get("singleVisual", {}).get("title", {}).get("text", "N/A")
                }
                visuals_info.append(vis_info)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to parse layout: {str(e)}"}, status_code=500)

    return {"visuals": visuals_info}
