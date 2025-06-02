import logging
import os
import pickle
import sys
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np

import io
import json
import requests


# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
logger = logging.getLogger(__name__)
logger.info(f"sys.path: {sys.path}")

try:
    from src.face_recognition import encoder
    logger.info("Successfully imported encoder from src.face_recognition")
except ImportError as e:
    logger.error(f"Failed to import encoder from src.face_recognition: {str(e)}")
    try:
        import encoder
        logger.info("Successfully imported encoder from root")
    except ImportError as e:
        logger.error(f"Failed to import encoder from root: {str(e)}")
        raise
try:
    from src.face_recognition.recognizer import FaceRecognizer
    logger.info("Successfully imported recognizer in main")
except ImportError as e:
    logger.error(f"Failed to import recognizer in recognizer: {str(e)}")
    raise
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
recognizer = FaceRecognizer(encodings_file="face_enc.pkl")
NESTJS_API_URL = "http://localhost:3000"  

def load_encodings(encodings_file: str = "face_enc.pkl") -> dict:
    data = {"encodings": [], "names": []}
    if os.path.exists(encodings_file):
        try:
            with open(encodings_file, "rb") as f:
                while True:
                    try:
                        loaded_data = pickle.load(f)
                        data["encodings"].extend(loaded_data["encodings"])
                        data["names"].extend(loaded_data["names"])
                    except EOFError:
                        break
        except Exception as e:
            logger.error(f"Error loading encodings: {str(e)}")
    return data

def save_encoding(user_id: str, encoding: list, encodings_file: str = "face_enc.pkl"):
    data = load_encodings(encodings_file)
    data["encodings"].append(encoding)
    data["names"].append(f"user_{user_id}")
    try:
        with open(encodings_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Encoding saved for user_{user_id}")
    except Exception as e:
        logger.error(f"Error saving encoding: {str(e)}")
        raise

@app.post("/extract_face")
async def extract_face(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image file")

        face_encodings = encoder.encode_face(image)
        if not face_encodings:
            raise ValueError("No face found in the image")

        output_dir = f"Images/user_{user_id}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/face.jpg"
        cv2.imwrite(output_path, image)  # Sauvegarder l'image complète
        logger.info(f"Image saved to {output_path}")

        save_encoding(user_id, face_encodings[0])
        recognizer.update_encodings()  # Mettre à jour les encodages dans FaceRecognizer

        return {"status": "success", "message": f"Face extracted and saved for user_{user_id}"}
    except Exception as e:
        logger.error(f"Error in extract_face: {str(e)}")
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(e)})

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image file")

        user_id = recognizer.recognize(image)
        if user_id:
            response = requests.get(f"{NESTJS_API_URL}/employees/{user_id.replace('user_', '')}")
            if response.status_code == 200:
                user_data = response.json()
                return {"status": "success", "user_id": user_id, "name": user_data.get["name"]}
            return {"status": "success", "user_id": user_id}
        raise ValueError("No face recognized")
    except Exception as e:
        logger.error(f"Error in recognize_face: {str(e)}")
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(e)})

@app.post("/recognize_image")
async def recognize_image(file: UploadFile = File(...)):
    # Log received file details for debugging
    logger.info(f"Received file: {file.filename}")
    logger.info(f"Content-Type: {file.content_type}")
    logger.info(f"File size: {file.size}")
    
    # Accept multiple image formats - be more permissive
    allowed_types = [
        "image/jpeg", "image/jpg", "image/png", 
        "image/heic", "image/webp", "image/bmp",
        "application/octet-stream"  # Sometimes mobile apps send this
    ]
    
    # Also check file extension if content-type is not reliable
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp']
    file_extension = None
    if file.filename:
        file_extension = os.path.splitext(file.filename.lower())[1]
    
    # Check both content type and file extension
    content_type_valid = file.content_type in allowed_types
    extension_valid = file_extension in allowed_extensions if file_extension else False
    
    if not content_type_valid and not extension_valid:
        error_msg = f"Unsupported file. Content-Type: {file.content_type}, Extension: {file_extension}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        # Read image bytes
        contents = await file.read()
        
        # Add validation for empty file
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        logger.info(f"File contents size: {len(contents)} bytes")
        
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Could not decode image - invalid image data")
            raise HTTPException(status_code=400, detail="Invalid image data - could not decode")
        
        logger.info(f"Image decoded successfully. Shape: {frame.shape}")
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Recognize the user (assuming your recognizer logic is the same)
        user_id = recognizer.recognize(frame) # Make sure recognizer is defined
        
        result = {"status": "unknown", "user_id": None, "name": None}
        
        if user_id:
            result["status"] = "success"
            result["user_id"] = user_id
            employee_id = int(user_id.replace('user_', ''))
            
            try:
                response = requests.get(f"{NESTJS_API_URL}/employees/{employee_id}", timeout=3)
                if response.status_code == 200:
                    user_data = response.json()
                    result["user_id"] = user_data["id"]
                    result["name"] = user_data["name"]
            except requests.RequestException as e:
                logger.error(f"Error contacting NestJS API: {e}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in recognize_image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")