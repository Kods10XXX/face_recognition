import logging
import os
import pickle
import sys
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
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

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            user_id = recognizer.recognize(frame)
            result = {"status": "unknown", "user_id": None, "name": None}

            if user_id:
                result["status"] = "success"
                result["user_id"] = user_id
                 # Extract numeric ID
                employee_id = int(user_id.replace('user_', ''))
                  # Call NestJS API with error handling
                response = requests.get(
                        f"{NESTJS_API_URL}/employees/{employee_id}",
                        timeout=3  # Add timeout
                    )
                if response.status_code == 200:
                    user_data = response.json()
                    result = {
                            "status": "success",
                            "user_id": user_data["id"],
                            "name": user_data["name"]  # Directly access name field
                        }
                    # Display on frame
                    cv2.putText(frame, f"ID: {employee_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Name: {user_data['name']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"User: {user_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_json(result)
            await websocket.send_bytes(buffer.tobytes())
    except Exception as e:
        logger.error(f"Error in video_feed: {str(e)}")
        await websocket.close()
    finally:
        cap.release()
        
        
