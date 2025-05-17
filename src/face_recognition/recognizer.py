import face_recognition
import cv2
import pickle
import os
import numpy as np
import sys
import logging

# Configurer le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logger.info(f"sys.path in recognizer: {sys.path}")

try:
    from src.face_recognition import encoder
    logger.info("Successfully imported encoder in recognizer")
except ImportError as e:
    logger.error(f"Failed to import encoder in recognizer: {str(e)}")
    raise

class FaceRecognizer:
    def __init__(self, encodings_file="face_enc.pkl"):
        self.encodings_file = encodings_file
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()

    def load_encodings(self):
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]

    def update_encodings(self):
        logger.info("Calling encoder.extract_encodings in update_encodings")
        encoder.extract_encodings("Images", self.encodings_file)
        self.load_encodings()

    def recognize(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return None
        
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                return self.known_names[first_match_index]
        return None