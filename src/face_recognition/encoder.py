from imutils import paths
import numpy as np
import face_recognition
import pickle
import cv2
import os
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_face(image: np.ndarray) -> List:
    """Extrait les encodages faciaux d'une image unique."""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        if not boxes:
            logger.warning("No faces detected in the image")
            return []
        
        encodings = face_recognition.face_encodings(rgb, boxes)
        logger.info(f"Extracted {len(encodings)} face encodings")
        return encodings
    except Exception as e:
        logger.error(f"Error encoding face: {str(e)}")
        raise

def extract_encodings(image_dir: str, output_file: str = "face_enc.pkl") -> Dict[str, List]:
    """Extrait les encodages faciaux de toutes les images dans un répertoire."""
    known_encodings = []
    known_names = []
      # Get all user directories
    user_dirs = [d for d in os.listdir(image_dir) 
                if os.path.isdir(os.path.join(image_dir, d)) and d.startswith("user_")]
    
    if not user_dirs:
        logger.error(f"No user directories found in {image_dir}")
        raise ValueError(f"No user directories found in {image_dir}")

    for user_dir in user_dirs:
        user_path = os.path.join(image_dir, user_dir)
        image_paths = list(paths.list_images(user_path))
    
        if not image_paths:
            logger.error(f"No images found in {image_dir}")
            raise ValueError(f"No images found in {image_dir}")

        for i, image_path in enumerate(image_paths):
            try:
                user_id = image_path.split(os.path.sep)[-2]
                if not user_id.startswith("user_"):
                    logger.warning(f"Skipping invalid directory name: {user_id}")
                    continue

                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    continue

                encodings = encode_face(image)
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(user_id)

            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                continue

    if not known_encodings:
        logger.error("No valid encodings extracted")
        raise ValueError("No valid encodings extracted")

    data = {"encodings": known_encodings, "names": known_names}
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Encodings saved to {output_file}")

    return data

if __name__ == "__main__":
    try:
        extract_encodings("C:/Users/hh036/OneDrive/Bureau/PFE/python-app/Images", "face_enc.pkl")
    except Exception as e:
        logger.error(f"Failed to extract encodings: {str(e)}")