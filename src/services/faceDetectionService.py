from mtcnn.mtcnn import MTCNN
import numpy as np
from PIL import Image
from typing import Dict, List
import io

class FaceDetectionService:
    def __init__(self):
        self.detector = MTCNN(device="CPU:0")
        self.confidenceThreshold = 0.95
    
    def extractFace(self, pixels : np.ndarray, box : Dict) -> np.ndarray:
        x1, y1, width, height = box
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        return face

    async def detectFace(self, image : bytes) -> List[Dict[str, any]]:
        image = Image.open(io.BytesIO(image))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        pixels = np.asarray(image)

        faces = self.detector.detect_faces(pixels)

        confidentFaces = [
            face for face in faces
            if face['confidence'] >= self.confidenceThreshold
        ]

        if not confidentFaces:
            return None
        
        face = confidentFaces[0]

        facePixels = self.extractFace(pixels=pixels, box=face['box'])

        return facePixels, face['confidence']