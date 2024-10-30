import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2 # type: ignore
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from PIL import Image

class FeatureExtractionService:
    def __init__(self):
        self.model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
    
    def preprocessFace(self, facePixels: np.ndarray) -> np.ndarray:
        if facePixels.shape[2] == 4:
            facePixels = facePixels[:,:,:3]

        image = Image.fromarray(facePixels)
        image = image.resize((250, 250))

        facePixels = img_to_array(image)
        facePixels = preprocess_input(facePixels)

        return facePixels
    
    async def extractFeatures(self, facePixels : np.ndarray) -> np.ndarray :
        processedFace = self.preprocessFace(facePixels=facePixels)
        expanded = np.expand_dims(processedFace, axis=0)
        features = self.model.predict(expanded, verbose = 0)
        return features[0]