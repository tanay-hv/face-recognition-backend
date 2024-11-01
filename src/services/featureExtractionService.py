import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import io

class FeatureExtractionService:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    async def extractFeatures(self, faceTensors : torch.Tensor) -> np.ndarray:
        faceTensors = faceTensors.to(self.device)

        if len(faceTensors.shape) == 3:
            faceTensors = faceTensors.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.model(faceTensors).cpu().numpy()

        return embeddings[0]