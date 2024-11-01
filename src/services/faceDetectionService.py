import torch
from facenet_pytorch import MTCNN
from PIL import Image
from typing import Tuple, Optional
import io

class FaceDetectionService:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        self.threshold = 0.95

    async def detectFace(self, image : bytes) -> Optional[Tuple[torch.Tensor, float]]:
        image = Image.open(io.BytesIO(image))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        faceTensor, prob = self.detector(image, return_prob=True)
        
        if faceTensor is None or prob < self.threshold:
            return None
        
        return faceTensor, prob