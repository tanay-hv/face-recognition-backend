import torch
from facenet_pytorch import MTCNN
from PIL import Image
from typing import Tuple, Optional
import io
import asyncio
from exception.exceptions import FaceNotDetected, LowSimilarityScore, InternalServer

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
        self.maxImageSize = 1024

    def optimiseImageSync(self, imageBytes : bytes) -> Image.Image :
        image = Image.open(io.BytesIO(imageBytes))

        if max(image.size) > self.maxImageSize:
            ratio = self.maxImageSize / max(image.size)
            newSize = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(newSize, Image.LANCZOS)

        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image

    async def optimiseImage(self, image : bytes) -> Image.Image :
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(None, self.optimiseImageSync, image)
        return image
        
    async def detectFace(self, imageBytes : bytes) -> Optional[Tuple[torch.Tensor, float]]:

        image = await self.optimiseImage(image=imageBytes)

        try :
            faceTensor, prob = self.detector(image, return_prob=True)

        except Exception as e :
            raise InternalServer(f"Something went wrong, try again later")
        
        if faceTensor is None:
            raise FaceNotDetected
        
        if prob < self.threshold:
            raise LowSimilarityScore
        
        return faceTensor, prob