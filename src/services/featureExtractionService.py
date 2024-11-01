import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
from exception.exceptions import BadRequest, InternalServer

class FeatureExtractionService:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    async def extractFeatures(self, faceTensors : torch.Tensor) -> np.ndarray:
        if faceTensors is None or faceTensors.size(0) == 0:
            raise BadRequest(message="invalid face tensors for feature extraction")
        
        try:
            faceTensors = faceTensors.to(self.device)

            if len(faceTensors.shape) == 3:
                faceTensors = faceTensors.unsqueeze(0)

            with torch.no_grad():
                embeddings = self.model(faceTensors).cpu().numpy()
        except Exception as e:
            raise InternalServer(message="Failed to extract features") from e

        return embeddings[0]