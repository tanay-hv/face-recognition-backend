import torch
import numpy as np
import tritonclient.http as triton_client
from exception.exceptions import BadRequest, InternalServer

class FeatureExtractionService:
    def __init__(self, 
                 triton_url: str = "localhost:8000", 
                 model_name: str = "facenet"):
        self.triton_client = triton_client.InferenceServerClient(url=triton_url)
        self.model_name = model_name

    async def extractFeatures(self, faceTensors: torch.Tensor) -> np.ndarray:
        if faceTensors is None or faceTensors.size(0) == 0:
            raise BadRequest(message="Invalid face tensors for feature extraction")
        
        try:
            if len(faceTensors.shape) == 3:
                faceTensors = faceTensors.unsqueeze(0)
            
            inputs = [triton_client.InferInput("input0", faceTensors.numpy().shape, "FP32")]
            inputs[0].set_data_from_numpy(faceTensors.numpy())
            outputs = [triton_client.InferRequestedOutput("output0")]
            response = self.triton_client.infer(self.model_name, inputs, outputs=outputs)
            embeddings = response.as_numpy("output0")

        except Exception as e: 
            raise InternalServer(message=f"Failed to extract features: {str(e)}") from e

        return embeddings[0]