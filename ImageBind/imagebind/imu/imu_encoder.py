import torch
import torch.nn as nn
from imagebind.models import imagebind_model
# from imagebind.models.imagebind_model import ModalityType
from types import SimpleNamespace

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)

class IMUEncoder(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # Create a minimal model with only IMU components
        self.model = imagebind_model.imagebind_huge(pretrained=False)
        
        # Remove all modality components except IMU
        for modality in list(self.model.modality_preprocessors.keys()):
            if modality != ModalityType.IMU:
                del self.model.modality_preprocessors[modality]
                del self.model.modality_trunks[modality]
                del self.model.modality_heads[modality]
                del self.model.modality_postprocessors[modality]
        
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def load_pretrained(self, pretrained_path):
        # Load the full pretrained model
        pretrained_model = imagebind_model.imagebind_huge(pretrained=True)
        
        # Copy IMU-related weights
        self.model.modality_preprocessors[ModalityType.IMU].load_state_dict(
            pretrained_model.modality_preprocessors[ModalityType.IMU].state_dict()
        )
        self.model.modality_trunks[ModalityType.IMU].load_state_dict(
            pretrained_model.modality_trunks[ModalityType.IMU].state_dict()
        )
        self.model.modality_heads[ModalityType.IMU].load_state_dict(
            pretrained_model.modality_heads[ModalityType.IMU].state_dict()
        )
        self.model.modality_postprocessors[ModalityType.IMU].load_state_dict(
            pretrained_model.modality_postprocessors[ModalityType.IMU].state_dict()
        )
    
    def forward(self, imu_data):
        # Ensure input is in the correct format
        if not isinstance(imu_data, dict):
            imu_data = {ModalityType.IMU: imu_data}
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(imu_data)
        
        return embeddings[ModalityType.IMU]

class IMUEncoderQuick(nn.Module):
    """A lightweight version of IMUEncoder that directly loads the pruned model."""
    def __init__(self, model_path=".checkpoints/imu_encoder.pth", device=None):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create minimal model structure
        self.model = imagebind_model.imagebind_huge(pretrained=False)
        
        # Remove all modality components except IMU
        for modality in list(self.model.modality_preprocessors.keys()):
            if modality != ModalityType.IMU:
                del self.model.modality_preprocessors[modality]
                del self.model.modality_trunks[modality]
                del self.model.modality_heads[modality]
                del self.model.modality_postprocessors[modality]
        
        # Load the pruned weights
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.to(self.device)
        self.eval()
    
    def forward(self, imu_data):
        # Ensure input is in the correct format
        if not isinstance(imu_data, dict):
            imu_data = {ModalityType.IMU: imu_data.to(self.device)}
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(imu_data)
        
        return embeddings[ModalityType.IMU]

def process_imu_data(imu_data):
    """Helper function to process IMU data into the correct format."""
    return torch.tensor(imu_data, dtype=torch.float32).transpose(0, 1).unsqueeze(0)


def prune_and_save_imu_encoder(pretrained_path, save_path):
    # Create and initialize the IMU encoder
    imu_encoder = IMUEncoder(pretrained_path)
    imu_encoder.eval()
    
    # Save the pruned model
    torch.save(imu_encoder.state_dict(), save_path)
    print(f"Pruned IMU encoder saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    pretrained_path = ".checkpoints/imagebind_huge.pth"
    save_path = ".checkpoints/imu_encoder.pth"
    prune_and_save_imu_encoder(pretrained_path, save_path) 