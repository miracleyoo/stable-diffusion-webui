import torch
import numpy as np
from imu_encoder import IMUEncoder
from imagebind.models.imagebind_model import ModalityType

def main():
    # Initialize the IMU encoder
    imu_encoder = IMUEncoder(pretrained_path=".checkpoints/imagebind_huge.pth")
    imu_encoder.eval()
    
    # Example IMU data (2000 timesteps, 6 channels)
    imu_data = np.random.randn(2000, 6)
    
    # Convert to tensor and reshape for the model
    # The model expects shape: [batch_size, channels, sequence_length]
    imu_tensor = torch.tensor(imu_data, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    
    # Get embeddings
    with torch.no_grad():
        embeddings = imu_encoder(imu_tensor)
    
    print(f"Input IMU data shape: {imu_tensor.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    
    # Save the pruned model for later use
    torch.save(imu_encoder.state_dict(), ".checkpoints/imu_encoder.pth")
    print("Saved pruned IMU encoder to .checkpoints/imu_encoder.pth")

if __name__ == "__main__":
    main() 