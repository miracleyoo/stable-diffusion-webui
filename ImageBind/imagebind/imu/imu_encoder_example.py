import torch
import numpy as np
from imu_encoder import IMUEncoder, IMUEncoderQuick, process_imu_data
from imagebind.models.imagebind_model import ModalityType


def main():
    # First, create and save the pruned model if it doesn't exist
    try:
        # Try to load the quick encoder to check if pruned model exists
        imu_encoder_quick = IMUEncoderQuick()
        print("Loaded pruned IMU encoder.")
    except:
        print("Creating pruned IMU encoder...")
        # Create and save the pruned model
        imu_encoder = IMUEncoder(pretrained_path=".checkpoints/imagebind_huge.pth")
        imu_encoder.eval()
        torch.save(imu_encoder.state_dict(), ".checkpoints/imu_encoder.pth")
        print("Saved pruned IMU encoder to .checkpoints/imu_encoder.pth")
    
    # Example IMU data (2000 timesteps, 6 channels)
    imu_data = np.random.randn(2000, 6)
    imu_tensor = process_imu_data(imu_data)
    
    # Method 1: Using IMUEncoderQuick (recommended for inference)
    print("\nUsing IMUEncoderQuick:")
    imu_encoder_quick = IMUEncoderQuick()
    with torch.no_grad():
        embeddings_quick = imu_encoder_quick(imu_tensor)
    print(f"Input IMU data shape: {imu_tensor.shape}")
    print(f"Output embeddings shape: {embeddings_quick.shape}")
    
    # Method 2: Using original IMUEncoder (for training or when you need to load from full model)
    print("\nUsing IMUEncoder:")
    imu_encoder = IMUEncoder(pretrained_path=".checkpoints/imagebind_huge.pth")
    imu_encoder.eval()
    with torch.no_grad():
        embeddings = imu_encoder(imu_tensor)
    print(f"Input IMU data shape: {imu_tensor.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    
    # Verify both methods give same results
    print("\nVerifying results match:")
    print(f"Max difference between embeddings: {torch.max(torch.abs(embeddings - embeddings_quick))}")

if __name__ == "__main__":
    main() 