import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
from astropy.io import fits
import os
from PIL import Image
import io
from lightcurvedataset import LightCurveDataset
from exoplanet import ExoplanetCNN
from exoplanetresnet import ExoplanetResNet
from nasa_main import load_json_to_dict
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch.utils.data import random_split
import glob
import re
from sklearn.metrics import roc_auc_score

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Example usage with dummy data
def create_dataset(data_folder):
    """Create a small dummy dataset for demonstration"""
    # In practice, you would have real FITS file paths and labels
    # dummy_files = ['dummy_file_1.fits', 'dummy_file_2.fits'] * 5
    # dummy_labels = [1, 0] * 5  # Alternating labels
    labels_dict = load_json_to_dict('merged_label_dict.json')
    fits_files = []
    labels = []
    """
    Counts ID folders under data_folder, and only plots .fits files from the first ID folder found.
    """
    id_folders = [item for item in os.listdir(data_folder)
                  if os.path.isdir(os.path.join(data_folder, item))]
    print(f"Found {len(id_folders)} ID folders in {data_folder}.")
    if not id_folders:
        print("No ID folders found.")
        return
    for id_folder in id_folders:
        id_path = os.path.join(data_folder, id_folder)
        for fname in os.listdir(id_path):
            if fname.lower().endswith('.fits'):
                fits_path = os.path.join(id_path, fname)
                fits_files.append(fits_path)
                labels.append(labels_dict.get(id_folder, 0))  # Get label for the ID folder
    return LightCurveDataset(fits_files, labels, transform=transform)

# 5. Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, log_dir="exoplanet", stop_loss=None):
    model.train()
    best_loss = float('inf')
    # Inside your training loop, before saving:
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    # Add date and time to log folder name
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = f"exoplanet_{now}"
    else:
        log_dir = f"{log_dir}_{now}"
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard writer

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_outputs = []

        for batch_idx, (filename, data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            # After: data, target = data.to(device), target.to(device)
            # img = data[0].cpu().numpy()  # Take the first image in the batch and move to CPU
            # # If channels are first (e.g., [3, 224, 224]), transpose to [224, 224, 3]
            # if img.shape[0] == 3:
            #     img = img.transpose(1, 2, 0)
            # # Undo normalization if needed
            # img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            # img = img.clip(0, 1)
            # plt.imshow(img)
            # plt.title("Sample input image")
            # plt.axis('off')
            # plt.tight_layout()
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # plt.show()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            # print("File Name : ", filename)
            # print("Output:", output.detach().cpu().numpy())
            # print("Target:", target.detach().cpu().numpy())
            # print("Loss:", loss.item())
            running_loss += loss.item()
            global_step = epoch * len(dataloader) + batch_idx
            # Log running loss to TensorBoard
            writer.add_scalar('Batch/RunningLoss', running_loss / (batch_idx + 1), global_step)
            # Calculate accuracy
            predicted = (output.squeeze() > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            # Log batch loss to TensorBoard
            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            if batch_idx % 4 == 0:
                # pass
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        final_epoch_acc = int(epoch_acc * 10000)  # Remove decimal points
        print(f'Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        # Calculate and log AUC
        try:
            auc = roc_auc_score(all_targets, all_outputs)
        except ValueError:
            auc = float('nan')  # Not enough classes to calculate AUC
        print(f'Epoch {epoch+1} AUC: {auc:.4f}')
        # Log epoch loss and accuracy to TensorBoard
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Epoch/AUC', auc, epoch)  # <-- Log AUC here
        # Save model with loss in filename if loss improves
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(models_dir, f"exoplanet_cnn_acc_{final_epoch_acc}_{now}.pth")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved as '{model_filename}'")
        # Stop training if loss is below threshold
        if stop_loss is not None and epoch_loss <= stop_loss:
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved as '{model_filename}'")
            print(f"Stopping early: loss {epoch_loss:.4f} reached threshold {stop_loss}")
            break
    writer.close()
    return model
def load_best_model(model_class, device, model_dir='.'):
    # Find all model files matching the pattern
    model_files = glob.glob(f"{model_dir}/exoplanet_cnn_acc_*.pth")
    best_acc = -1
    best_file = None
    pattern = re.compile(r"exoplanet_cnn_acc_(\d+).*\.pth")
    for file in model_files:
        filename = os.path.basename(file)
        match = pattern.search(filename)
        if match:
            acc = int(match.group(1))
            if acc > best_acc:
                best_acc = acc
                best_file = file
    if best_file is not None:
        print(f"Loading best model: {best_file} (acc={best_acc})")
        model = model_class().to(device)
        model.load_state_dict(torch.load(best_file, map_location=device))
        # model.eval()
        return model
    else:
        raise FileNotFoundError("No model files found matching pattern.")
# 6. Main execution
if __name__ == "__main__":
    # Create dataset (replace with your actual data)
    dataset = create_dataset('koi_data')
     # Split dataset into train and test sets (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test sets
    # Suppose labels is a list/array of all your labels (0 or 1)
    # Get the indices for the train split
    train_indices = train_dataset.indices
    labels = np.array([dataset.labels[i] for i in train_indices])  # 0 for non-exoplanet, 1 for exoplanet

    # Compute class weights (inverse frequency)
    class_sample_count = np.array([np.sum(labels == 0), np.sum(labels == 1)])
    weight = 1. / (class_sample_count + 1e-6)  # Prevent division by zero
    samples_weight = weight[labels]
    samples_weight = np.nan_to_num(samples_weight, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Initialize model, loss, and optimizer
    # model = ExoplanetCNN().to(device)
    # using ResNet18
    # model = ExoplanetResNet().to(device)
    
    
    # Train the model
    print("Starting training...")
    model = load_best_model(ExoplanetCNN, device, model_dir='./models')
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=10)
    # Load the saved model
    # trained_model = ExoplanetCNN().to(device)
    # trained_model.load_state_dict(torch.load('exoplanet_cnn_acc_8629.pth', map_location=device))
    # print("Model loaded from 'exoplanet_cnn_acc_8629.pth'")

    # Save the trained model with a datetime in the filename
    # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_filename = f"exoplanet_cnn_{now}.pth"
    # torch.save(trained_model.state_dict(), model_filename)
    # print(f"Model saved as '{model_filename}'")
    
    
    
    # Example prediction function
    def predict_exoplanet(test_loader, model):
        """
        Predict exoplanet probabilities for all samples in the test_loader.
        Prints prediction and probability for each sample.
        """
        model.eval()
        results = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(device)
                outputs = model(data)
                probabilities = outputs.squeeze().cpu().numpy()
                for i, prob in enumerate(np.atleast_1d(probabilities)):
                    prediction = "Exoplanet" if prob > 0.5 else "No exoplanet"
                    print(f"Sample {batch_idx * test_loader.batch_size + i}: Prediction: {prediction} (Probability: {prob:.4f})")
                    results.append(prob)
        return results

    # Example usage after training:
    # predict_exoplanet(test_loader, trained_model)