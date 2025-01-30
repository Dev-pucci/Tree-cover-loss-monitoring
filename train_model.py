import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader

# Define the SimpleCNN class
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Assuming input size is 224x224
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a dataset and data loader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = FakeData(transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model and define loss and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model (this is a dummy example with fake data)
for epoch in range(5):  # Train for 5 epochs
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Set the model to evaluation mode
model.eval()

# Create a dummy input for exporting the model
dummy_input = torch.randn(1, 3, 224, 224)

# Check the model and dummy input
print("Model ready for export.")
print("Dummy input shape:", dummy_input.shape)

# Export the model to ONNX format
onnx_file_path = "model.onnx"
try:
    # This will print out what is happening during the export
    print("Starting model export...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file_path, 
        export_params=True, 
        opset_version=11,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Model exported to", onnx_file_path)
except Exception as e:
    print("An error occurred during model export:", e)

# Check if the file was created
if os.path.exists(onnx_file_path):
    print("ONNX file successfully created.")
else:
    print("ONNX file was not created.")