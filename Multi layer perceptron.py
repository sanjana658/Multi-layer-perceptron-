import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
# URL containing the raw Pima Indians Diabetes CSV file
data_url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
df = pd.read_csv(data_url)

# Separate features (X) and target label (y)
X = df.drop(columns=['Outcome']).values
y = df['Outcome'].values

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features because deep learning models perform better with normalized data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 2. PYTORCH DATASET & DATALOADER
# ==========================================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        # Convert arrays to PyTorch Tensors (float32 for features, float32 for binary loss)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Reshape to (N, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TabularDataset(X_train, y_train)
test_dataset = TabularDataset(X_test, y_test)

# Mini-batch loading
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==========================================
# 3. BUILD THE MULTILAYER PERCEPTRON (MLP)
# ==========================================
class DiabetesMLP(nn.Module):
    def __init__(self, input_dim):
        super(DiabetesMLP, self).__init__()
        # 8 Input features -> Hidden Layer 1 -> Hidden Layer 2 -> 1 Output Neuron
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),                  # Non-linearity
            nn.Linear(16, 8),
            nn.ReLU(),                  # Non-linearity
            nn.Linear(8, 1),
            nn.Sigmoid()                # Forces output to be a probability between 0 and 1
        )
        
    def forward(self, x):
        return self.network(x)

# Model configuration
model = DiabetesMLP(input_dim=X_train.shape[1])
criterion = nn.BCELoss() # Binary Cross Entropy Loss for 0/1 tasks
optimizer = optim.Adam(model.parameters(), lr=0.01) # Modern adaptive optimizer

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
epochs = 50
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()          # Reset existing gradients
        predictions = model(batch_X)    # Forward pass
        loss = criterion(predictions, batch_y) # Compute error
        loss.backward()                 # Backward pass (calculate gradients)
        optimizer.step()                # Update network weights
        
        epoch_loss += loss.item() * batch_X.size(0)
    
    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        total_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_epoch_loss:.4f}")

# ==========================================
# 5. EVALUATE ALL METRICS
# ==========================================
model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculations for speed and memory saving
    # Convert entire test set to tensors
    test_X_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Get raw probability outputs (0.0 to 1.0)
    probabilities = model(test_X_tensor).numpy()
    # Convert probabilities to hard binary choices (0 or 1) using a 0.5 threshold
    predictions = (probabilities >= 0.5).astype(int)

# Calculate standard classification evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

# Print Final Evaluation Dashboard
print("\n" + "="*35)
print("     FINAL EVALUATION METRICS     ")
print("="*35)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("="*35)
