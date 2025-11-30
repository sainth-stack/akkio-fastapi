"""
Production-ready Image Classification API
Uses EfficientNet/MobileNet for fast, accurate image classification
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List, Optional, Dict
import os
import io
import json
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

image_classification_router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "image_classification"
TRAINING_DATA_DIR = PROJECT_ROOT / "image_training_data"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Image preprocessing for MobileNetV2
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    """Custom dataset for image classification"""
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def create_model(num_classes: int) -> nn.Module:
    """Create MobileNetV2 model for transfer learning"""
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze early layers
    for param in model.features[:-3].parameters():
        param.requires_grad = False
    
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    return model


def train_model(model: nn.Module, train_loader: DataLoader, epochs: int = 5, device: str = 'cpu') -> nn.Module:
    """Train the image classification model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    print("\n" + "="*70)
    print("🚀 TRAINING STARTED")
    print("="*70)
    print(f"Device: {device.upper()}")
    print(f"Epochs: {epochs}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Total training images: {len(train_loader.dataset)}")
    print("="*70 + "\n")
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n📊 Epoch {epoch+1}/{epochs}")
        print("-" * 70)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Show progress every few batches
            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0 or (batch_idx + 1) == len(train_loader):
                current_acc = 100 * correct / total
                current_loss = running_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] ({progress:.0f}%) - "
                      f"Loss: {current_loss:.4f}, Accuracy: {current_acc:.2f}%")
        
        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch+1} Complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    return model


@image_classification_router.post("/api/image/train")
async def train_image_model(
    model_name: str = Form(...),
    user_email: str = Form("admin@example.com"),
    epochs: int = Form(5),
    files: List[UploadFile] = File(...),
    labels: str = Form(...)  # JSON string of labels corresponding to files
):
    """
    Train a new image classification model
    
    Args:
        model_name: Name for the model
        user_email: User identifier
        epochs: Number of training epochs
        files: List of image files
        labels: JSON string array of labels (e.g., '["defect", "no_defect", "defect", ...]')
    """
    try:
        # Validate inputs
        if not model_name or not model_name.strip():
            raise HTTPException(400, "Model name is required")
        
        if len(files) < 2:
            raise HTTPException(400, "At least 2 images are required for training")
        
        # Parse labels
        try:
            label_list = json.loads(labels)
            if len(label_list) != len(files):
                raise HTTPException(400, "Number of labels must match number of files")
        except json.JSONDecodeError:
            raise HTTPException(400, "Labels must be a valid JSON array")
        
        # Create safe model name
        safe_model_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in model_name.strip())
        user_id = "".join(c if c.isalnum() else '_' for c in user_email)
        
        # Create model directory
        model_dir = MODELS_DIR / user_id / safe_model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training data directory
        training_dir = TRAINING_DATA_DIR / user_id / safe_model_name
        if training_dir.exists():
            shutil.rmtree(training_dir)
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique class names
        class_names = sorted(list(set(label_list)))
        num_classes = len(class_names)
        
        if num_classes < 2:
            raise HTTPException(400, "At least 2 different classes are required")
        
        print("\n" + "="*70)
        print(f"📦 MODEL TRAINING SETUP: {model_name}")
        print("="*70)
        print(f"Total images: {len(files)}")
        print(f"Classes detected: {num_classes}")
        for class_name in class_names:
            count = label_list.count(class_name)
            print(f"  - {class_name}: {count} images")
        print("="*70 + "\n")
        
        # Create class name to index mapping
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Save images and prepare dataset
        image_paths = []
        numeric_labels = []
        
        print("💾 Saving and validating images...")
        
        # Filter out system files and non-image files
        valid_files = []
        valid_labels = []
        skipped_files = []
        
        for file, label in zip(files, label_list):
            # Skip system files and non-image files
            filename = file.filename.lower()
            if any(filename.endswith(skip) for skip in ['.ds_store', 'thumbs.db', 'desktop.ini']):
                skipped_files.append(file.filename)
                continue
            if '.ds_store' in filename or 'thumbs.db' in filename:
                skipped_files.append(file.filename)
                continue
            
            valid_files.append(file)
            valid_labels.append(label)
        
        if skipped_files:
            print(f"⚠️  Skipped {len(skipped_files)} system files: {', '.join(skipped_files[:5])}")
            if len(skipped_files) > 5:
                print(f"   ... and {len(skipped_files) - 5} more")
        
        # Update the lists to use only valid files
        files = valid_files
        label_list = valid_labels
        
        if len(files) < 2:
            raise HTTPException(400, f"At least 2 valid image files are required (skipped {len(skipped_files)} system files)")
        
        print(f"✅ Processing {len(files)} valid image files\n")
        
        for i, (file, label) in enumerate(zip(files, label_list)):
            # Validate image
            try:
                content = await file.read()
                img = Image.open(io.BytesIO(content)).convert('RGB')
            except Exception as e:
                print(f"⚠️  Skipping invalid file '{file.filename}': {str(e)}")
                continue
            
            # Save image
            class_dir = training_dir / label
            class_dir.mkdir(exist_ok=True)
            
            ext = os.path.splitext(file.filename)[1] or '.jpg'
            image_path = class_dir / f"img_{i:04d}{ext}"
            img.save(image_path)
            
            image_paths.append(image_path)
            numeric_labels.append(class_to_idx[label])
            
            # Show progress for large datasets
            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  Processed {i + 1}/{len(files)} images...")
        
        print(f"✅ All {len(files)} images validated and saved\n")
        
        # Create dataset and dataloader
        print("🔧 Creating dataset and dataloader...")
        dataset = ImageDataset(image_paths, numeric_labels, transform=transform_aug)
        batch_size = min(16, len(dataset))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"  Batch size: {batch_size}")
        print(f"  Total batches: {len(train_loader)}\n")
        
        # Check for GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Training device: {device.upper()}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print()
        
        # Create and train model
        print("🏗️  Creating MobileNetV2 model...")
        model = create_model(num_classes)
        print(f"✅ Model created with {num_classes} output classes\n")
        
        model = train_model(model, train_loader, epochs=epochs, device=device)
        
        # Save model
        print("💾 Saving trained model...")
        model_path = model_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to: {model_path}\n")
        
        # Save metadata
        print("📝 Saving model metadata...")
        metadata = {
            "model_name": model_name,
            "safe_model_name": safe_model_name,
            "user_email": user_email,
            "num_classes": num_classes,
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "num_training_images": len(files),
            "epochs": epochs,
            "created_at": datetime.now().isoformat(),
            "model_architecture": "MobileNetV2"
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to: {metadata_path}\n")
        
        # Delete training images to save space (keep only the model)
        print("🧹 Cleaning up training data to save disk space...")
        try:
            shutil.rmtree(training_dir)
            print(f"✅ Training images deleted (model is saved)\n")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete training data: {e}\n")
            # Not a critical error, continue
        
        print("="*70)
        print(f"✨ MODEL '{model_name}' IS READY FOR USE!")
        print("="*70)
        print(f"Model name: {safe_model_name}")
        print(f"Classes: {', '.join(class_names)}")
        print(f"Total images trained: {len(files)}")
        print(f"Model path: {model_path}")
        print("="*70 + "\n")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Model '{model_name}' trained successfully",
            "model_name": safe_model_name,
            "num_classes": num_classes,
            "class_names": class_names,
            "num_images": len(files),
            "epochs": epochs,
            "model_path": str(model_path)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Training failed: {str(e)}")


@image_classification_router.get("/api/image/models")
async def list_trained_models(user_email: str = "admin@example.com"):
    """
    List all trained models for a user
    
    Returns:
        List of model metadata
    """
    try:
        user_id = "".join(c if c.isalnum() else '_' for c in user_email)
        user_models_dir = MODELS_DIR / user_id
        
        if not user_models_dir.exists():
            return JSONResponse(content={"models": []})
        
        models_list = []
        for model_dir in user_models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models_list.append(metadata)
        
        # Sort by creation date (newest first)
        models_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return JSONResponse(content={"models": models_list})
    
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        raise HTTPException(500, f"Failed to list models: {str(e)}")


@image_classification_router.post("/api/image/predict")
async def predict_image(
    model_name: str = Form(...),
    user_email: str = Form("admin@example.com"),
    file: UploadFile = File(...)
):
    """
    Predict image class using a trained model
    
    Args:
        model_name: Name of the trained model to use
        user_email: User identifier
        file: Image file to classify
    
    Returns:
        Prediction result with class and confidence
    """
    try:
        # Validate inputs
        if not model_name:
            raise HTTPException(400, "Model name is required")
        
        if not file:
            raise HTTPException(400, "Image file is required")
        
        # Load image
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content)).convert('RGB')
        except Exception as e:
            raise HTTPException(400, f"Invalid image file: {str(e)}")
        
        # Find model
        user_id = "".join(c if c.isalnum() else '_' for c in user_email)
        model_dir = MODELS_DIR / user_id / model_name
        
        if not model_dir.exists():
            raise HTTPException(404, f"Model '{model_name}' not found")
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise HTTPException(404, f"Model metadata not found for '{model_name}'")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        num_classes = metadata['num_classes']
        class_names = metadata['class_names']
        
        # Load model
        model_path = model_dir / "model.pth"
        if not model_path.exists():
            raise HTTPException(404, f"Model weights not found for '{model_name}'")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_model(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Preprocess image
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = {
            class_names[i]: float(probabilities[0][i].item())
            for i in range(num_classes)
        }
        
        return JSONResponse(content={
            "status": "success",
            "prediction": predicted_class,
            "confidence": confidence_score,
            "all_probabilities": all_probs,
            "model_name": model_name,
            "model_architecture": metadata.get('model_architecture', 'MobileNetV2')
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@image_classification_router.delete("/api/image/models/{model_name}")
async def delete_model(model_name: str, user_email: str = "admin@example.com"):
    """Delete a trained model"""
    try:
        user_id = "".join(c if c.isalnum() else '_' for c in user_email)
        model_dir = MODELS_DIR / user_id / model_name
        
        if not model_dir.exists():
            raise HTTPException(404, f"Model '{model_name}' not found")
        
        # Delete model directory
        shutil.rmtree(model_dir)
        
        # Also delete training data
        training_dir = TRAINING_DATA_DIR / user_id / model_name
        if training_dir.exists():
            shutil.rmtree(training_dir)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Model '{model_name}' deleted successfully"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Delete error: {str(e)}")
        raise HTTPException(500, f"Failed to delete model: {str(e)}")

