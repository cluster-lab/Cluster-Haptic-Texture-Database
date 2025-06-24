
"""
This module implements a multi-modal classification system that combines audio and accelerometer data.
It uses a dual-stream architecture where each modality (audio and accelerometer data) is processed
by a separate feature extractor, and then the features are fused for final classification.

The architecture supports different backbone models (ResNet, ViT, RegNet) for feature extraction
from each modality, and provides flexibility in how the features are combined.

Key components:
- FeatureExtractor: A wrapper for various backbone models to extract features
- AudioAccelMultiClassification: The main class that handles the dual-stream architecture,
  training, evaluation, and visualization of results

This approach allows for effective texture classification by leveraging complementary information
from both acoustic and vibration signals.
"""


import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTFeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
import numpy as np
import time
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import timm  # ファイルの先頭に追加
import os
import torch.nn as nn
from torchvision.models.resnet import ResNet34_Weights

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class FeatureExtractor(nn.Module):
    def __init__(self, base_model, feature_size, in_channels=1, model_type='resnet'):
        super(FeatureExtractor, self).__init__()
        self.model_type = model_type
        
        if model_type == 'vit':
            self.model = base_model
            # ViTの出力次元を変更
            self.model.head = nn.Linear(self.model.head.in_features, feature_size)
        else:  # resnet or regnet
            # 入力チャンネル数に合わせて最初の畳み込み層を変更
            original_conv = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            num_features = base_model.fc.in_features
            base_model.fc = nn.Linear(num_features, feature_size)
            self.model = base_model

    def forward(self, x):
        x = self.model(x)
        x = F.relu(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # MultiheadAttention expects (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        attended, _ = self.attention(x, x, x)
        # Add & Norm
        x = self.norm1(attended + x)
        # Feed forward
        ff_output = self.ff(x)
        # Add & Norm
        x = self.norm2(ff_output + x)
        # Return to original shape
        return x.transpose(0, 1)

class AudioAccelModel(nn.Module):
    def __init__(self, audio_model, accel_model, feature_size, num_classes, num_transformer_layers=1, model_type='resnet'):
        super(AudioAccelModel, self).__init__()
        # 入力チャンネル数を指定して特徴抽出器を初期化
        self.audio_encoder = FeatureExtractor(audio_model, feature_size, in_channels=1, model_type=model_type)
        self.accel_encoder = FeatureExtractor(accel_model, feature_size, in_channels=3, model_type=model_type)

        
        # 特徴量を結合した後の次元
        combined_dim = feature_size * 2
        
        # Transformer層の追加
        transformer_layers = []
        for _ in range(num_transformer_layers):
            transformer_layers.append(TransformerBlock(
                embed_dim=combined_dim, 
                num_heads=8, 
                ff_dim=combined_dim*4, 
                dropout=0.1
            ))
        self.transformer = nn.Sequential(*transformer_layers)
        
        # 分類層
        self.fc = nn.Linear(combined_dim, num_classes)

    def forward(self, audio, accel):
        # encoder
        audio_features = self.audio_encoder(audio)
        accel_features = self.accel_encoder(accel)
        
        # concat
        combined = torch.cat((audio_features, accel_features), dim=1)
        
        # Transformerに入力するために形状を変更 (batch_size, sequence_length=1, features)
        combined = combined.unsqueeze(1)
        
        # Transformer処理
        transformed = self.transformer(combined)
        
        # 分類のために形状を変更
        transformed = transformed.squeeze(1)
        
        # 分類
        output = self.fc(transformed)
        
        return output

class AudioAccelMultiClassification:
    def __init__(self, dataloaders, dataset_sizes, class_names, feature_size=256, experiment_name='audio_accel_texture_classification', num_epochs=25, learning_rate=0.00005, model="resnet"):
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.class_names = class_names
        self.feature_size = feature_size
        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # writer
        self.writer = None
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load pre-trained models
        if model == "resnet":
            model_audio = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            model_accel = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            model_type = 'resnet'
        elif model == "regnet":
            model_audio = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1)
            model_accel = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1)
            model_type = 'regnet'
        elif model == "vit":
            model_audio = timm.create_model('vit_small_patch32_224', pretrained=True, in_chans=1)
            model_accel = timm.create_model('vit_small_patch32_224', pretrained=True, in_chans=3)
            model_type = 'vit'
        else:
            raise ValueError("Invalid model name. Choose from: 'resnet', 'regnet', 'vit'")

        # Create the multimodal classifier model
        self.model = AudioAccelModel(
            audio_model=model_audio,
            accel_model=model_accel,
            feature_size=feature_size,
            num_classes=len(class_names),
            model_type=model_type  # モデルタイプを渡す
        )
        self.model = self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def train_model(self, log_visualization=True):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        for epoch in range(self.num_epochs):
            if log_visualization:
                print(f'Epoch {epoch}/{self.num_epochs - 1}')
                print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for audio_data, accel_data, labels in self.dataloaders[phase]:
                    audio_data = audio_data.to(self.device)
                    accel_data = accel_data.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward
                    # Track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(audio_data, accel_data)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Statistics
                    running_loss += loss.item() * audio_data.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                if log_visualization:
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    self.scheduler.step(epoch_loss)

                # Log history
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())
                    if self.writer:
                        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
                        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())
                    if self.writer:
                        self.writer.add_scalar('Loss/val', epoch_loss, epoch)
                        self.writer.add_scalar('Accuracy/val', epoch_acc, epoch)

                # Deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            if log_visualization:
                print()

        time_elapsed = time.time() - since
        
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

    def save_model(self, path=None):
        if path is None:
            path = '.log/' + self.experiment_name + '/best_model.pth'
        torch.save(self.model.state_dict(), path)

    def evaluate_model(self, log_visualization=True):
        self.model.eval()
        running_corrects = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for audio_data, accel_data, labels in self.dataloaders['test']:
                audio_data = audio_data.to(self.device)
                accel_data = accel_data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(audio_data, accel_data)
                _, preds = torch.max(outputs, 1)

                all_preds.append(preds)
                all_labels.append(labels)

                running_corrects += torch.sum(preds == labels.data)

        test_acc = running_corrects.double() / self.dataset_sizes['test']

        # Calculate Precision, Recall, F1 Score
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')


        if log_visualization:
            print(f'Test Acc: {test_acc:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Plot confusion matrix
        if self.writer:
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix')
            self.writer.add_figure('Confusion Matrix', fig, self.num_epochs)
            plt.close(fig)

        # tensor -> float
        test_acc = test_acc.item()

        return test_acc

    def open_writer(self):
        self.writer = SummaryWriter(log_dir='.log/' + self.experiment_name)

    def close_writer(self):
        if self.writer:
            self.writer.close()