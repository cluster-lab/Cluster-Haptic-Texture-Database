import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import timm
from transformers import ViTForImageClassification, ViTFeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
import numpy as np
import time
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class VitClassification:
    def __init__(self, dataloaders, dataset_sizes, class_names, experiment_name='vit_classification', num_epochs=25, learning_rate=0.00005):
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.class_names = class_names
        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # writer
        self.writer = None
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # prepare model
        # データセットのサンプルから入力チャネル数と入力サイズを取得
        sample_data, _ = dataloaders['train'].dataset[0]
        input_channels = sample_data.shape[0]  # チャネル数 (通常は1か3)
        input_height = sample_data.shape[1]  # 高さ
        input_width = sample_data.shape[2]  # 幅
        print(f"Input Channels: {input_channels}, Height: {input_height}, Width: {input_width}")

        # モデルの準備
        # PyTorch ViT (なんかうまくいかん)
        # self.model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        # self.model.conv_proj = torch.nn.Conv2d(input_channels, 1024, kernel_size=(32, 32), stride=(32, 32))
        # num_ftrs = self.model.heads.head.in_features
        # self.model.heads.head = nn.Linear(num_ftrs, len(self.class_names))
        
        # timm ViTを使用
        self.model = timm.create_model('vit_small_patch32_224', pretrained=True, num_classes=len(self.class_names), in_chans=input_channels)
       
        # print(self.model)
        self.model = self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)

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

            # 各エポックはトレーニングとバリデーションのフェーズからなる
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # モデルをトレーニングモードにセット
                else:
                    self.model.eval()   # モデルを評価モードにセット

                running_loss = 0.0
                running_corrects = 0

                # データのイテレート
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # パラメータの勾配をゼロにセット
                    self.optimizer.zero_grad()

                    # 順伝播
                    # トレーニングフェーズでは勾配を計算する
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # トレーニングフェーズでのバックプロパゲーションと最適化
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # 統計を収集
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                if log_visualization:
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    self.scheduler.step(epoch_loss)

                # ログを記録
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

                # モデルをディープコピーする
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            if log_visualization:
                print()

        time_elapsed = time.time() - since
        
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # ベストモデルの重みをロード
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
            for inputs, labels in self.dataloaders['test']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
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

        # 混同行列の計算
        cm = confusion_matrix(all_labels, all_preds)

        # 混同行列をプロット
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