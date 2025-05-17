import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time
from sklearn.decomposition import PCA
import torch


class SVClassification:
    def __init__(self, train_dataloader, test_dataloader, class_names, log_visualize=False):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.class_names = class_names
        self.log_visualize = log_visualize
        self.pca = PCA(n_components=100)  # 次元削減用のPCA

    def extract_features(self, dataloader):
        features = []
        labels = []
        
        for batch in dataloader:
            if len(batch) == 3:  # multi_input の場合
                sample1, sample2, lbls = batch
                # 両方のサンプルを結合して特徴量とする
                inputs = torch.cat((sample1.view(sample1.size(0), -1), 
                                  sample2.view(sample2.size(0), -1)), dim=1)
            else:  # 通常の単一入力の場合
                inputs, lbls = batch
                inputs = inputs.view(inputs.size(0), -1)
            
            # float16に変換
            inputs = inputs.half()
            features.append(inputs.numpy().astype(np.float16))
            labels.append(lbls.numpy().astype(np.int16))
            
            # メモリ解放
            del inputs
            del lbls
        
        return np.concatenate(features), np.concatenate(labels)

    def train_and_evaluate(self):
        print("SVC Training and Evaluation")
        start_time = time.time()

        # トレーニングデータの特徴抽出
        print("Extracting training features...")
        train_features, train_labels = self.extract_features(self.train_dataloader)
        # memory release
        del self.train_dataloader

        # SVCの学習
        print("Training SVC...")
        svc = SVC(kernel='rbf', cache_size=1000)  # キャッシュサイズを制限
        svc.fit(train_features, train_labels)
        
        # メモリ解放
        del train_features
        del train_labels
        
        # テストデータの特徴抽出
        print("Extracting test features...")
        test_features, test_labels = self.extract_features(self.test_dataloader)
        
        # 予測と評価
        print("Making predictions...")
        predictions = svc.predict(test_features)
        acc = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='macro')
        recall = recall_score(test_labels, predictions, average='macro')
        f1 = f1_score(test_labels, predictions, average='macro')
        cm = confusion_matrix(test_labels, predictions)

        print(f'SVC Accuracy: {acc:.4f}')
        print(f'SVC Precision: {precision:.4f}')
        print(f'SVC Recall: {recall:.4f}')
        print(f'SVC F1 Score: {f1:.4f}')
        print(f'Time: {time.time() - start_time:.2f}s')
        
        if self.log_visualize:
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('SVC Confusion Matrix')
            plt.show()