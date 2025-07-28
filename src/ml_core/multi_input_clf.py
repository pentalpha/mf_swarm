import os
import pickle
import json
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
from tqdm import tqdm
from sklearn import metrics

from ml_core.custom_statistics import faster_fmax

def get_batch_of_indexes(n_samples, batch_size):
    n_batchs = int(n_samples / batch_size)
    if n_batchs < (n_samples / batch_size):
        n_batchs += 1
    
    indexes = []
    for batch_i in range(n_batchs):
        start = batch_size * batch_i
        end = start + batch_size
        if end > n_samples:
            end = n_samples
        indexes.append(list(range(start,end)))
    
    return indexes

class ProteinFuncDataset(Dataset):
    def __init__(self, data_x, data_y):
        # data_x: list de (nome, np.ndarray (N, D_i))
        self.feature_arrays = [torch.from_numpy(arr).float() for name, arr in data_x]
        self.y = torch.from_numpy(data_y).float()
    def __len__(self):
        return self.y.size(0)
    def __getitem__(self, idx):
        xs = [feat[idx] for feat in self.feature_arrays]
        return xs, self.y[idx]
    
# Sub-rede para cada feature
class FeatureSubNet(nn.Module):
    def __init__(self, in_dim, l1_dim, l2_dim, leaky_alpha, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, l1_dim)
        self.bn1 = nn.BatchNorm1d(l1_dim)
        self.act1 = nn.LeakyReLU(negative_slope=leaky_alpha)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(l1_dim, l2_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

# Rede principal
class MultiInputNet(nn.Module):
    def __init__(self, feature_names, params_dict, n_labels):
        super().__init__()
        input_dims = [params_dict['input_dims'][f] for f in feature_names]
        # 1) Cria as sub-redes na mesma ordem de `feature_names`
        self.subnets = nn.ModuleList()
        for name, in_dim in zip(feature_names, input_dims):
            p = params_dict[name]
            self.subnets.append(FeatureSubNet(
                in_dim=in_dim,
                l1_dim=p["l1_dim"],
                l2_dim=p["l2_dim"],
                leaky_alpha=p["leakyrelu_1_alpha"],
                dropout=p["dropout_rate"]
            ))
        # 2) Calcula concat_dim somando só os l2_dim das features
        concat_dim = sum(params_dict[name]["l2_dim"] for name in feature_names)
        final_p = params_dict["final"]
        self.bn_comb = nn.BatchNorm1d(concat_dim)
        self.fc_comb = nn.Linear(concat_dim, final_p["final_dim"])
        self.act_comb = nn.ReLU()
        self.drop_comb = nn.Dropout(final_p["dropout_rate"])
        self.out = nn.Linear(final_p["final_dim"], n_labels)
        self.sigmoid = nn.Sigmoid()
        self.n_epochs = 0
        self.n_labels = n_labels
        self.feature_names = feature_names
        self.params_dict = params_dict
        self.history = []

    def save(self, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        metaparams = {
            'params_dict': self.params_dict,
            'n_epochs': self.n_epochs,
            'feature_names': self.feature_names,
            'n_labels': self.n_labels,
            'history': self.history
        }
        json.dump(metaparams, open(f"{output_dir}/metaparams.json", 'w'), indent=5)
        torch.save(self.state_dict(), f"{output_dir}/best_state.pth")
    
    def load(output_dir):
        metaparams = json.load(open(f"{output_dir}/metaparams.json", 'r'))
        model = MultiInputNet(metaparams['feature_names'],
            metaparams['params_dict'],
            metaparams['n_labels'])
        model.history = metaparams['history']
        model.load_state_dict(torch.load(f"{output_dir}/best_state.pth"))
        model.eval()
        model.n_epochs = metaparams['n_epochs']
        return model

    def forward(self, xs):
        # xs: lista de tensores, na ordem feature_names
        outs = [net(x) for net, x in zip(self.subnets, xs)]
        x = torch.cat(outs, dim=1)
        x = self.bn_comb(x)
        x = self.fc_comb(x)
        x = self.act_comb(x)
        x = self.drop_comb(x)
        x = self.out(x)
        return self.sigmoid(x)

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader:   torch.utils.data.DataLoader,
            n_epochs:    int,
            lr:          float,
            patience:    int,
            lr_decay_every: int,
            device:      torch.device):
        """
        Treina o modelo usando Early Stopping e LR Scheduler.
        - train_loader, val_loader: DataLoaders com (xs, y)
        - n_epochs: número máximo de épocas
        - lr: learning rate inicial
        - patience: épocas sem melhora para parar
        - lr_decay_every: reduz LR à metade a cada x épocas
        - device: 'cpu' ou 'cuda'
        """
        self.to(device)
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 0.5**(e // lr_decay_every))

        best_metric_val = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, n_epochs+1):
            #print('fit')
            # ---- Treino ----
            self.train()
            train_loss = 0.0
            for xs, y in train_loader:
                xs = [x.to(device) for x in xs]
                y  = y.to(device)
                optimizer.zero_grad()
                pred = self(xs)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * y.size(0)
            train_loss /= len(train_loader.dataset)

            #print('test pred')
            # ---- Validação ----
            self.eval()
            val_loss = 0.0
            all_preds, all_trues = [], []
            with torch.no_grad():
                for xs, y in val_loader:
                    xs = [x.to(device) for x in xs]
                    y  = y.to(device)
                    pred = self(xs)
                    val_loss += criterion(pred, y).item() * y.size(0)
                    all_preds.append(pred.cpu().numpy())
                    all_trues.append(y.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            scheduler.step()

            #print('val2')
            # Concatena para métricas
            all_preds = np.vstack(all_preds)
            all_trues = np.vstack(all_trues)
            auprc = metrics.average_precision_score(all_trues, all_preds, average='weighted')
            auprc_rounded = round(auprc, 2)
            print(f"Epoch {epoch}/{n_epochs} — "
                  f"train_loss: {train_loss:.6f}, "
                  f"val_loss: {val_loss:.6f}, "
                  f"AUPRC: {auprc:.4f}")
            self.history.append({
                'Epoch': f'{epoch}/{n_epochs}',
                "train_loss": f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}',
                'AUPRC': f'{auprc:.4f}'
            })
            #print('Save')
            # Early stopping
            if val_loss < best_metric_val:
                best_metric_val = val_loss
                print('Loss improvement!')
                epochs_no_improve = 0
                #torch.save(self.state_dict(), "best_model.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at {epoch} epochs, after {epochs_no_improve} without improvements.")
                    break
        self.n_epochs = epoch
        # Carrega melhor modelo
        #self.load_state_dict(torch.load("best_model.pth", map_location=device))
        return best_metric_val

    def predict_and_test(self,
                data_loader: torch.utils.data.DataLoader,
                device:      torch.device):
        """
        Cria matrizes de predição e de verdade a partir de um DataLoader
        Retorna (all_preds, all_trues) como np.ndarrays.
        """
        self.eval()
        self.to(device)
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xs, y in data_loader:
                xs = [x.to(device) for x in xs]
                '''for x in xs:
                    print(x)
                    print(x.shape)'''
                y  = y.to(device)
                pred = self(xs)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(y.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_trues = np.vstack(all_trues)
        return all_preds, all_trues
    
    def predict(self, feature_vecs: List[np.ndarray], 
            device: torch.device = None, verbose=True, batch_size=1000,):
        """
        Execução em batch do modelo.
        """
        self.eval()
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        all_preds = []
        n_samples = feature_vecs[0].shape[0]
        batch_indexes = get_batch_of_indexes(n_samples, batch_size)
        pred_iterator = tqdm(batch_indexes) if verbose else batch_indexes
        for indexes in pred_iterator:
            with torch.no_grad():
                xs = [torch.from_numpy(f[indexes]).float().to(device) 
                    for f in feature_vecs]
                '''for x in xs:
                    print(x)
                    print(x.shape)'''
                pred = self(xs)
                all_preds.append(pred.cpu().numpy())
        all_preds = np.vstack(all_preds)
        return all_preds
    
def makeMultiClassifierModel(train_x, train_y, test_x, test_y, params_dict):
    print('Preparando')
    feature_names = [n for n, x in train_x]
    # Prepara DataLoader
    train_ds = ProteinFuncDataset(train_x, train_y)
    test_ds  = ProteinFuncDataset(test_x, test_y)
    bs = params_dict["final"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs)
    output_dim = train_y.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Definindo modelo')
    model = MultiInputNet(feature_names, params_dict, output_dim)
    print('Fit')
    # Treinar
    best_val_loss = model.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        n_epochs=params_dict["final"]["epochs"],
        #n_epochs=4,
        lr=params_dict["final"]["learning_rate"],
        patience=params_dict["final"]["patience"],
        lr_decay_every=10,
        device=device
    )
    n_epochs = model.n_epochs
    epochs_norm = n_epochs / 100
    epochs_norm2 = 1.0 - epochs_norm
    y_pred = model.predict([f for name, f in test_x], device=device, verbose=False)

    roc_auc_score_mac = metrics.roc_auc_score(test_y, y_pred, average='macro')
    roc_auc_score_w = metrics.roc_auc_score(test_y, y_pred, average='weighted')
    auprc_mac = metrics.average_precision_score(test_y, y_pred)
    auprc_w = metrics.average_precision_score(test_y, y_pred, average='weighted')

    fmax, bestrh = faster_fmax(y_pred, test_y)
    test_stats = {
        'ROC AUC': float(roc_auc_score_mac),
        'ROC AUC W': float(roc_auc_score_w),
        'AUPRC': float(auprc_mac),
        'AUPRC W': float(auprc_w),
        'Fmax': float(fmax),
        'Best Fmax Threshold': float(bestrh),
        'Proteins': len(train_y) + len(test_y),
        'quickness': epochs_norm2
    }

    metric_weights = [('Fmax', 4), ('ROC AUC W', 4), ('AUPRC W', 4), ('quickness', 2)]
    w_total = sum([w for m, w in metric_weights])
    test_stats['fitness'] = sum([test_stats[m]*w for m, w in metric_weights]) / w_total

    return model, test_stats