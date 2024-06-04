import os
import dgl
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def get_logger(filename, verbosity=1, name=None, mode='a'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# ---------- Graph Augmentation ----------
def augment(G: dgl.DGLGraph, X: torch.Tensor,
            edge_drop_rate: float, feat_mask_rate: float) -> (dgl.DGLGraph, torch.Tensor):
    """Edge Dropping + Feature Masking"""
    G = drop_edge(G, edge_drop_rate)
    X = mask_feat(X, feat_mask_rate)

    return G, X


def drop_edge(G: dgl.DGLGraph, drop_prob: float) -> dgl.DGLGraph:
    """Edge Dropping"""
    n_nodes = G.num_nodes()
    n_edges = G.num_edges()
    mask_rates = torch.full((n_edges,), fill_value=drop_prob,
                            dtype=torch.float)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1).to(G.device)

    src, dst = G.edges()
    src = src[mask_idx]
    dst = dst[mask_idx]

    G = dgl.graph((src, dst), num_nodes=n_nodes)

    return G


def mask_feat(X: torch.Tensor, mask_prob: float) -> torch.Tensor:
    """Feature Masking"""
    drop_mask = (
            torch.empty((X.size(1),), dtype=torch.float32, device=X.device).uniform_()
            < mask_prob
    )
    X = X.clone()
    X[:, drop_mask] = 0

    return X


# ---------- Evaluation Tools ----------
def split4NC(n_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    """Split node set for Node Classification."""
    assert train_ratio + test_ratio < 1
    train_size = int(n_samples * train_ratio)
    test_size = int(n_samples * test_ratio)
    indices = torch.randperm(n_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = torch.nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, X):
        Z = self.fc(X)
        return Z


class LREvaluator4NC:
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        for key in ['train', 'test', 'valid']:
            assert key in split
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = torch.nn.LogSoftmax(dim=-1)
        criterion = torch.nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    Y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, Y_pred, average='micro')
                    test_macro = f1_score(y_test, Y_pred, average='macro')

                    y_val = y[split['valid']].detach().cpu().numpy()
                    Y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, Y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro

                    pbar.set_postfix({'best test MiF1': best_test_micro, 'MaF1': best_test_macro})
                    pbar.update(self.test_interval)

        return {
            'MiF1': best_test_micro,
            'MaF1': best_test_macro
        }


def node_classification(Z: torch.FloatTensor, Y: torch.LongTensor,
                        dataset: str, masks=None, n_repeats: int = 10, logger=None, lr=0.01, wd=0.):
    """Evaluate node representations on node classification."""
    logger = print if logger is None else logger.info

    fix_seed(0)
    n_nodes = Z.shape[0]
    MiF1s = []
    MaF1s = []
    if dataset == 'WikiCS':
        train_masks = masks[0]
        val_masks = masks[1]
        test_mask = masks[2]
        indices = torch.arange(n_nodes, device=Z.device)
        for i in range(20):
            split = {
                'train': indices[train_masks[:, i]],
                'valid': indices[val_masks[:, i]],
                'test': indices[test_mask]
            }
            res = LREvaluator4NC(num_epochs=3000, learning_rate=lr, weight_decay=wd).evaluate(Z, Y, split)
            MiF1s.append(res['MiF1'])
            MaF1s.append(res['MaF1'])
    else:
        if masks is not None:
            train_mask = masks[0]
            val_mask = masks[1]
            test_mask = masks[2]
            indices = torch.arange(n_nodes, device=Z.device)
            for i in range(n_repeats):
                split = {
                    'train': indices[train_mask],
                    'valid': indices[val_mask],
                    'test': indices[test_mask]
                }
                res = LREvaluator4NC(num_epochs=3000, learning_rate=lr, weight_decay=wd).evaluate(Z, Y, split)
                MiF1s.append(res['MiF1'])
                MaF1s.append(res['MaF1'])
        else:
            for i in range(n_repeats):
                split = split4NC(n_nodes, train_ratio=0.1, test_ratio=0.8)
                res = LREvaluator4NC(num_epochs=3000, learning_rate=lr, weight_decay=wd).evaluate(Z, Y, split)
                MiF1s.append(res['MiF1'])
                MaF1s.append(res['MaF1'])
    MiF1s = np.array(MiF1s)
    MaF1s = np.array(MaF1s)
    micro_mean = MiF1s.mean() * 100
    micro_std = MiF1s.std() * 100
    macro_mean = MaF1s.mean() * 100
    macro_std = MaF1s.std() * 100
    s = f"MiF1={micro_mean:.2f}+-{micro_std:.2f}, MaF1={macro_mean:.2f}+-{macro_std:.2f}"
    logger(s)
