import torch
import torch.nn.functional as F
from encoder import GCN, MLP
from torch.optim import Adam
from dgl import add_self_loop, to_bidirected
from utils import get_logger, fix_seed, augment, node_classification
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, \
    WikiCSDataset, AmazonCoBuyComputerDataset, CoauthorCSDataset
from ogb.nodeproppred import DglNodePropPredDataset


def normalize(Z):
    """batch-normalization"""
    return (Z - Z.mean(0)) / Z.std(0)


def uniformity(Z):
    """Wasserstein Distance between Z and N(0, I)"""
    # Z has been batch-normalized
    n, d = Z.shape

    C = Z.T @ Z / (n-1)
    L, Q = torch.linalg.eigh(C)
    uni = -2 * torch.clamp(L, min=1e-8).sqrt().sum()
    
    # L_ = torch.diag((L+1e-4).sqrt())
    # C_ = Q @ L_ @ Q.T
    # uni = - 2 * torch.trace(C_) 
    return uni


if __name__ == '__main__':
    # Load dataset
    dataset = 'Cora'
    if dataset == 'Cora':
        G = CoraGraphDataset()[0]
    elif dataset == 'CiteSeer':
        G = CiteseerGraphDataset()[0]
    elif dataset == 'PubMed':
        G = PubmedGraphDataset()[0]
    elif dataset == 'WikiCS':
        G = WikiCSDataset()[0]
    elif dataset == 'Computer':
        G = AmazonCoBuyComputerDataset()[0]
    elif dataset == 'CoauthorCS':
        G = CoauthorCSDataset()[0]
    elif dataset == 'ArXiv':
        data = DglNodePropPredDataset(name='ogbn-arxiv')
        split_idx = data.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        G, Y = data[0]
        Y = Y.squeeze()
        G = to_bidirected(G, copy_ndata=True)
    else:
        G = CoraGraphDataset()[0]

    X = G.ndata.pop('feat')
    if dataset != 'ArXiv':
        Y = G.ndata.pop('label')

    # Set hyper-parameters
    device = 'cuda:0'
    in_dim = X.shape[1]
    # Copy from params.txt
    lam = 0.1
    epochs = 80
    pd, pm = 0.3, 0.1
    lr, wd = 1e-3, 1e-5
    hid_dims = [256, 256]
    lr2, wd2 = 1e-2, 1e-4


    logger = get_logger(f'./{dataset}.log')
    params = {
        'lam': lam,
        'epochs': epochs,
        'pd, pm': (pd, pm),
        'lr, wd': (lr, wd),
        'hid_dims': hid_dims,
        'lr2, wd2': (lr2, wd2)
    }
    logger.info(str(params))

    fix_seed(0)  # for reproduction

    if dataset != 'CoauthorCS':
        encoder = GCN(in_dim, hid_dims, act_fn=F.elu)
    else:
        encoder = MLP(in_dim, hid_dims, act_fn=F.elu)
    encoder = encoder.to(device)
    
    # Pre-train
    optimizer = Adam(encoder.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(epochs):
        encoder.train()
        G1, X1 = augment(G, X, pd, pm)
        G2, X2 = augment(G, X, pd, pm)
        G1 = add_self_loop(G1).to(device)
        G2 = add_self_loop(G2).to(device)
        X1, X2 = X1.to(device), X2.to(device)
        Z1, Z2 = encoder(G1, X1), encoder(G2, X2)
        Z1 = normalize(Z1)
        Z2 = normalize(Z2)
        
        loss_inv = - (Z1 * Z2).sum() / Z1.shape[0]
    
        loss_uni = 0.5 * (uniformity(Z1) + uniformity(Z2)) 
        
        loss = loss_inv + lam * loss_uni

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        print(f'E:{epoch} Loss:{loss.item():.4f} '
              f'INV:{loss_inv.item():.4f} UNI:{loss_uni.item():.4f}')
    
    # Evaluation
    encoder.eval()
    G = add_self_loop(G).to(device)
    X = X.to(device)
    Y = Y.to(device)
    with torch.no_grad():
        Z = encoder(G, X)
    if dataset == 'ArXiv':
        masks = (train_idx, valid_idx, test_idx)
        node_classification(Z, Y, dataset, logger=logger, lr=lr2, wd=wd2)
    elif dataset in ['Cora', 'Citeseer', 'Pubmed', 'WikiCS']:
        train_mask = G.ndata.pop('train_mask').to(torch.bool)
        val_mask = G.ndata.pop('val_mask').to(torch.bool)
        test_mask = G.ndata.pop('test_mask').to(torch.bool)
        masks = (train_mask, val_mask, test_mask)
        node_classification(Z, Y, dataset, masks=masks, logger=logger, lr=lr2, wd=wd2)
    else:
        node_classification(Z, Y, dataset=dataset, masks=None, logger=logger, lr=lr2, wd=wd2)
    logger.info('')
