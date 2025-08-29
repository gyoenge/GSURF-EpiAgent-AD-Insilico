import json
import torch
from torch.utils.data import Dataset
import numpy as np


class CellDatasetForUFEWithLabel(Dataset):
    """
    UFE(CCA/SR)용 입력 + Contrastive용 Status 라벨을 함께 반환
    - adata.X: 신호 (SR)
    - obs['cell_sentences']: cCRE ID 리스트
    - obs['Status']: 레이블 (문자열 → 정수 맵)
    """
    def __init__(self, adata, cell_sentences, labels,
                 max_length=8192, alpha_for_CCA=1, num_cCRE=1355445,
                 is_random=False):
        self.adata = adata.copy()
        self.max_length = max_length
        self.alpha_for_CCA = alpha_for_CCA
        self.num_cCRE = num_cCRE
        self.is_random = is_random

        if not isinstance(cell_sentences, list):
            cell_sentences = list(cell_sentences)
        self.cell_sentences = [json.loads(x) if isinstance(x, str) else x for x in cell_sentences]
        self.labels = list(labels)

    def __len__(self):
        return len(self.cell_sentences)

    def __getitem__(self, idx):
        cell = np.array(self.cell_sentences[idx], dtype=int)
        signals = self.adata[idx].X.toarray().reshape(-1).astype(np.float32)

        inaccessible = np.ones(self.num_cCRE, dtype=bool)
        adjusted = cell - 4          
        adjusted = adjusted[(adjusted >= 0) & (adjusted < self.num_cCRE)]
        inaccessible[adjusted] = False
        neg_pool = np.where(inaccessible)[0] + 4

        k_neg = int(len(cell) * self.alpha_for_CCA)
        if k_neg <= len(neg_pool):
            neg = np.random.choice(neg_pool, size=k_neg, replace=False)
        else:
            neg = np.random.choice(neg_pool, size=k_neg, replace=True)

        ex_ids = np.concatenate([cell, neg])
        ex_acc = np.concatenate([np.ones_like(cell, dtype=np.float32),
                                 np.zeros_like(neg, dtype=np.float32)])

        if len(cell) > self.max_length - 2:
            if self.is_random:
                sel = np.sort(np.random.choice(len(cell), self.max_length - 2, replace=False))
                cell = cell[sel]
            else:
                cell = cell[:self.max_length - 2]
        cell = [1] + cell.tolist() + [2]  # [CLS]=1, [SEP]=2

        return (
            torch.tensor(cell, dtype=torch.long),
            torch.tensor(signals, dtype=torch.float32),
            torch.tensor(ex_ids, dtype=torch.long),
            torch.tensor(ex_acc, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

def collate_fn_ufe_with_label(batch):
    cells, signals, ex_ids, ex_acc, labels = zip(*batch)
    max_len = max(x.size(0) for x in cells)
    padded_cells = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=0) for x in cells
    ])
    signals_b = torch.stack(signals)
    ex_ids_b = list(ex_ids)  
    ex_acc_b = torch.cat(ex_acc)
    labels_b = torch.stack(labels)
    return padded_cells, signals_b, ex_ids_b, ex_acc_b, labels_b
