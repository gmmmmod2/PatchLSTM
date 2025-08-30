"""Works output keys:
- 'poi_type' : List[List[str]]
- 'neighbor_poi_type' : List[List[List[str]]]
- 'hour' : List[int]
- 'stay_duration' : List[float] (seconds)
- 'incident' : List[float]
- 'is_anomaly' : bool
- 'h3_index' : List[str]
"""
from __future__ import annotations
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

PAD_IDX = 0
HOUR_PAD = -1  # avoid collision with real hour=0

class TrajectoryDataset(Dataset):
    def __init__(self, pkl_path: str, max_len: int = 120, max_seq: int = 16,
                 max_neighbors: int = 7, scale_stay_to_hours: bool = True,
                 log1p_stay: bool = False, label_as_float: bool = False,
                 vocab_from_train_ids: List[str] | None = None):
        with open(pkl_path, 'rb') as f:
            self.traj_dict = pickle.load(f)
        self.traj_ids = list(self.traj_dict.keys())
        self.max_len = int(max_len)
        self.max_seq = int(max_seq)
        self.max_neighbors = int(max_neighbors)
        self.scale_stay_to_hours = bool(scale_stay_to_hours)
        self.log1p_stay = bool(log1p_stay)
        self.label_as_float = bool(label_as_float)

        # Optional: build vocab on training subset only to avoid leakage
        if vocab_from_train_ids is None:
            vocab_source_ids = self.traj_ids
        else:
            vocab_source_ids = vocab_from_train_ids
        self.poi_vocab = self._create_poi_vocab(vocab_source_ids)
        self.vocab_size = len(self.poi_vocab) + 1  # + pad idx

    def _create_poi_vocab(self, source_ids: List[str]):
        all_poi_types = set()
        for tid in source_ids:
            traj_data = self.traj_dict[tid]
            # main cell POIs
            for sublist in traj_data.get('poi_type', []) or []:
                for poi in (sublist or []):
                    if isinstance(poi, str):
                        all_poi_types.add(poi)
            # neighbor POIs
            for neighbors in traj_data.get('neighbor_poi_type', []) or []:
                for neighbor in (neighbors or []):
                    for npoi in (neighbor or []):
                        if isinstance(npoi, str):
                            all_poi_types.add(npoi)
        # Always include a special token to keep embedding index stable
        vocab = {poi: idx + 1 for idx, poi in enumerate(sorted(all_poi_types))}
        return vocab

    def __len__(self):
        return len(self.traj_ids)

    def __getitem__(self, idx: int):
        tid = self.traj_ids[idx]
        d = self.traj_dict[tid]

        # --- time-aligned fields & masks ---
        hour = list(d.get('hour', []))
        stay = list(d.get('stay_duration', []))
        if self.scale_stay_to_hours:
            stay = [s / 3600.0 for s in stay]
        if self.log1p_stay:
            stay = [np.log1p(max(s, 0.0)) for s in stay]

        L = min(len(hour), len(stay))
        hour = hour[:L]; stay = stay[:L]
        # pad/truncate to max_len
        hour_pad = hour[:self.max_len] + [HOUR_PAD] * max(0, self.max_len - len(hour))
        stay_pad = stay[:self.max_len] + [0.0] * max(0, self.max_len - len(stay))
        seq_mask = [1] * min(len(hour), self.max_len) + [0] * max(0, self.max_len - len(hour))

        # --- incidents (optional) ---
        inc = list(d.get('incident', [0.0] * len(hour)))
        inc = inc[:self.max_len] + [0.0] * max(0, self.max_len - len(inc))

        # --- POI ids ---
        poi_ids = []  # [L, max_seq]
        poi_mask = []
        for sublist in (d.get('poi_type', []) or [])[:self.max_len]:
            ids = [self.poi_vocab.get(pt, PAD_IDX) for pt in (sublist or [])]
            ids = ids[:self.max_seq] + [PAD_IDX] * max(0, self.max_seq - len(ids))
            mask = [1 if x != PAD_IDX else 0 for x in ids]
            poi_ids.append(ids)
            poi_mask.append(mask)
        # pad time steps if trajectory shorter than max_len
        while len(poi_ids) < self.max_len:
            poi_ids.append([PAD_IDX] * self.max_seq)
            poi_mask.append([0] * self.max_seq)

        # --- neighbor POI ids ---
        # shape target: [L, max_neighbors, max_seq]
        neigh_ids = []
        neigh_mask = []  # [L, max_neighbors]
        neighbors_seq = (d.get('neighbor_poi_type', []) or [])[:self.max_len]
        for neighbors in neighbors_seq:
            row_ids = []
            row_mask = []
            neighbors = neighbors or []
            # normalize neighbor count
            neighbors = neighbors[:self.max_neighbors] + [[]] * max(0, self.max_neighbors - len(neighbors))
            for n_pt in neighbors:
                ids = [self.poi_vocab.get(pt, PAD_IDX) for pt in (n_pt or [])]
                ids = ids[:self.max_seq] + [PAD_IDX] * max(0, self.max_seq - len(ids))
                row_ids.append(ids)
                row_mask.append(1 if any(x != PAD_IDX for x in ids) else 0)
            neigh_ids.append(row_ids)
            neigh_mask.append(row_mask)
        # pad time steps
        while len(neigh_ids) < self.max_len:
            neigh_ids.append([[PAD_IDX] * self.max_seq for _ in range(self.max_neighbors)])
            neigh_mask.append([0] * self.max_neighbors)

        # --- label ---
        y = float(d.get('is_anomaly', 0)) if self.label_as_float else int(bool(d.get('is_anomaly', 0)))

        sample = {
            'tid': tid,
            'hour': torch.tensor(hour_pad, dtype=torch.long),
            'stay_duration': torch.tensor(stay_pad, dtype=torch.float),
            'incident': torch.tensor(inc, dtype=torch.float),
            'seq_mask': torch.tensor(seq_mask, dtype=torch.bool),
            'poi_ids': torch.tensor(poi_ids, dtype=torch.long),
            'poi_mask': torch.tensor(poi_mask, dtype=torch.bool),
            'neighbor_ids': torch.tensor(neigh_ids, dtype=torch.long),
            'neighbor_mask': torch.tensor(neigh_mask, dtype=torch.bool),
            'labels': torch.tensor(y, dtype=torch.float if self.label_as_float else torch.long),
        }
        return sample


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch: List[dict], device: torch.device | None = None):
    def to(x):
        return x if device is None else x.to(device)

    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys() if k != 'tid'}
    out = {k: to(v) for k, v in out.items()}
    out['tid'] = [b['tid'] for b in batch]
    # expand feature dims for models expecting channels
    out['hour'] = out['hour'].unsqueeze(-1)
    out['stay_duration'] = out['stay_duration'].unsqueeze(-1)
    out['incident'] = out['incident'].unsqueeze(-1)
    return out


def _get_all_labels(dataset: TrajectoryDataset) -> np.ndarray:
    return np.array([int(bool(dataset.traj_dict[tid].get('is_anomaly', 0))) for tid in dataset.traj_ids], dtype=int)


def _stratified_split_indices(labels: np.ndarray, split=(0.8, 0.1, 0.1), seed: int = 42):
    assert abs(sum(split) - 1.0) < 1e-8, "Split ratios must sum to 1.0"
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    if len(classes) < 2:
        # Degenerate case: only one class present; fall back to random split
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(split[0] * n))
        n_val = int(round(split[1] * n))
        n_test = n - n_train - n_val
        return idx[:n_train].tolist(), idx[n_train:n_train+n_val].tolist(), idx[n_train+n_val:].tolist()

    idx_train, idx_val, idx_test = [], [], []
    for c in classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n = len(c_idx)
        n_train = int(round(split[0] * n))
        n_val = int(round(split[1] * n))
        n_test = n - n_train - n_val
        c_train = c_idx[:n_train]
        c_val = c_idx[n_train:n_train+n_val]
        c_test = c_idx[n_train+n_val:]
        idx_train.append(c_train); idx_val.append(c_val); idx_test.append(c_test)
    idx_train = np.concatenate(idx_train); rng.shuffle(idx_train)
    idx_val = np.concatenate(idx_val); rng.shuffle(idx_val)
    idx_test = np.concatenate(idx_test); rng.shuffle(idx_test)
    return idx_train.tolist(), idx_val.tolist(), idx_test.tolist()


def _format_binary_ratio(labels_subset: np.ndarray) -> str:
    n1 = int((labels_subset == 1).sum()); n0 = int((labels_subset == 0).sum())
    total = n1 + n0
    if total == 0: return "(1/0 : 0%/0%)"
    p1 = int(round(100.0 * n1 / total)); p0 = 100 - p1
    return f"(1/0 : {p1}%/{p0}%)"


def create_dataloaders(data_path: str, batch_size: int = 8, split=(0.8, 0.1, 0.1),
                       max_len: int = 120, max_seq: int = 16, max_neighbors: int = 7,
                       device=None, seed: int = 42, verbose: bool = True,
                       label_as_float: bool = False, scale_stay_to_hours: bool = True,
                       log1p_stay: bool = False):
    assert abs(sum(split) - 1.0) < 1e-8, "Split ratios must sum to 1.0"
    set_seed(seed)

    # First pass: temp dataset for labels + split
    tmp_dataset = TrajectoryDataset(data_path, max_len=max_len, max_seq=max_seq,
                                    max_neighbors=max_neighbors,
                                    scale_stay_to_hours=scale_stay_to_hours,
                                    log1p_stay=log1p_stay,
                                    label_as_float=label_as_float)
    labels = _get_all_labels(tmp_dataset)
    train_idx, val_idx, test_idx = _stratified_split_indices(labels, split=split, seed=seed)

    # Rebuild dataset with vocab learned only from train (avoid leakage across folds)
    train_ids = [tmp_dataset.traj_ids[i] for i in train_idx]
    dataset = TrajectoryDataset(data_path, max_len=max_len, max_seq=max_seq,
                                max_neighbors=max_neighbors,
                                scale_stay_to_hours=scale_stay_to_hours,
                                log1p_stay=log1p_stay,
                                label_as_float=label_as_float,
                                vocab_from_train_ids=train_ids)

    if verbose:
        print(f"Vocab size (including PAD=0): {dataset.vocab_size}")
        print(f"Train {_format_binary_ratio(labels[np.array(train_idx, dtype=int)])}")
        print(f"Val   {_format_binary_ratio(labels[np.array(val_idx, dtype=int)])}")
        print(f"Test  {_format_binary_ratio(labels[np.array(test_idx, dtype=int)])}")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, device=device))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, device=device))
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, device=device))

    return train_loader, val_loader, test_loader, dataset.vocab_size
