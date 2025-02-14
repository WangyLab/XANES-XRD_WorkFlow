import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

def load_data(xrd_csv_path, json_path):
    """Load and merge XRD data from CSV and additional properties from JSON."""
    df1 = pd.read_csv(xrd_csv_path)
    df2 = pd.read_json(json_path)
    df = pd.merge(df2, df1[['material_id', '2theta_list', 'intensity_list']], on='material_id', how='left')
    return df

def analyze_sequence_lengths(df):
    """Analyze the sequence length of 2theta peaks."""
    sequence_lengths = [len(ast.literal_eval(row['2theta_list'])) for _, row in df.iterrows()]
    return {
        'min': min(sequence_lengths),
        'max': max(sequence_lengths),
        'mean': np.mean(sequence_lengths),
        'median': np.median(sequence_lengths),
        '95th_percentile': sorted(sequence_lengths)[int(0.95 * len(sequence_lengths))]
    }
    
def preprocess_xrd_data(df):
    """Extract and normalize peaks while mapping crystal systems."""
    peaks_info = []
    crystal_systems = []
    scaler = MinMaxScaler()
    scaler.fit(np.array([10, 90]).reshape(-1, 1))
    
    for _, row in df.iterrows():
        peaks_2theta = ast.literal_eval(row['2theta_list'])
        peaks_2theta = scaler.transform(np.array(peaks_2theta).reshape(-1, 1)).flatten().tolist()
        peaks_info.append(peaks_2theta)
        
        match = re.search(r"crystal_system=<CrystalSystem\.(.*?):", row['symmetry'])
        crystal_systems.append(match.group(1) if match else 'unknown')
    
    return peaks_info, crystal_systems

def map_crystal_systems(crystal_systems):
    """Map crystal systems to numerical groups."""
    group_mapping = {
        'cubic': 'group1', 'hex_': 'group2', 'trig': 'group3',
        'ortho': 'group4', 'tet': 'group5', 'mono': 'group6', 'tri': 'group7'
    }
    grouped_crystal_systems = [group_mapping.get(crystal, 'unknown') for crystal in crystal_systems]
    
    unique_groups = sorted(set(grouped_crystal_systems))
    group_to_index = {group: index for index, group in enumerate(unique_groups)}
    mapped_targets = [group_to_index[group] for group in grouped_crystal_systems]
    
    return mapped_targets

class SortedXRD_Dataset(Dataset):
    """Custom dataset that sorts sequences by length for efficient batch processing."""
    def __init__(self, sequences, targets):
        sorted_indices = sorted(range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True)
        self.sequences = [sequences[i] for i in sorted_indices]
        self.targets = [targets[i] for i in sorted_indices]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]
    
def prepare_dataset(xrd_csv, json_path):
    """Load, process, and return the dataset for training."""
    df = load_data(xrd_csv, json_path)
    print("Sequence length statistics:", analyze_sequence_lengths(df))
    
    peaks_info, crystal_systems = preprocess_xrd_data(df)
    mapped_targets = map_crystal_systems(crystal_systems)
    
    dataset = SortedXRD_Dataset(peaks_info, mapped_targets)
    return dataset

def collate_fn(batch):
    sequences, targets = zip(*batch)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in sequences]
    masks = [[1 if val != 0 else 0 for val in seq] for seq in padded_sequences]
    lengths = [len(seq) for seq in sequences]
    # Convert to tensors
    sequences_tensor = torch.tensor(padded_sequences, dtype=torch.float32)
    masks_tensor = torch.tensor(masks, dtype=torch.bool)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return sequences_tensor, lengths_tensor, targets_tensor

# Split data
def dataset_random_split(dataset, train_size=0.7, val_size=0.15, test_size=0.15, new_split=False):
    if new_split == True:
        train_size = int(train_size * len(dataset))
        val_size = int(val_size * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        # Save indices
        train_idx = train_dataset.indices
        val_idx = val_dataset.indices
        test_idx = test_dataset.indices
        
        np.save('train_idx.npy', np.array(train_idx))
        np.save('val_idx.npy', np.array(val_idx))
        np.save('test_idx.npy', np.array(test_idx))
    
    # Load subsets
    train_dataset = Subset(dataset, np.load('train_idx.npy'))
    val_dataset = Subset(dataset, np.load('val_idx.npy'))
    test_dataset = Subset(dataset, np.load('test_idx.npy'))
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader