import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Subset


def dataset_random_split(path, dataset, train_size=0.7, val_size=0.15, test_size=0.15, new_split=True):
    if new_split == True:
        train_size = int(train_size * len(dataset))
        val_size = int(val_size * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        # Save indices
        train_idx = train_dataset.indices
        val_idx = val_dataset.indices
        test_idx = test_dataset.indices
        
        np.save(f'{path}/train_idx.npy', np.array(train_idx))
        np.save(f'{path}/val_idx.npy', np.array(val_idx))
        np.save(f'{path}/test_idx.npy', np.array(test_idx))
    
    # Load subsets
    train_dataset = Subset(dataset, np.load(f'{path}/train_idx.npy'))
    val_dataset = Subset(dataset, np.load(f'{path}/val_idx.npy'))
    test_dataset = Subset(dataset, np.load(f'{path}/test_idx.npy'))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader