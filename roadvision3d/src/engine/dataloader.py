import numpy as np
from torch.utils.data import DataLoader
from roadvision3d.src.datasets.kitti import KITTI
from roadvision3d.src.datasets.dair_kitti import DAIR_KITTI
from roadvision3d.src.datasets.dair import DAIR
from roadvision3d.src.datasets.rope3d import Rope3D



def build_dataloader(cfg):
    # --------------  build kitti dataset ----------------
    if cfg['type'] == 'kitti':
        train_set = KITTI(split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_set = KITTI(split='val', cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=cfg['drop_last_val'])
        test_set = KITTI(split='test', cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader, test_loader
    
    # --------------  build dair_kitti dataset ----------------
    if cfg['type'] == 'dair_kitti':
        train_set = DAIR_KITTI(split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_set = DAIR_KITTI(split='val', cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=cfg['drop_last_val'])
        test_set = DAIR_KITTI(split='test', cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader, test_loader
    
    # --------------  build dair dataset ----------------
    if cfg['type'] == 'dair':
        train_set = DAIR(split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_set = DAIR(split='val', cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=cfg['drop_last_val'])
        test_set = DAIR(split='test', cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader, test_loader
    
    # --------------  build Rope3D dataset ----------------
    if cfg['type'] == 'rope3d':
        train_set = Rope3D(split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_set = Rope3D(split='val', cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=cfg['drop_last_val'])
        test_set = Rope3D(split='test', cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader, test_loader
    
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])