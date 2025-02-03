import os
import argparse
import yaml
import logging
import torch
import gc
import shutil
import argparse



import roadvision3d

from roadvision3d.src.engine.dataloader import build_dataloader
from roadvision3d.src.engine.model_builder import build_model
from roadvision3d.src.engine.tester import Tester
from roadvision3d.src.engine.trainer import Trainer
from roadvision3d.src.engine.optimizer import build_optimizer
from roadvision3d.src.engine.scheduler import build_lr_scheduler
from roadvision3d.src.engine.logger import Logger
from roadvision3d.src.engine.wandb_logger import WandbLogger



def parse_args():
    parser = argparse.ArgumentParser(description='Train or evaluate the model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training.')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on the validation set.')
    parser.add_argument('--test', action='store_true', help='Run evaluation on the test set.')
    parser.add_argument('--resume_model', type=str, default=None, help='Path to the model to resume from. Overwrites cfg[\'tester\'][\'resume_model\'] if provided.')
    return parser.parse_args()

   

def main():
    # load cfg
    args = parse_args()
    device = args.device
    torch.cuda.set_device(device)
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # Overwrite resume_model if provided
    if args.resume_model is not None:
        cfg['tester']['resume_model'] = args.resume_model

    os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)

    # Save a copy of the config file to log directory
    config_dst = os.path.join(cfg['trainer']['log_dir'], os.path.basename(args.config))
    shutil.copy(args.config, config_dst)
    
    logger = Logger(cfg['trainer']['log_dir'], 'train.log', cfg['trainer']['max_epoch'])

    # Initialize wandb
    wandb_logger = WandbLogger(
        project="RoadVision3D",
        config=cfg,
        name=cfg['wandb']['name'],  # Optional: Customize name
        notes=cfg['wandb']['notes'],  # Optional: Add any notes
        tags=["3D", "object-detection"]  # Optional: Add tags
    )    

    #  build dataloader
    print('Building dataloader...')
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])
    print('dataloader done')
    
    # build model
    print('Building model...')
    model = build_model(cfg, device)
    print('model done')

    # evaluation mode
    if args.evaluate:
        tester = Tester(cfg['tester'], cfg['dataset'], model, device, val_loader, logger)
        tester.test()
        return

    if args.test:
        tester = Tester(cfg['tester'], cfg['dataset'], model, device, test_loader, logger)
        tester.test()
        return
    
    #  build optimizer
    print('Building optimizer...')
    optimizer = build_optimizer(cfg['optimizer'], model)
    print('optimizer done')

    # build lr & bnm scheduler
    print('Building scheduler...')
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)
    print('scheduler done')

    print('Building trainer...')

    trainer = Trainer(cfg=cfg,
                      model=model,
                      device=device,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      wandb_logger=wandb_logger)
    print('Begin training...')
    trainer.train()



if __name__ == '__main__':
    main()