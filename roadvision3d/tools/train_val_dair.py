import os
import argparse
import yaml
import logging

import roadvision3d

from roadvision3d.src.engine.dataloader import build_dataloader
from roadvision3d.src.engine.model_builder import build_model
from roadvision3d.src.engine.tester import Tester
from roadvision3d.src.engine.trainer import Trainer
from roadvision3d.src.engine.optimizer import build_optimizer
from roadvision3d.src.engine.scheduler import build_lr_scheduler
from roadvision3d.src.engine.logger import Logger


# TODO: update this
parser = argparse.ArgumentParser(description='implementation of MonoLSS')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--config', type=str, default='/home/javier/pytorch/RoadVision3D/roadvision3d/configs/dair_kitti.yaml')
# parser.add_argument('--config', type=str, default='/home/javi/Desktop/server/RoadVision3D/roadvision3d/configs/kitti.yaml')
args = parser.parse_args()    

def main():
    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
    logger = Logger(cfg['trainer']['log_dir'], 'train.log', cfg['trainer']['max_epoch'])

    import shutil
    # TODO: Update this
    # if not args.evaluate:
    #     if not args.test:
    #         if os.path.exists(os.path.join(cfg['trainer']['log_dir'], 'lib/')):
    #             shutil.rmtree(os.path.join(cfg['trainer']['log_dir'], 'lib/'))
    #     if not args.test:
    #         shutil.copytree('./lib', os.path.join(cfg['trainer']['log_dir'], 'lib/'))
        
    
    #  build dataloader
    print('Building dataloader...')
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])
    print('dataloader done')

    # build model
    print('Building model...')
    model = build_model(cfg)
    print('model done')

    # evaluation mode
    if args.evaluate:
        tester = Tester(cfg['tester'], cfg['dataset'], model, val_loader, logger)
        tester.test()
        return

    if args.test:
        tester = Tester(cfg['tester'], cfg['dataset'], model, test_loader, logger)
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
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger)
    print('Begin training...')
    trainer.train()



if __name__ == '__main__':
    main()