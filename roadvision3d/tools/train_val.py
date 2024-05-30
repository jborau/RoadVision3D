import os
import argparse
import yaml
import logging

import roadvision3d

from roadvision3d.src.engine.dataloader import build_dataloader
from roadvision3d.src.engine.model_builder import build_model


# TODO: update this
parser = argparse.ArgumentParser(description='implementation of MonoLSS')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--config', type=str, default='/home/javier/pytorch/RoadVision3D/roadvision3d/configs/kitti.yaml')
args = parser.parse_args()

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)
    

def main():
    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'], 'train.log'))

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
    model = build_model(cfg['model'], train_loader.dataset.cls_mean_size)
    print('model done')




if __name__ == '__main__':
    main()