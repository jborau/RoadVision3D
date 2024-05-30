import yaml

config_file = '/home/javier/pytorch/RoadVision3D/roadvision3d/configs/kitti.yaml'

cfg = yaml.load(open(config_file, 'r'), Loader=yaml.Loader)

print(cfg['dataset'])