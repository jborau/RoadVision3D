import os
import tqdm

import torch
import torch.nn as nn
import numpy as np
import pdb
from roadvision3d.src.engine.model_saver import get_checkpoint_state
from roadvision3d.src.engine.model_saver import save_checkpoint
from roadvision3d.src.engine.model_saver import load_checkpoint
from roadvision3d.src.models.losses.loss_function import LSS_Loss,Hierarchical_Task_Learning
from roadvision3d.src.engine.decode_helper import extract_dets_from_outputs
from roadvision3d.src.engine.decode_helper import decode_detections

from roadvision3d.src.engine import eval


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 wandb_logger=None):
        self.cfg = cfg
        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.cfg_dataset = cfg['dataset']
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.wandb_logger = wandb_logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        self.label_dir = cfg['dataset']['label_dir']
        self.eval_cls = cfg['dataset']['eval_cls']
        if self.cfg_train.get('resume_model', None):
            assert os.path.exists(self.cfg_train['resume_model'])
            self.epoch = load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger, map_location=self.device)
            self.lr_scheduler.last_epoch = self.epoch - 1
        # self.model = torch.nn.DataParallel(model).to(self.device)
        self.model = model.to(self.device)
    def train(self):
        start_epoch = self.epoch
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(self.cfg, ei_loss)
        for epoch in range(start_epoch, self.cfg_train['max_epoch']):
            # train one epoch
            self.logger.log_train_epoch(epoch)
            current_lr = (self.warmup_lr_scheduler.get_lr()[0] if self.warmup_lr_scheduler is not None and epoch < 5
                          else self.lr_scheduler.get_lr()[0])
            self.logger.log_lr(current_lr)

            # Log learning rate to wandb
            if self.wandb_logger:
                self.wandb_logger.log_metrics({"learning_rate": current_lr, "epoch": epoch})

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss,self.epoch)

            self.logger.log_weights(loss_weights)

            ei_loss = self.train_one_epoch(loss_weights)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if ((self.epoch % self.cfg_train['eval_frequency']) == 0 and \
                self.epoch >= self.cfg_train['eval_start']):
                self.logger.log_val_epoch(self.epoch)
                results = self.eval_one_epoch()
                self.logger.log_val_results(results, ap_mode = 40)

                # Log evaluation results to wandb
                # Log evaluation results to wandb
                if self.wandb_logger:
                    flat_results = {}
                    for cls, metrics in results.items():
                        for metric, values in metrics.items():
                            # Assume values are a list corresponding to [easy, moderate, hard]
                            flat_results[f"eval/{cls}_{metric}_easy"] = values[0]  # Easy level
                            flat_results[f"eval/{cls}_{metric}_moderate"] = values[1]  # Moderate level
                            flat_results[f"eval/{cls}_{metric}_hard"] = values[2]  # Hard level
                    flat_results["epoch"] = self.epoch  # Add epoch explicitly for graphing

                    # Log flattened results
                    self.wandb_logger.log_metrics(flat_results)



            if ((self.epoch % self.cfg_train['save_frequency']) == 0
                and self.epoch >= self.cfg_train['eval_start']):
                os.makedirs(self.cfg_train['log_dir']+'/checkpoints', exist_ok=True)
                ckpt_name = os.path.join(self.cfg_train['log_dir']+'/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)

                # Log model checkpoint to wandb
                if self.wandb_logger:
                    self.wandb_logger.log_model(ckpt_name)

        self.logger.finish_training()
        if self.wandb_logger:
            self.wandb_logger.finish()
        return None
    
    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():        
            for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)
    
                # train one batch
                # criterion = LSS_Loss(self.epoch)
                # criterion = self.model.loss(self.epoch)
                # outputs = self.model(inputs,coord_ranges,calibs,targets)
                # _, loss_terms = criterion(outputs, targets)
                loss_terms = self.model(inputs, calibs, targets, coord_ranges, self.epoch) # , targets, self.epoch)
                
                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]      
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch             
        return disp_dict

    def train_one_epoch(self,loss_weights=None):
        self.model.train()

        disp_dict = {}
        stat_dict = {}
        for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
            if type(inputs) != dict:
                inputs = inputs.to(self.device)
            else:
                for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            for key in targets.keys(): targets[key] = targets[key].to(self.device)
            # train one batch
            self.optimizer.zero_grad()
            # criterion = LSS_Loss(self.epoch)
            # criterion = self.model.loss(self.epoch)
            # outputs = self.model(inputs,coord_ranges,calibs,targets)

            # total_loss, loss_terms = criterion(outputs, targets)
            loss_terms = self.model(inputs, calibs, targets, coord_ranges, self.epoch) # , targets, self.epoch)
            total_loss = float(sum(loss for loss in loss_terms.values()))

            
            if loss_weights is not None:
                total_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    total_loss += loss_weights[key].detach()*loss_terms[key]
            total_loss.backward()
            self.optimizer.step()

            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0

                if isinstance(loss_terms[key], int):
                    stat_dict[key] += (loss_terms[key])
                else:
                    stat_dict[key] += (loss_terms[key]).detach()
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                # disp_dict[key] += loss_terms[key]
                if isinstance(loss_terms[key], int):
                    disp_dict[key] += (loss_terms[key])
                else:
                    disp_dict[key] += (loss_terms[key]).detach()
            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                self.logger.log_iter(trained_batch, len(self.train_loader), disp_dict, self.cfg_train['disp_frequency'])

            # Log batch-level metrics to wandb
            if self.wandb_logger:
                self.wandb_logger.log_metrics({
                    f"batch/{key}": loss_terms[key].item() for key in loss_terms
                })
                self.wandb_logger.log_metrics({"batch/total_loss": total_loss.item(), "batch_idx": batch_idx})

                
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
        
        # Log epoch-level metrics to wandb
        if self.wandb_logger:
            # Log individual loss terms with explicit epoch
            for key in stat_dict:
                self.wandb_logger.log_metrics({
                    f"epoch/{key}": stat_dict[key].item(),
                    "epoch": self.epoch + 1 # Explicitly log the epoch
                })

            # Log the total loss with explicit epoch
            total_loss = sum(stat_dict.values()).item()
            self.wandb_logger.log_metrics({
                "epoch/total_loss": total_loss,
                "epoch": self.epoch + 1  # Explicitly log the epoch
            })

                            
        return stat_dict    
    def eval_one_epoch(self):
        self.model.eval()

        results = {}
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device) 
                coord_ranges = coord_ranges.to(self.device)
    
                info = {key: (val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val) for key, val in info.items()}
                info['calibs'] = [self.test_loader.dataset.get_calib(index)  for index in info['img_id']]
                dets = self.model(inputs, calibs, coord_ranges=coord_ranges, mode='val', info=info) # , targets, self.epoch)


                # dets = extract_dets_from_outputs(outputs, K=50)
                # dets = dets.detach().cpu().numpy()
                
                # # get corresponding calibs & transform tensor to numpy
                # calibs = [self.test_loader.dataset.get_calib(index)  for index in info['img_id']]
                # info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                # cls_mean_size = self.test_loader.dataset.cls_mean_size
                # dets = decode_detections(dets = dets,
                #                         info = info,
                #                         calibs = calibs,
                #                         cls_mean_size=cls_mean_size,
                #                         threshold = self.cfg_test['threshold'])
                results.update(dets)
                progress_bar.update()
            progress_bar.close()
        # self.save_results(results)
        out_dir = os.path.join(self.cfg_train['out_dir'], 'EPOCH_' + str(self.epoch))
        self.save_results(results, out_dir)
        results = eval.eval_from_scrach(
            self.label_dir,
            os.path.join(out_dir, 'data'),
            self.cfg_dataset,
            'trainval',
            self.eval_cls,
            ap_mode=40)
        return results

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            # Check for both normal int and numpy integer types
            if isinstance(img_id, (int, np.integer)):
                img_id = int(img_id)  # Convert to a normal Python int if it's a numpy integer
                out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                out_path = os.path.join(output_dir, f'{img_id}.txt')
        
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()        
        
      