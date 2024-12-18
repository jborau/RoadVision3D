import logging
import os
import time
from datetime import datetime, timedelta
import json
import torch

from .analysis import generate_and_save_plots

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


class Logger:
    def __init__(self, log_dir, log_name, total_epoch):
        self.log_file = os.path.join(log_dir, log_name)
        self.json_file = os.path.join(log_dir, "train.json")
        self.logger = create_logger(self.log_file)
        self.total_epoch = total_epoch
        self.data_for_json = self.load_existing_data()

    def load_existing_data(self):
        """Carga datos existentes desde el archivo JSON si existe."""
        try:
            with open(self.json_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return []
        
    def write_json(self):
        """Writes the accumulated data in JSON format to a file."""
        with open(self.json_file, 'w') as f:
            json.dump(self.data_for_json, f, indent=4)

    def log_train_epoch(self, epoch):
        if epoch == 0:
            self.start_time = time.time()
        self.epoch = epoch
        self.logger.info('------ TRAIN EPOCH %03d ------' %(self.epoch + 1))

    def log_lr(self, lr):
        self.lr = lr
        self.logger.info('Learning Rate: %.8f' % self.lr)

    def log_weights(self, loss_weights):
        log_str = 'Weights: '
        for key in sorted(loss_weights.keys()):
            log_str += ' %s:%.4f,' %(key[:-4], loss_weights[key])   
        self.logger.info(log_str)

    def log_iter(self, iter, total_iter, disp_dict, disp_freq):
        log_str = 'BATCH[%04d/%04d]' % (iter, total_iter)
        data_dict = {
            'epoch': self.epoch + 1,
            'iter': iter,
            'lr': self.lr,
            'data': {}
        }

        for key in sorted(disp_dict.keys()):
            # Perform the division
            disp_value = disp_dict[key] / disp_freq

            # Convert to float if it's a Tensor
            if isinstance(disp_value, torch.Tensor):
                value = disp_value.item()
            else:
                value = float(disp_value)

            log_str += ' %s:%.4f,' % (key, value)
            data_dict['data'][key] = value  # Store the float value
            disp_dict[key] = 0  # Reset statistics
        
        # Calculate ETA
        elapsed_time = time.time() - self.start_time
        iters_completed = iter + total_iter * self.epoch
        total_iterations = total_iter * self.total_epoch
        eta = (elapsed_time * (total_iterations - iters_completed) / iters_completed) if iters_completed != 0 else 0
        eta_str = str(timedelta(seconds=int(eta)))
        log_str += f' ETA: {eta_str}'

        self.logger.info(log_str)
        self.data_for_json.append(data_dict)
        self.write_json()

    def log_val_epoch(self, epoch):
        self.logger.info('------ EVAL EPOCH %03d ------' % (epoch))

    def log_test_epoch(self):
        self.logger.info('------ TEST ------')

    def log_val_results(self, results, ap_mode):
        formatted_results = []
        for cls, metrics in results.items():
            formatted_results.append(f"Class: {cls} AP" + str(ap_mode))
            for metric, values in metrics.items():
                values_str = ', '.join(map(str, values))
                formatted_results.append(f"  {metric}: {values_str}")
        self.logger.info('\n'.join(formatted_results))
    
    def finish_training(self):
        self.logger.info('------ TRAINING FINISHED ------')
        self.logger.info('Generating plots...')
        generate_and_save_plots(self.json_file)
        self.logger.info('Plots generated successfully.')
