import os

from tools.common.runner import Runner
from tools.common.util import get_logger


NET_TO_FILE = {
    'fcn-r': 'train_mnist.py',
    'alexnet-r': 'train_cifar10.py --network alexnet',
    'resnet-56': 'train_cifar10.py --network resnet',
    'lstm': 'train_rnn.py --sequence-lens 32'
}

NET_TO_DATA = {
    'fcn-r': 'mnist',
    'alexnet-r': 'cifar10_32',
    'resnet-56': 'cifar10_32',
    'lstm': 'ptb'
}


class MXNetRunner(Runner):

    def __init__(self, data_dir):
        self.logger = get_logger('mxnet')
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self._prepare_data(data_dir)
    
    def _prepare_data(self, data_dir):
        self.data_dir = os.path.join(data_dir, 'mxnet')
        if not os.path.isdir(self.data_dir):
            zip_path = self.download_file('http://dlbench.comp.hkbu.edu.hk/s/data/mxnet.zip',
                'E95B65B8CFECBEEA901D53B62C93EB200783AEF444C89E2B3F97CA974C9586BA', data_dir)
            self.decompress_zip(zip_path, data_dir)

    def _parse_log_and_calculate_result(self, log_path, exp):
        total_time, num_epoch, num_example = 0.0, 0, exp.epoch_size
        with open(log_path, mode='r') as fin:
            for line in fin.readlines():
                if line.find('Time cost=') >= 0: # like 'INFO:root:Epoch[1] Time cost=3.737'
                    total_time += float(line.split('=')[1].strip())
                    num_epoch += 1
                if line.startswith('len of data train'): # like 'len of data train===================== 35879'
                    # LSTM script discards sentences too long, so epoch size differs from experiment parameter
                    num_example = int(line.split()[-1])
        if total_time > 0:
            epoch_time = total_time / num_epoch
            batch_count = num_example / exp.batch_size
            seconds_per_batch = epoch_time / batch_count
            self.logger.info('Average seconds per batch: %.4f' % seconds_per_batch)
            return seconds_per_batch
        else:
            self.logger.error('Cannot find "Time cost=" in %s!' % log_path)
    
    def start_experiment(self, exp, log_dir):
        self.logger.info('Run with ' + str(exp.__dict__))
        if exp.net_name in NET_TO_FILE:
            # Prepare runtime and command
            env, device_argument = None, ''
            if exp.is_gpu():
                if exp.device_count > 1:
                    self.logger.warning('Skip as multiple GPU card is not supported!')
                    return
                env = self.create_gpu_env(exp.device_id)
                device_argument = ' --gpus ' + exp.device_id
            else:
                env = self.create_cpu_env(exp.device_count)
            script_path = self.base_dir + os.path.sep + NET_TO_FILE[exp.net_name]
            dataset = os.path.join(self.data_dir, NET_TO_DATA[exp.net_name])
            cmd = ('python -u %s --data-dir %s --batch-size %s --num-epochs %d --num-examples %d --lr %f' %
                (script_path, dataset, exp.batch_size, exp.num_epoch, exp.epoch_size, exp.learning_rate))
            cmd += device_argument
            log_path = os.path.join(log_dir, 'mxnet_%s.log' % str(exp).replace(";", "_"))
             
            # Execute and fetch result
            cwd = os.path.dirname(script_path.split()[0]) # Remove arguments
            if self.execute_cmd(cmd, log_path, cwd=cwd, env=env):
                return self._parse_log_and_calculate_result(log_path, exp)
        else:
            self.logger.warning('Skip as %s does not register!' % exp.net_name)
