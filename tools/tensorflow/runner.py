import os

from tools.common.runner import Runner
from tools.common.util import get_logger, get_timestr


NET_TO_FILE = {
    'fcn-r': 'fc' + os.path.sep + 'fcn5_mnist.py',
    'alexnet-r': 'cnn/alexnet/alexnet_cifar10.py'.replace('/', os.path.sep),
    'resnet-56': 'cnn/resnet/resnet_cifar10.py'.replace('/', os.path.sep),
    'lstm': 'rnn/lstm/lstm.py'.replace('/', os.path.sep)
}

NET_TO_DATA = {
    'fcn-r': 'MNIST_data',
    'alexnet-r': 'cifar-10-batches-py',
    'resnet-56': 'cifar-10-batches-py',
    'lstm': 'lstm_data' + os.path.sep + 'data'
}


class TensorFlowRunner(Runner):

    def __init__(self, data_dir):
        self.logger = get_logger('tensorflow')
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self._prepare_data(data_dir)
    
    def _prepare_data(self, data_dir):
        self.data_dir = os.path.join(data_dir, 'tensorflow')
        if not os.path.isdir(self.data_dir):
            zip_path = self.download_file('http://dlbench.comp.hkbu.edu.hk/s/data/tensorflow.zip',
                '342D1F7CAC27D6025856101CE491EAF5B90F83A2A24E16C4CD7717A093360B0D', data_dir)
            self.decompress_zip(zip_path, data_dir)

    def start_experiment(self, exp, log_dir):
        self.logger.info('Run with ' + str(exp.__dict__))
        if exp.net_name in NET_TO_FILE:
            # Prepare runtime and command
            env = None
            if exp.is_gpu():
                if exp.device_count > 1:
                    self.logger.warning('Skip as multiple GPU card is not supported!')
                    return
                self.create_gpu_env(exp.device_id)
            else:
                env = self.create_cpu_env(exp.device_count)
            script_path = os.path.join(self.base_dir, NET_TO_FILE[exp.net_name])
            dataset = os.path.join(self.data_dir, NET_TO_DATA[exp.net_name])
            cmd = ('python -u %s --data_dir=%s --batch_size=%d --epochs=%d --epoch_size=%d --device_id=%s --lr=%f' %
                (script_path, dataset, exp.batch_size, exp.num_epoch, exp.epoch_size, exp.device_id, exp.learning_rate))
            log_path = os.path.join(log_dir, 'tensorflow_%s.log' % str(exp).replace(';', '_'))
            
            # Execute and fetch result
            if self.execute_cmd(cmd, log_path, cwd=os.path.dirname(script_path), env=env):
                with open(log_path, mode='r') as fin:
                    for line in fin.readlines():
                        if line.startswith('average_batch_time'):
                            seconds_per_batch = float(line.split(':')[-1].strip())
                            self.logger.info('Average seconds per batch: %.4f' % seconds_per_batch)
                            return seconds_per_batch
                self.logger.error('Cannot find "average_batch_time" in %s!' % log_path)
        else:
            self.logger.warning('Skip as %s does not register!' % exp.net_name)

