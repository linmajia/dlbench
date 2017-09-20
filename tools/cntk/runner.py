import os

from tools.common.runner import Runner
from tools.common.util import get_logger


NET_TO_FILE = {
    'fcn-r': 'fc' + os.path.sep + 'fcn5.cntk',
    'alexnet-r': 'cnn/alexnet/alexnet.cntk'.replace('/', os.path.sep),
    'resnet-56': 'cnn/resnet/resnet.cntk'.replace('/', os.path.sep),
    'lstm': 'rnn/lstm/lstm.cntk'.replace('/', os.path.sep)
}

NET_TO_DATA = {
    'fcn-r': 'mnist',
    'alexnet-r': 'cifar10',
    'resnet-56': 'cifar10',
    'lstm': 'ptb'
}


class CntkRunner(Runner):

    def __init__(self, data_dir):
        self.logger = get_logger('cntk')
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self._prepare_data(data_dir)
    
    def _prepare_data(self, data_dir):
        self.data_dir = os.path.join(data_dir, 'cntk')
        if not os.path.isdir(self.data_dir):
            zip_path = self.download_file('http://dlbench.comp.hkbu.edu.hk/s/data/cntk.zip',
                '9A16CF9888CF7A58522E7C7FBB1B066871D5B4FE9E1E429615DB709AD1F6C195', data_dir)
            self.decompress_zip(zip_path, data_dir)

    def _parse_log_and_calculate_result(self, log_path, batch_size, node_count=1):
        # Find lines like 'Finished Epoch[ 2 of 4]: [Training] ce = 2.29091309 * 5000; errs = 86.780% * 5000; totalSamplesSeen = 10000; learningRatePerSample = 9.7656248e-06; epochTime=0.112997s'
        useful_lines = []
        with open(log_path, mode='r') as fin:
            for line in fin.readlines():
                if line.startswith('Finished Epoch'): useful_lines.append(line)
        
        # Accumulate and calculate average time per batch
        if len(useful_lines) > 0:
            total_time, epoch_count = 0.0, 0
            for line in useful_lines:
                fields = line.split('=')
                total_time += float(line.split('=')[-1].strip().rstrip('s'))
                epoch_count += 1
            epoch_time = total_time / epoch_count
            epoch_size = int(useful_lines[0].split('=')[1].split(';')[0].split('*')[1].strip())
            actual_batch_size = batch_size * node_count
            batch_count = (epoch_size + actual_batch_size - 1) / actual_batch_size
            seconds_per_batch = epoch_time / batch_count
            self.logger.info('Average seconds per batch: %.4f' % seconds_per_batch)
            return seconds_per_batch
        else:
            self.logger.error('Cannot find "Finished Epoch" in %s!' % log_path)        

    def start_experiment(self, exp, log_dir):
        self.logger.info('Run with ' + str(exp.__dict__))
        if exp.net_name in NET_TO_FILE:
            # Prepare runtime and command
            env, device_id = (None, 'auto')
            if exp.is_gpu():
                if exp.device_count > 1:
                    self.logger.warning('Skip as multiple GPU card is not supported!')
                    return
                env = self.create_gpu_env(exp.device_id)
            else:
                env = self.create_cpu_env(exp.device_count)
                device_id = -1
            script_path = os.path.join(self.base_dir, NET_TO_FILE[exp.net_name])
            dataset = os.path.join(self.data_dir, NET_TO_DATA[exp.net_name])
            cmd = ('cntk configFile=%s makeMode=false DataDir=%s deviceId=%s minibatchSize=%d epochSize=%d maxEpochs=%d' %
                (script_path, dataset, device_id, exp.batch_size, max(0, exp.epoch_size), exp.num_epoch))
            log_path = os.path.join(log_dir, 'cntk_%s.log' % str(exp).replace(";", "_"))
            
            # Execute and fetch result
            if self.execute_cmd(cmd, log_path, cwd=os.path.dirname(script_path), env=env):
                return self._parse_log_and_calculate_result(log_path, exp.batch_size)
        else:
            self.logger.warning('Skip as %s does not register!' % exp.net_name)
