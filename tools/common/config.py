import platform

class ExperimentConfig(object):

    def __init__(self, str_to_eval):
        self.net_name = None
        self.device_id = "-1" # CPU
        self.device_count = 1
        self.batch_size = 1
        self.epoch_size = 1
        self.num_epoch = 0
        self.learning_rate = 0.01
        if str_to_eval: self.load_from_str(str_to_eval)

    def load_from_str(self, str_to_eval):
        fields = str_to_eval.split(';')
        # fields[0] is tag text to be ignored for compatibility issue
        self.net_name = fields[1]
        self.device_id = fields[2]
        self.device_count = int(fields[3])
        self.batch_size = int(fields[4])
        self.num_epoch = int(fields[5])
        self.epoch_size = int(fields[6])
        self.learning_rate = float(fields[7])
    
    def is_gpu(self):
        return ('-' not in self.device_id)

    def __str__(self):
        return ('%s;%s;%d;%d;%d;%d;%f' %
            (self.net_name, self.device_id, self.device_count, self.batch_size, self.epoch_size, self.num_epoch, self.learning_rate))


class BenchmarkConfig(object):

    def __init__(self, path=None):
        self.host_name = platform.node()
        self.cpu_name = platform.processor()
        self.gpu_name = None
        self.cuda = None
        self.cudnn = None
        self.cuda_driver = None
        self.tools = []
        self.experiments = []        
        if path: self.load_from_file(path)

    def load_from_file(self, path):
        with open(path, mode='r', encoding='utf-8') as f:
            config_experiments = False
            for line in f.readlines():
                line = line.split('#')[0].replace('\t','').replace(' ', '') # Remove comments and spaces
                if not line: continue # Skip empty lines
                if not config_experiments:
                    if 'tools:' in line:
                        self.tools = [item.lower() for item in line.split(':')[1].split(',')]
                    elif '{' in line:
                        config_experiments = True
                    elif 'gpu_name:' in line:
                        self.gpu_name = line.split(':')[1]
                    elif 'cuda:' in line:
                        self.cuda = line.split(':')[1]
                    elif 'cudnn:' in line:
                        self.cudnn = line.split(':')[1]
                    elif 'cuda_driver:' in line:
                        self.cuda_driver = line.split(':')[1]
                else:
                    if '}' in line:
                        config_experiments = False
                    else:
                        exp = ExperimentConfig(line)
                        self.experiments.append(exp)
    
    def __str__(self):
        experiments = [str(exp) for exp in self.experiments]
        return (('host_name:%s\n'
            + 'tools:%s\n'
            + 'cpu_name:%s\n'
            + 'gpu_name:%s\n'
            + 'cuda:%s\n'
            + 'cudnn:%s\n'
            + 'cuda_driver:%s\n'
            + '{\n'
            + '%s\n'
            + '}') %
            (self.host_name, ','.join(self.tools), self.cpu_name, self.gpu_name, self.cuda, self.cudnn, self.cuda_driver, '\n'.join(experiments)))
