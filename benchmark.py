import argparse
import importlib
import os

from tools.common.config import BenchmarkConfig
from tools.common.util import get_logger, get_timestr


def visualize_result(statistics):
    ''' Draw a table displaying benchmark result. '''
    statistics.sort(key=lambda item: item[1].net_name + '_' + item[0]) # Sort by network, tool
    text = 'net_name,tool,device_id,thread_or_gpu_count,batch_size,epoch_size,num_epoch,learning_rate,seconds_per_batch\n'
    for tool, exp, time in statistics:
        text += ('%s,%s,%s,%d,%d,%d,%d,%.4f,%.4f\n' %
            (exp.net_name, tool, exp.device_id, exp.device_count, exp.batch_size, exp.epoch_size, exp.num_epoch, exp.learning_rate, time))
    return text


def get_runner(tool, data_dir):
    ''' Use reflection technique to instantiate runner of each tool. '''
    module = importlib.import_module('tools.%s.runner' % tool)
    for attr_name in dir(module):
        if attr_name.lower() == tool + 'runner':
            clazz = getattr(module, attr_name)
            return clazz(data_dir)
    raise Exception("Cannot find runner class in %s!" % module)


def dispatch(args, logger):
    # Prepare directory for output
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'dataset')
    log_dir = os.path.abspath(args.result) if args.result else os.path.join(root_dir, 'logs', 'run_' + get_timestr())
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    logger.info('Log directory is: ' + log_dir)
    
    # Execute each DL experiment
    statistics = []
    logger.info('Start to run DL benchmark ===>')
    config = BenchmarkConfig(args.config)
    logger.info('Benchmark Configuration is:\n' + str(config))
    for tool in config.tools:
        runner = get_runner(tool, data_dir)
        for exp in config.experiments:
            seconds_per_batch = runner.start_experiment(exp, log_dir)
            if seconds_per_batch: statistics.append((tool, exp, seconds_per_batch))
    logger.info('Finish benchmark and statistics is:')

    # Save result and print
    text = visualize_result(statistics)
    print(text)
    log_path = os.path.join(log_dir, 'result.csv')
    with open(log_path, mode='w') as fout:
        logger.info('Result is saved to: ' + log_path)
        fout.write(text)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Benchmark deep learning tools')
    parser.add_argument('-config', required=True, type=str, help='Path to the config file.')
    parser.add_argument('-result', default=None, type=str, help='Directory to store the result.')
    args, _ = parser.parse_known_args()
    
    # Launch dispatcher
    logger = get_logger('dlbench')
    logger.info('Parsed args: ' + str(args))
    dispatch(args, logger)
