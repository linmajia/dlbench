import os
import requests
import subprocess as sp
import sys
import zipfile

from tools.common.util import calculate_sha256


class Runner(object):

    @classmethod
    def create_cpu_env(clz, thread_count):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '-1' # like '1,2'
        env['OMP_NUM_THREADS'] = str(thread_count)
        env['OPENBLAS_NUM_THREADS'] = str(thread_count)
        env['MKL_NUM_THREADS'] = str(thread_count)
        return env

    @classmethod
    def create_gpu_env(clz, gpu_id_list):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_id_list # like '1,2'
        if 'OMP_NUM_THREADS' in env: del env['OMP_NUM_THREADS']
        if 'OPENBLAS_NUM_THREADS' in env: del env['OPENBLAS_NUM_THREADS']
        if 'MKL_NUM_THREADS' in env: del env['MKL_NUM_THREADS']
        return env

    def download_file(self, url, sha256, data_dir):
        if not os.path.isdir(data_dir): os.makedirs(data_dir)
        download = True
        file_name = url.split('/')[-1]
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            self.logger.info('Finding file at: ' + url)
            if sha256:
                file_hash = calculate_sha256(file_path).lower()
                if sha256.lower() == file_hash.lower():
                    download = False
                else:
                    self.logger.info('Deleting %s as mismatched SHA256' % file_path)
                    os.remove(file_path)
            else:
                download = False    
        if download:
            self.logger.info('Downloading file from: ' + url)
            response = requests.get(url, stream=True)
            with open(file_path, "wb") as fout:
                index = 0
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:
                        fout.write(chunk)
                        index += 1
                        if index % 500 == 0:
                            sys.stderr.write('.')
                            sys.stderr.flush()
                sys.stderr.write('\n')
        return file_path
        
    def decompress_zip(self, zip_path, data_dir):
        self.logger.info('Decompressing zip to: ' + data_dir)
        with zipfile.ZipFile(zip_path, 'r') as fin:
            fin.extractall(data_dir)
    
    def execute_cmd(self, cmd, log_path, cwd=None, env=None):
        if log_path: cmd += ' >%s 2>&1' % log_path
        self.logger.info('Executing CMD: ' + cmd)
        process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, cwd=cwd, env=env, shell=True)
        if not log_path:
            for line in iter(process.stdout.readline, b''): print(line.decode())
        process.wait()
        if process.returncode == 0:
            self.logger.info('CMD returned exit code 0.')
            return True
        else:
            self.logger.error('CMD failed with exit code %d!' % process.returncode)
            return False

    def start_experiment(self, exp, log_dir):
        ''' Execute experiment by parameters and return average seconds per batch. '''
        raise Exception('Method "start_experiment" is not Implemented!')
