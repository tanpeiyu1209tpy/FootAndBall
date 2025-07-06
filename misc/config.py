import os
import configparser
import time

class Params:
    def __init__(self, path):
        assert os.path.exists(path), 'Cannot find configuration file: {}'.format(path)
        self.path = path
        config = configparser.ConfigParser()
        config.read(self.path)
        params = config['DEFAULT']
        
        # 数据集相关参数
        self.issia_path = params.get('issia_path', None)
        if self.issia_path is not None:
            temp = params.get('issia_train_cameras', '1, 2, 3, 4')
            self.issia_train_cameras = [int(e) for e in temp.split(',')]
            temp = params.get('issia_val_cameras', '5')
            self.issia_val_cameras = [int(e) for e in temp.split(',')]
            
        # 训练相关参数
        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 4)
        self.epochs = params.getint('epochs', 20)
        self.lr = params.getfloat('lr', 1e-3)
        
        # 删除model相关参数，因为通过命令行控制
        # 删除temporal相关参数，因为通过命令行控制
        
        self._check_params()
    
    def _check_params(self):
        assert os.path.exists(self.issia_path), "Cannot access ISSIA CNR dataset: {}".format(self.issia_path)
        for c in self.issia_train_cameras:
            assert 1 <= c <= 6, 'ISSIA CNR camera number must be between 1 and 6. Is: {}'.format(c)
        for c in self.issia_val_cameras:
            assert 1 <= c <= 6, 'ISSIA CNR camera number must be between 1 and 6. Is: {}'.format(c)
    
    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))
        print('')

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

# 删除TemporalParams类，因为不需要了
