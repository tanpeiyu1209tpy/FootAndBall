# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming
'''
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
        self.issia_path = params.get('issia_path', None)
        if self.issia_path is not None:
            temp = params.get('issia_train_cameras', '1, 2, 3, 4')
            self.issia_train_cameras = [int(e) for e in temp.split(',')]
            temp = params.get('issia_val_cameras', '5')
            self.issia_val_cameras = [int(e) for e in temp.split(',')]
            

        #self.spd_path = params.get('spd_path', None)
        #if self.spd_path is not None:
        #    temp = params.get('spd_set', '1, 2')
        #    self.spd_set = [int(e) for e in temp.split(',')]

        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 4)
        self.epochs = params.getint('epochs', 20)
        self.lr = params.getfloat('lr', 1e-3)

        self.model = params.get('model', 'fb1')
        self.model_name = 'model_{}_{}'.format(self.model, get_datetime())

        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.issia_path), "Cannot access ISSIA CNR dataset: {}".format(self.issia_path)
        #assert os.path.exists(self.spd_path), "Cannot access SoccerPlayerDetection_bmvc17 dataset: {}".format(self.spd_path)
        for c in self.issia_train_cameras:
            assert 1 <= c <= 6, 'ISSIA CNR camera number must be between 1 and 6. Is: {}'.format(c)
        for c in self.issia_val_cameras:
            assert 1 <= c <= 6, 'ISSIA CNR camera number must be between 1 and 6. Is: {}'.format(c)
        #for c in self.spd_set:
        #    assert c == 1 or c == 2, 'SPD dataset number must be 1 or 2. Is: {}'.format(c)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))
        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

class TemporalParams(Params):
    """
    MODIFICATION: Extended parameters for temporal model
    """
    def __init__(self, path):
        super().__init__(path)
        
        config = configparser.ConfigParser()
        config.read(self.path)
        params = config['DEFAULT']
        
        # Additional temporal parameters
        self.temporal_frames = params.getint('temporal_frames', 3)
        self.use_motion_module = params.getboolean('use_motion_module', True)
        
    def print(self):
        super().print()
        print('Temporal frames: {}'.format(self.temporal_frames))
        print('Use motion module: {}'.format(self.use_motion_module))
'''
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
