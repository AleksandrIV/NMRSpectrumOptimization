import torch
import numpy as np
from model_single import ComplexModel
from collections import OrderedDict


def load_pth_model(filename):
    # for new pth format and r_global added
    state_dict=torch.load(filename)
    p=[((state_dict[f'models.{i}.v0']).double(), 
        (state_dict[f'models.{i}.J0']).double(), 
        (state_dict[f'models.{i}.w0']).double(), 
        (state_dict[f'models.{i}.r0']).double(), 
        (state_dict[f'models.{i}.proton_config']).double(), 
        (state_dict[f'models.{i}.freq']), 
        (state_dict[f'models.{i}.ppm0']).double()) for i in range(np.unique([int(s[7]) for s in torch.load(filename).keys() if s[:7]=='models.']).shape[0])]
    if list(torch.load(filename).keys())[0]=='r_global':
        pext=[]
        pext.append((state_dict[f'r_global']))
        pext.extend(p)
    else:
        pext=[torch.ones(np.unique([int(s[7]) for s in torch.load(filename).keys() if s[:7]=='models.']).shape[0])]
        pext.extend(p)
    return ComplexModel(pext)


def load_pth_file(filename):
    # for new pth format and r_global added
    state_dict=torch.load(filename)
    p=[((state_dict[f'models.{i}.v0']).double(), 
        (state_dict[f'models.{i}.J0']).double(), 
        (state_dict[f'models.{i}.w0']).double(), 
        (state_dict[f'models.{i}.r0']).double(), 
        (state_dict[f'models.{i}.proton_config']).double(), 
        (state_dict[f'models.{i}.freq']), 
        (state_dict[f'models.{i}.ppm0']).double()) for i in range(np.unique([int(s[7]) for s in torch.load(filename).keys() if s[:7]=='models.']).shape[0])]
    if list(torch.load(filename).keys())[0]=='r_global':
        pext=[]
        pext.append((state_dict[f'r_global']))
        pext.extend(p)
    else:
        pext=[torch.ones(np.unique([int(s[7]) for s in torch.load(filename).keys() if s[:7]=='models.']).shape[0])]
        pext.extend(p)
    return pext

def load_pth_from_state_dict(state_dict):
    # for new pth format and r_global added
    p=[((state_dict[f'models.{i}.v0']).double(), 
        (state_dict[f'models.{i}.J0']).double(), 
        (state_dict[f'models.{i}.w0']).double(), 
        (state_dict[f'models.{i}.r0']).double(), 
        (state_dict[f'models.{i}.proton_config']).double(), 
        (state_dict[f'models.{i}.freq']), 
        (state_dict[f'models.{i}.ppm0']).double()) for i in range(np.unique([int(s[7]) for s in state_dict.keys() if s[:7]=='models.']).shape[0])]
    if list(state_dict.keys())[0]=='r_global':
        pext=[]
        pext.append((state_dict[f'r_global']))
        pext.extend(p)
    else:
        pext=[torch.ones(np.unique([int(s[7]) for s in state_dict.keys() if s[:7]=='models.']).shape[0])]
        pext.extend(p)
    return pext


class StateDictGenerator:
    def __init__(self, num_models):
        self.num_models = num_models
        self.state_dict = OrderedDict()
        self.state_dict["r_global"] = torch.ones(num_models)  # Глобальный параметр

        # Создаем вложенный OrderedDict для моделей
        for i in range(num_models):
            self.state_dict[f"models.{i}.v0"] = None
            self.state_dict[f"models.{i}.J0"] = None
            self.state_dict[f"models.{i}.w0"] = None
            self.state_dict[f"models.{i}.r0"] = None
            self.state_dict[f"models.{i}.freq"] = None
            self.state_dict[f"models.{i}.ppm0"] = None
            
            

    def set_param(self, model_idx, param_name, value):
        """Устанавливает параметр для модели с индексом model_idx"""
        self.state_dict[f"models.{model_idx}.{param_name}"] = value
        
    def get_state_dict(self):
        """Возвращает OrderedDict со всеми параметрами
        то, что выдает model2.state_dict()"""
        return self.state_dict

    def get_pth(self):
        return load_pth_from_state_dict(self.state_dict)