from torchvision import models

from model import simsiam
from model.model_wrapper import ModelWrapper


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, working_dir, is_train, device, dropout=0.4):
        model = simsiam.SimSiam(models.__dict__[args.arch])
        return ModelWrapper(args, working_dir, model, is_train, device)
