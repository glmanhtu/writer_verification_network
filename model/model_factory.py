import os.path

import torch
from torchvision import models

from model import simsiam, triplet
from model.model_wrapper import ModelWrapper


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, working_dir, is_train, device, dropout=0.4):
        if args.network == 'simsiam':
            model = simsiam.SimSiam(models.__dict__[args.arch], dim=args.ss_dim, pred_dim=args.ss_pred_dim)
            if os.path.isfile(args.ss_pretrained):
                pretrained_model = torch.load(args.ss_pretrained, map_location='cpu')
                state_dict = pretrained_model['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder up to before the embedding layer
                    if k.startswith('module.encoder') or k.startswith('module.predictor'):
                        # remove prefix
                        state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]
                model.load_state_dict(pretrained_model['state_dict'])
        else:
            model = triplet.TripletNetwork(models.__dict__[args.arch])
        return ModelWrapper(args, working_dir, model, is_train, device)
