import argparse
import os


class BaseOptions:
    def __init__(self, save_conf):
        self._parser = argparse.ArgumentParser()
        self._initialized = False
        self._opt = None
        self._save_conf = save_conf

    def is_train(self):
        raise NotImplementedError()

    def initialize(self):
        self._parser.add_argument('--tm_dataset_path', type=str, help='Path to the TM dataset')
        self._parser.add_argument('--letters', type=str, default=['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ',
                                                                  'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ϲ', 'τ', 'υ',
                                                                  'φ', 'χ', 'ψ', 'ω'], nargs="+")
        self._parser.add_argument('--triplet_files', type=str, default=[], nargs='+')
        self._parser.add_argument('--ss_dim', type=int, default=512, help='Simsiam dim')
        self._parser.add_argument('--ss_pred_dim', type=int, default=1024, help='Simsiam pred dim')
        self._parser.add_argument('--image_size', type=int, default=64, help='Input image size')
        self._parser.add_argument('--batch_size', type=int, default=196, help='Input batch size')
        self._parser.add_argument('--optimizer', type=str, default='Adam')
        self._parser.add_argument('--ss_pretrained', type=str, default='')
        self._parser.add_argument('--mode', type=str, default='Testing')
        self._parser.add_argument('--cuda', action='store_true', help="Whether to use GPU")
        self._parser.add_argument('--resume', action='store_true', help="Whether to use GPU")
        self._parser.add_argument('--arch', type=str, default='resnet18')
        self._parser.add_argument('--network', type=str, default='simsiam', choices=['simsiam', 'triplet'])
        self._parser.add_argument('--save_freq_iter', type=int, default=10,
                                  help='save the training losses to the summary writer every # iterations')
        self._parser.add_argument('--n_threads_train', default=8, type=int, help='# threads for loading data')
        self._parser.add_argument('--n_threads_test', default=8, type=int, help='# threads for loading data')
        self._parser.add_argument('--group', type=str, default='experiment',
                                  help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--name', type=str, default='experiment_1',
                                  help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--wb_mode', type=str, default='online', help='Wandb sync mode')
        self._parser.add_argument('--wb_entity', type=str, default='glmanhtu', help='Wandb entity name')
        self._parser.add_argument('--wb_project', type=str, default='writer-verification-network', help='Wandb project')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--lr', type=float, default=1e-4,
                                  help="The initial learning rate")
        self._parser.add_argument('--dropout', type=float, default=0.5, help="Default dropout")
        self._parser.add_argument('--lr_policy', type=str, default='none', choices=['step', 'none'])
        self._parser.add_argument('--lr_decay_epochs', type=int, default=100,
                                  help='reduce the lr to 0.1*lr for every # epochs')
        self._parser.add_argument('--n_epochs_per_eval', type=int, default=10,
                                  help='Run eval every n training epochs')
        self._parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
        self._parser.add_argument('--nepochs', type=int, default=800)
        self._parser.add_argument('--early_stop', type=int, default=50)
        self._parser.add_argument('--triplet_margin', type=float, default=1.)
        self._parser.add_argument('--n_samples_per_tm', type=int, default=8)

        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or test
        self._opt.is_train = self.is_train()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        if self._save_conf:
            self._save(args)

        return self._opt

    @staticmethod
    def _print(args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        if self.is_train and not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        else:
            assert os.path.exists(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
