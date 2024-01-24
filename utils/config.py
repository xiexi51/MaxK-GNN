""" Config class for search/augment """
import argparse
import os
from functools import partial


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser

class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text

class TrainConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("GIN/GCN/GraphSAGE Training config")
        parser.add_argument('--dataset',  default='yelp', choices = ['reddit', 'flickr', 'yelp', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'], #, 
                            help="Dataset name ('reddit', 'flickr', 'yelp', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins').") #, 'ogbn-proteins'
        parser.add_argument('--data_path', default='./data/', help='Dataset path')
        parser.add_argument('--model', default='sage', type=str, choices = ['sage', 'gcn', 'gin', 'gnn_res'],
                                    help="Model used in the training ('sage', 'gcn', 'gin', 'gnn_res')")
        parser.add_argument('--selfloop', default=False, action='store_true', help='add selfloop or not') #5e-4
        parser.add_argument('--epochs', type=int, default=1000, help='# of training epochs')
        parser.add_argument('--w_lr', type=float, default=0.01, help='lr for weights')
        parser.add_argument('--w_weight_decay', type=float, default=0, help='weight decay for weights') #5e-4
        parser.add_argument('--enable_lookahead', default=False, action='store_true', help='Using lookahead optimizer or not')
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help='Hidden dimension size')
        parser.add_argument('--hidden_layers', default=3, type=int,
                            help='Hidden dimension layers')
        parser.add_argument('--nonlinear', default='maxk', type=str, choices = ['maxk', 'relu'],
                            help='Nonlinear function used in the model')
        parser.add_argument('--maxk', default=32, type=int,
                            help='k value for maxk non-linearity')
        parser.add_argument('--dropout', type=float, default=0.5, help='feature dropout ratio') #5e-4
        parser.add_argument('--norm', default=False, action='store_true', help='add normalization layer or not') #5e-4
        parser.add_argument('--gpu', type=int, default=0, help='gpu device used in the experiment')
        parser.add_argument('--seed', type=int, default=97, help='random seed')
        parser.add_argument('-e', '--evaluate', default=None, type=str, metavar='PATH',
                            help='path to checkpoint (default: none), evaluate model')
        parser.add_argument('--path', default='./run/', type=str, metavar='PATH',
                            help='path to save the model and logging')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.plot_path = os.path.join(self.path, 'plots')