""" Config """
import argparse
import os


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


class Config(BaseConfig):
    def build_parser(self):
        parser = argparse.ArgumentParser("Config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--data', help='CIFAR10 / CIFAR100', default='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=100)
        parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
        parser.add_argument('--momentum', type=float, default=0.9)
        #parser.add_argument('--weight_decay', type=float, default=3e-4)
        #parser.add_argument('--grad_clip', type=float, default=5.,
        #                    help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--epochs', type=int, default=400,
                            help='# of training epochs (default: 400)')
        parser.add_argument('--channels', default="64,128,256,512,512")
        parser.add_argument('--global_drop_ratio', type=float, default=0.5,
                            help="ratio of global drop path (default: 0.5)")
        parser.add_argument('--p_local_drop', type=float, default=0.15,
                            help="local drop path probability (default: 0.15)")
        parser.add_argument('--dropout_probs', default="0.0, 0.1, 0.2, 0.3, 0.4")
        parser.add_argument('--blocks', type=int, default=5, help='default: 5')
        parser.add_argument('--columns', type=int, default=3, help='default: 3')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        #parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        #parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        #parser.add_argument('--resume')
        #parser.add_argument('--evaluate', action='store_true', default=False)
        parser.add_argument('--off-drops', action='store_true', default=False)
        parser.add_argument('--gap', action='store_true', default=False)

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data = self.data.lower().strip()
        self.data_path = './data/'
        self.channels = [int(c) for c in self.channels.split(',')]
        self.dropout_probs = [float(p) for p in self.dropout_probs.split(',')]
        assert len(self.channels) == len(self.dropout_probs) == self.blocks

        left = self.epochs // 2
        self.lr_milestone = [left]
        for i in range(3):
            left //= 2
            self.lr_milestone.append(self.lr_milestone[-1] + left)

        if self.off_drops:
            print("!!! Dropout and droppath are off !!!")
            for i in range(self.blocks):
                self.dropout_probs[i] = 0.
            self.p_local_drop = 0.
            self.global_drop_ratio = 0.
