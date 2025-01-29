import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.msa2net import Msa2Net_V11 as Model

from trainernoisybef import trainer_synapse
from fvcore.nn import FlopCountAnalysis
import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/hdd/deeplearning/frequencymiccai2024w/project_TransUNet/data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='/hdd/deeplearning/frequencymiccai2024w/project_TransUNet/data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--dstr_fast', type=bool,
                    default=False, help='SynapseDatasetFast: will load all data into RAM')
parser.add_argument('--en_lnum', type=int,
                    default=3, help='en_lnum: Laplacian layers (Pyramid) for the encoder')
parser.add_argument('--br_lnum', type=int,
                    default=3, help='br_lnum: Laplacian layers (Pyramid) for the bridge')
parser.add_argument('--de_lnum', type=int,
                    default=3, help='de_lnum: Laplacian layers (Pyramid) for the decoder')
parser.add_argument('--compact', type=bool,
                    default=False, help='compact with 3 blocks insted of 4 blocks')
parser.add_argument('--continue_tr', type=bool,
                    default=False, help='continue training from the last saved epoch')
parser.add_argument('--optimizer', type=str,
                    default='SGD', help='optimizer: [SGD, AdamW])')
parser.add_argument('--dice_loss_weight', type=float,
                    default=0.6, help="You need to determine <x> (default=0.6): => [loss = (1-x)*ce_loss + x*dice_loss]")
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, 
                    default='/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/reza/sina/miccai-v4-main-with-prompt-adaptive-0.25-noisy-main',help='output dir') 
parser.add_argument('--max_iterations', type=int,
                    default=90000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=20, help='batch_size per gpu')
parser.add_argument('--num_workers', type=int,
                    default=4, help='num_workers')
parser.add_argument('--eval_interval', type=int,
                    default=10, help='eval_interval')
parser.add_argument('--model_name', type=str,
                    default='WaveFormer_cross', help='model_name')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

parser.add_argument('--bridge_layers', type=int, default=1, help='number of bridge layers')

parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.05,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--add_noise', type=bool,
                    default=False, help='Permission for Adding Noises')
parser.add_argument('--train_std', type=float,
                    default=0.0, help='standard deviation of gaussian noise in train phase')
parser.add_argument('--train_prob', type = float,
                     default = 0.45, help = 'the occurance likelihood of noise in train phase')
parser.add_argument('--test_std', type=float,
                    default=0.05, help='standard deviation of gaussian noise in evaluation phase')
parser.add_argument('--test_prob', type = float,
                     default = 0.99, help = 'the occurance likelihood of noise in evaluation phase')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()

args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.compact:
        net = ModelCompact(
            num_classes=args.num_classes, 
            bridge_layers=args.bridge_layers,
            encoder_L_num=args.en_lnum, 
            bridge_L_num=args.br_lnum, 
            decoder_L_num=args.de_lnum
        ).cuda(0)
    else:
        net = Model(
            num_classes=args.num_classes
        ).cuda(0)
        
    input = torch.rand((1,3,224,224)).cuda(0)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     flops = FlopCountAnalysis(net, input)
#     model_flops = flops.total()
    print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
    
#     print(f"MAdds: {round(model_flops * 1e-9, 2)} G")
    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, args.output_dir)
