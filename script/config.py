import argparse
import torch
import os

parser = argparse.ArgumentParser(description='HMPTGN')
# 1.dataset
parser.add_argument('--dataset', type=str, default='enron10', help='datasets')
parser.add_argument('--data_pt_path', type=str, default='', help='need to be modified')
parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
parser.add_argument('--nhid', type=int, default=16, help='dim of hidden embedding')
parser.add_argument('--nout', type=int, default=16, help='dim of output embedding')

# 2.experiments
parser.add_argument('--max_epoch', type=int, default=100, help='number of epochs to train.')
parser.add_argument('--testlength', type=int, default=145, help='length for test, default:50')
parser.add_argument('--vallength', type=int, default=0, help='length for validation, default:0')
#device change
parser.add_argument('--device', type=str, default='cuda', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--seed', type=int, default=1024, help='random seed')
parser.add_argument('--repeat', type=int, default=1, help='running times')
parser.add_argument('--use_rtt_feature', action='store_true', help='whether to use rtt features')
parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
parser.add_argument('--output_folder', type=str, default='output', help='need to be modified')
parser.add_argument('--use_htc', type=int, default=1, help='use htc or not, default: 1')
parser.add_argument('--use_hta', type=int, default=1, help='use hta or not, default: 1')
parser.add_argument('--debug_content', type=str, default='', help='debug_mode content')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')
parser.add_argument('--log_interval', type=int, default=1, help='log interval, default: 20,[20,40,...]')
parser.add_argument('--pre_defined_feature', default=None, help='pre-defined node feature')
parser.add_argument('--save_embeddings', type=int, default=0, help='save or not, default:0')
parser.add_argument('--debug_mode', type=int, default=0, help='debug_mode, 0: normal running; 1: debugging mode')
parser.add_argument('--min_epoch', type=int, default=50, help='min epoch')

# 3.models
parser.add_argument('--model', type=str, default='HMPTGN', help='model name')
parser.add_argument('--manifold', type=str, default='PoincareBall', help='Hyperbolic model')
parser.add_argument('--use_gru', type=bool, default=True, help='use gru or not')
parser.add_argument('--what_gru', type=str, default='hgru', help='use hgru or gru')
parser.add_argument('--use_hyperdecoder', type=bool, default=True, help='use hyperbolic decoder or not')
parser.add_argument('--EPS', type=float, default=1e-8, help='eps')
parser.add_argument('--nb_window', type=int, default=5, help='the length of window')
parser.add_argument('--bias', type=bool, default=True, help='use bias or not')
parser.add_argument('--trainable_feat', type=int, default=0,
                    help='using trainable feat or one-hot feat, default: none-trainable feat')
parser.add_argument('--dysat_window', type=int, default=-1,
                    help='temporal window in DySAT, -1 -> FULL')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (1 - keep probability).')
parser.add_argument('--heads', type=int, default=4, help='attention heads.')
parser.add_argument('--egcn_type', type=str, default='EGCNH', help='Type of EGCN: EGCNH or EGCNO')
parser.add_argument('--curvature', type=float, default=1.0, help='curvature value')
parser.add_argument('--fixed_curvature', type=int, default=1, help='fixed (1) curvature or not (0)')
parser.add_argument('--aggregation', type=str, default='deg', help='aggregation method: [deg, att]')
parser.add_argument('--spatial_dilated_factors', type=list, default=[1, 2],
                    help='dilated factor for dilated spatial convolution')
parser.add_argument('--casual_conv_depth', type=int, default=1, help='number of temporal casual convolution layers')
parser.add_argument('--casual_conv_kernel_size', type=int, default=2,
                    help='temporal casual convolution kernel size')

# 4. RTT Prediction (Multi-task Learning)
parser.add_argument('--enable_rtt_prediction', type=bool, default=False,
                    help='enable RTT prediction task alongside link prediction')
parser.add_argument('--rtt_loss_weight', type=float, default=10.0,
                    help='weight for RTT prediction loss in multi-task learning')
parser.add_argument('--rtt_hidden_dim', type=int, default=32,
                    help='hidden dimension for RTT regression head MLP')

args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda".format(args.device_id))
    print('using gpu:{} to train the model'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

model_folder = 'HERMIT' if args.model == 'HMPTGN' else args.model
args.output_folder = '../data/output/log/{}/{}/'.format(args.dataset, model_folder)
args.result_txt = '../data/output/results/{}_{}_result.txt'.format(args.dataset, model_folder)

# open debugging mode
if args.debug_mode == 1:
    print('start debugging mode!')
    folder = '../data/output/ablation_study/{}/'.format(args.debug_content)
    args.result_txt = folder + '{}_{}_result.txt'.format(args.dataset, args.model)
    if not os.path.isdir(folder):
        os.makedirs(folder)

# update the parameters for different datasets
if args.dataset in ['enron10', 'dblp']:
    args.testlength = 3  # using one-hot feature as input

if args.dataset in ['uci']:
    args.testlength = 11  # using trainable feature as input
    args.trainable_feat = 1

if args.dataset in ['mathoverflow']:
    args.testlength = 5  # using trainable feature as input
    args.trainable_feat = 1

if args.dataset in ['fbw']:  # length: 36
    args.testlength = 3
    args.trainable_feat = 1  # using trainable feature as input

if args.dataset in ['disease']:
    args.testlength = 3
    args.pre_defined_feature = 1  # using pre_defined_feature as input

if args.dataset in ['caida']:
    # args.testlength = 50
    # args.use_rtt_feature = True  # Disable this as we are using trainable features
    args.trainable_feat = 1      # Enable trainable features (128D)
    args.nfeat = 128             # Set feature dimension to 128
    args.lr = 0.0001             # Set learning rate to 0.0001 for fine-grained convergence
    args.EPS = 1e-5
    args.enable_rtt_prediction = True
    args.save_embeddings = 1     # Save embeddings for downstream RTT prediction
    args.max_epoch = 50          # As requested
    args.min_epoch = 20          # Lower min_epoch
    args.vallength = 70          # About 5% of training data for validation

if args.dataset in ['caida_rtt']:
    args.testlength = 145
    args.trainable_feat = 1


