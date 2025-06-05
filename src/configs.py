# -*- coding: utf-8 -*-
"""
@Time ： 2025/3/5 11:23
@Auth ： heshuai.sec@gmail.com
@File ：configs.py
"""
import argparse
import yaml

VALID_SCORES = {
    "1": 424,
    "1.333333333": 6,
    "1.666666667": 14,
    "2": 4701,
    "2.333333333": 69,
    "2.666666667": 76,
    "3": 2458,
    "3.333333333": 81,
    "3.666666667": 80,
    "4": 1144,
    "4.333333333": 96,
    "4.666666667": 93,
    "5": 758
}

VALID_SCORES = [float(k) for k in list(VALID_SCORES.keys())]
VALID_SCORES = [round(k, 5) for k in VALID_SCORES]


def get_args():
    parser = argparse.ArgumentParser()
    # Training settings
    # parser.add_argument('-gpu_n', default=4, type=int, help="how many gpu")
    # parser.add_argument('-g', '--gpuid', default=0, type=int, help="which gpu to use")
    parser.add_argument('--config', default=None, type=str, help='config file')
    # parser.add_argument("--local_rank", default=0, type=int, help='rank in current node')
    parser.add_argument("--local-rank", default=0, type=int, help='rank in current node')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
    # Experiment settings
    parser.add_argument("--experiment_name", required=False, default='exp_test', type=str,
                        help="name of this experiment")
    parser.add_argument("--load_checkpoint", default='', type=str, help="the name of the checkpoint to be loaded")
    parser.add_argument("--bytenas_path", type=str, default='./exp', help="path of bytenas")  # 存放实验相关内容
    parser.add_argument("--log_step", type=int, default=10, help="log step")
    # dataset related
    parser.add_argument("--data_path", type=str, default='xxx', help="path of data")  # 训练/测试数据存放路径
    parser.add_argument('--negative_num', type=int, default=9000)
    parser.add_argument('--hflip', type=float, default=0.5, help='horizontal flip')

    parser.add_argument('--iters', required=False, type=int, default=10000, metavar='N',
                        help='number of total iterations to run')
    parser.add_argument('--epoch', required=False, type=int, default=50, metavar='N',
                        help='number of total iterations to run')
    parser.add_argument('--val_split', type=int, default=1000)
    parser.add_argument('--batch_size', default=8, type=int, metavar='N', help='the batchsize for each gpu')
    parser.add_argument('--batch_size_warmup', default=16, type=int, metavar='N', help='the batchsize for each gpu')
    parser.add_argument('--batch_size_val', default=16, type=int, metavar='N', help='the batchsize for each gpu')

    parser.add_argument('--accumulate_step', default=16, type=int, metavar='N',
                        help='accumulate_step * batch_size = actual batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='base learning rate')
    parser.add_argument('--backbone_lr_decay', default=0.1, type=float, help='backbone learning rate decay')
    parser.add_argument('--min_lr', default=0.0, type=float, help='min learning rate')
    parser.add_argument('--val_iter', default=1, type=int, metavar='N', help='every epoch')
    parser.add_argument('--save_iter', default=1, type=int, metavar='N', help='every epoch')
    parser.add_argument('--weighted_loss', action='store_true', help='weighted loss for heatmap prevent output all 0')
    parser.add_argument('--mask_loss', type=str, default='mse')
    parser.add_argument('--score_loss', type=str, default='mse')
    parser.add_argument('--model', type=str, default='altclip', choices=['altclip', 'florence2', 'florence2-large-ft'])
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--mask_size', type=int, default=512)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'debug', 'eval'])
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_rank', default=8, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--score_head', type=str, default='linear')
    parser.add_argument('--mask_head', type=str, default='concat', choices=['concat', 'last', 'concat_cls'])
    parser.add_argument('--freeze_params', type=str, default='', help='freeze params, split by comma')
    parser.add_argument('--unfreeze_params', type=str, default='lora,heatmap_predictor,score_predictor',
                        help='freeze params, split by comma')
    parser.add_argument('--mask_loss_weight', type=float, default=1.0)
    parser.add_argument('--score_loss_weight', type=float, default=1.0)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)

    args = parser.parse_args()
    if args.lora_rank <= 0:
        if args.local_rank == 0:
            print('disable lora because lora_rank <= 0')
        args.use_lora = False
    else:
        if args.local_rank == 0: print('use lora, lora_rank:', args.lora_rank)
        args.use_lora = True
    if args.batch_size_warmup is None:
        args.batch_size_warmup = args.batch_size
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                if args.local_rank == 0: print('set', key, value)
                setattr(args, key, value)
    args.freeze_params = args.freeze_params.split(',')
    args.unfreeze_params = args.unfreeze_params.split(',')
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='train config file')
    parser.add_argument("--load_checkpoint", default='', type=str, help="the name of the checkpoint to be loaded")
    parser.add_argument('--submit_save_threshold', type=int, default=90)
    parser.add_argument('--data', type=str, default='test', choices=['val', 'test'])

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            if key not in args or getattr(args, key) is None:
                setattr(args, key, value)
    return args
