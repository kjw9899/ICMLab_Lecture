import argparse
import os

def parse_args() :
    parser = argparse.ArgumentParser(description='Jaewon')
    parser.add_argument('--feature_size', type=int, default=2048, help = 'size of feature (UCF:2048/XD:1024)')
    parser.add_argument('--rgb-list', default='data/ucf_tencrop_1d/ucf-i3d.list', help='list of rgb features ')
    parser.add_argument('--test-rgb-list', default='data/ucf_tencrop_1d/ucf-i3d-test.list', help='list of test rgb features ')
    parser.add_argument('--seg_length', type=int, default=32, help='default:32')

    parser.add_argument('--gpus', type=str, default='0', help='gpus')
    
    # model parameter
    parser.add_argument('--lr', type=float, default=0.00002, help='learning rates for steps(list form)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in a batch of data (default: 16)')
    parser.add_argument('--workers', default=4, help='number of workers in dataloader')
    parser.add_argument('--threshold', default=0.7, type=int, help='threshold for generating pseudo label')
    
    parser.add_argument('--model-name', default='jaewon', help='name to save model')
    parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
    parser.add_argument('--num-classes', type=int, default=1, help='number of class')
    parser.add_argument('--datasetname', default='UCF', help='dataset to train on (default:UCF/XD/UCF-bg-fg-sepa )')
    parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
    parser.add_argument('--max-epoch', type=int, default=1000, help='maximum iteration to train (default: 100)')


    args = parser.parse_args()
    os.environ['CUDA_VIDIBLE_DEVICES'] = args.gpus
    args.gpus = [i for i in range(len(args.gpus.split(' ')))]
    
    return args