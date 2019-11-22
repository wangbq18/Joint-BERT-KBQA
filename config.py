import argparse
import os

# warmup_linear_constant
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='/root/pretrain_model_weights/torch/chinese/chinese_wwm_ext_pytorch', help="Directory containing the BERT model in PyTorch")

parser.add_argument('--clip_grad', default=2, type=int, help="")
parser.add_argument('--seed', default=42, type=int, help="random seed for initialization") # 8
parser.add_argument('--schedule', default='warmup_linear', help="schedule for optimizer")
parser.add_argument('--weight_decay', default=0.01, type=float, help="")
parser.add_argument('--warmup', default=0.1, type=float, help="")

parser.add_argument('--model_dir', default='experiments/baseline', help="model directory")
parser.add_argument('--epoch_num', default=6, type=int, help="num of epoch")
parser.add_argument('--nega_num', default=4, type=int, help="num of negative predicates")
parser.add_argument('--batch_size', default=32, type=int,  help="batch size")
parser.add_argument('--ner_max_len', default=32, type=int, help="max sequence length for ner task")
parser.add_argument('--re_max_len', default=64, type=int, help="max sequence length for re task")
parser.add_argument('--learning_rate', default=5e-5, type=float, help="learning rate")

parser.add_argument('--do_train_and_eval', action='store_true', help="do_train_and_eval")
parser.add_argument('--do_eval', action='store_true', help="do_eval")
parser.add_argument('--do_predict', action='store_true', help="do_predict")

args = parser.parse_args()

