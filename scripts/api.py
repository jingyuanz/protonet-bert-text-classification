import sys
sys.path.append('./')
sys.path.append('../')
from transformers import BertModel
import argparse
from trainer import *
from conf.config import *
from dataloader import FewShotDataLoader
from matcher import *
import pickle
from predictor import *
import numpy as np
from datetime import datetime
import os
os.environ['KMP_WARNINGS'] = '0'
#np.random.seed(1)
#torch.manual_seed(1)
#torch.cuda.manual_seed_all(1)
#torch.backends.cudnn.deterministic = True
class API:
    def __init__(self, args):
        self.parse_args(args)

    def parse_args(self, args):
        self.model = args.model
        self.mode = args.mode
        if self.model == 'protonet':
            self.config = ProtoNetConfig()
            self.override_config(args, self.config)
            dl = FewShotDataLoader(self.config)
            encoder = self.get_encoder(self.config)
            trainer = ProtoNetTrainer(encoder, None, dl, self.config, EuclideanLoss(self.config.dist_epsilon))
            predictor = ProtoNetPredictor(self.config)

        elif self.model == 'bert':
            self.config = BertClassificationConfig()
            self.override_config(args, self.config)
            encoder = self.get_encoder(self.config)
            dl = FewShotDataLoader(self.config)
            trainer = BertClassificationTrainer(encoder, SimpleClassifier(self.config.n_classes, dropout=self.config.dropout), dl, self.config, CrossEntropyLoss())
            predictor = BertPredictor(self.config)

        elif self.model == 'shenyuan':
            self.config = ShenyuanConfig()
            self.override_config(args, self.config)
            dl = FewShotDataLoader(self.config)
            encoder = self.get_encoder(self.config)
            trainer = ShenyuanTrainer(encoder, SimpleRegressor(), dl, self.config, BCELoss())
            predictor = ShenyuanPredictor(self.config)
        else:
            raise NotImplementedError
        self.trainer = trainer
        self.predictor = predictor

    def override_config(self, args, config):
        #parsed_args = args.parse_args()
        for arg in vars(args):
            arg_val = getattr(args, arg)
            if arg_val:
                if arg == 'device':
                    arg_val = T.device("cuda:{}".format(arg_val) if T.cuda.is_available() else "cpu")

                setattr(config, arg, arg_val)
        if args.model == 'protonet':
            config.t_total = config.epoch
        elif args.model == 'bert':
            with open(config.train_path,'r') as f:
                nstep = len(f.readlines())//config.batch_size
            config.t_total = nstep * config.epoch
        else:
            config.t_total = config.nstep * config.epoch

    def get_encoder(self,config):
        if hasattr(config, "encoder_initialization_path"):
            fn = config.encoder_initialization_path
            print("loading encoder from ", fn, '...')
            encoder = BertModel.from_pretrained(config.encoder_initialization_path)
            print('encoder loaded.')
        else:
            t = config.bert_type
            print("loading pretrained encoder of type ", t, ' from transformers ...')
            print("if this hangs too long, pls restart the program")
            encoder = BertModel.from_pretrained(config.bert_type)
            print('encoder loaded.')
        return encoder

    def train(self, load=False):
        if load:
            self.trainer.load_model(self.config.encoder_load_path, self.config.matcher_load_path)
        self.trainer.load_data()
        best_acc, best_report, best_epoch = self.trainer.train()
        print(best_report)
        s = "Model{}, achieved best accuracy {} at epoch {}".format(self.model, best_acc, best_epoch)
        print(s)
        print("TRAINING FINISHED : ")
        log_fn = self.config.log_dir+"{}_{}.log".format(datetime.now().date(), self.model)
        with open(log_fn, 'a', encoding='utf-8') as f:
            s = '\n'.join([best_report, s])+'\n\n'
            f.write(s)
        print('log written to {}'.format(log_fn))

    def evaluate(self, load=False):
        if load:
            self.trainer.load_model(self.config.encoder_load_path, self.config.matcher_load_path)
        self.trainer.load_data()
        self.trainer.evaluate()

    def predict(self):
        self.predictor.load_model_components()
        while 1:
    #        try:
            sent = input("pls input sentence: ")
            self.predictor.predict(sent)
     #       except:
        #        print("sth goes wrong, try again")

    def exec(self):
        if self.mode == 'predict':
            self.predict()
        else:
            self.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='protonet', help='train with protonet or ordinary bert finetuning {protonet, bert, matching}', type=str)
    parser.add_argument("--mode", default='train', help="choose from {train, predict} mode", type=str)
    parser.add_argument("--epoch", help='num of epoch', type=int)
    parser.add_argument('--lr', help='learning rate', type=float)
    parser.add_argument('--warmup', help='portion of steps to warm up', type=float)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--train_path', help='training data file path', type=str)
    parser.add_argument('--test_path', help='testing data file path', type=str)
    parser.add_argument('--max_sent_len', help='max sentence length to pad', type=int)
    parser.add_argument('--encoder_load_path', help='path to load pretrained/continue-training encoder', type=str)
    parser.add_argument('--encoder_save_path', help='path to save encoder', type=str)
    parser.add_argument('--matcher_load_path', help='path to load pretrained/continue-training matcher', type=str)
    parser.add_argument('--matcher_save_path', help='path to save matcher', type=str)
    parser.add_argument('--shot', help='number of samples per class per epoch for protonet (K-way-N-shot tasks)', type=int)
    parser.add_argument('--k', help='number of classes per epoch for protonet (K-way-N-shot tasks)', type=int)
    parser.add_argument('--n_classes', help='total number of classes', type=int)
    parser.add_argument('--dist_epsilon', help='distance epsilon to be added to euclidean distance', type=float)
    parser.add_argument('--device', help='cuda device index or cpu to be used', type=int)
    parser.add_argument('--dropout', help='dropout ratio (not keep ratio)', type=float)
    parser.add_argument('--class_map_save_path', help='path to save class mapping json', type=str)
    parser.add_argument('--class_map_load_path', help='path to load class mapping json', type=str)
    parser.add_argument('--n_support', help='number of supporting samples per class per epoch', type=int)
    parser.add_argument('--eval_n_support', help='number of supporting samples per class per epoch', type=int)
    parser.add_argument('--eval_batch_size', help='eval batch_size, note this should be small for protonet if \
    total number of classes or eval n_support is big')
    parser.add_argument('--center_save_path', help='path to save centers', type=str)
    parser.add_argument('--center_load_path', help='path to load centers', type=str)

    args = parser.parse_args()
    api = API(args)
    api.exec()


