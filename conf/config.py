import torch as T
from string import punctuation

class BaseConfig:
    def __init__(self):
        '''data path'''
        self.test_path = 'data/test.tsv.formatted'
        self.train_path = 'data/train.tsv.formatted'
        self.test_path = 'data/test_intent.txt'
        self.train_path = 'data/train_intent.txt.small'
        self.test_path = 'data/ptest.txt'
        self.train_path = 'data/ptrain.txt'
        ''' log file dir'''
        self.log_dir = 'log/'

        '''maximum sentence length'''
        self.max_sent_len = 30
        ''' learning rate'''
        self.lr = 5e-5
        '''encoder output dimension'''
        self.hidden_dim = 768
        ''' cuda device number '''
        self.device = T.device("cuda:2" if T.cuda.is_available() else "cpu")
        ''' just in case, no gpu'''
        self.cpu = T.device("cpu")
        '''some punctuations'''
        self.puncs = '【】{}（）！-=——+@#￥%……&*（）——+、。，；‘“：、|、·~《》，。、？' + punctuation
        '''dropout rate for classifier'''
        self.dropout = 0.3
        '''path to save index-to-classname map (for inference)'''
        self.class_map_save_path = 'model/bert/class_map.json'
        self.class_map_load_path = 'model/bert/class_map.json'

        '''learning rate decay'''
        #self.lr_decay = 0.98
        '''learning rate decay interval'''
        #self.lr_decay_step_size = 5
        '''if you need to use other pretrained encoder in BERT file format, pls specify dir path, otherwise default bert-base-chinese will be used'''
    #        self.encoder_initialization_path = './model/ernie/'
        '''warm up ratio'''
        self.warmup = 0.05
        '''evaluation interval'''
        self.eval_interval = 3
        '''pretrained bert from 'transformers' package '''
        self.bert_type = 'bert-base-chinese'

class BertClassificationConfig(BaseConfig):
    def __init__(self):
        super(BertClassificationConfig, self).__init__()
        '''this should be a directory if using bert, ernie etc.'''
        self.encoder_save_path = './model/bert/'
        self.matcher_save_path = './model/bert/matcher.model'
        self.encoder_load_path = './model/bert/'
        self.matcher_load_path = './model/bert/matcher.model'
        ''' number of epoch '''
        self.epoch = 50
        '''number of classes'''
        self.n_classes = 86
        '''training batch size'''
        self.batch_size = 64
        '''total number of training step'''
        with open(self.train_path,'r') as f:
            nstep = len(f.readlines())//self.batch_size
        self.t_total = nstep * self.epoch
        print("TOTAL TRAINING STEP: ", self.t_total)


class ProtoNetConfig(BaseConfig):
    def __init__(self):
        super(ProtoNetConfig, self).__init__()
        '''encoder path'''
        self.encoder_save_path = './model/protonet/'
        self.encoder_load_path = './model/protonet/'
        self.center_load_path = './model/protonet/centers.pkl'
        self.center_save_path = './model/protonet/centers.pkl'
        ''' number of epoch '''
        self.epoch = 1000
        '''evaluation batch size'''
        self.eval_batch_size = 2
        '''number of supports to compute center of each class'''
        self.n_support = 5
        '''number of supports to compute center at eval time'''
        self.eval_n_support = 5
        '''number of training steps per epoch'''
        self.n_batch = 1
        ''' number of classes and number of sampled instances for training (k-way-n-shot)'''
        self.k = 30
        ''' number of samples per step (denoted as N), note that negative samples will automatically be computed, 
        leaving N positive samples and (k-1)*N negative samples'''
        self.shot = 2
        '''total number of training step'''
        self.t_total = self.n_batch * self.epoch
        print("TOTAL TRAINING STEP: ", self.t_total)
        '''distance epsilon'''
        self.dist_epsilon = 1e-1
        '''path to save centers corresponding to each class (as list)'''
        # self.lr_decay = 0.98
        # self.lr_decay_step_size = 5

class ShenyuanConfig(BaseConfig):
    def __init__(self):
        super(ShenyuanConfig, self).__init__()
        self.encoder_save_path = './model/shenyuan/'
        self.matcher_save_path = './model/shenyuan/matcher.model'
        self.encoder_load_path = './model/shenyuan/'
        self.matcher_load_path = './model/shenyuan/matcher.model'
        ''' number of epoch '''
        self.epoch = 100
        '''ratio of neg:pos training data sampling'''
        self.neg_ratio = 1
        '''training batch size'''
        self.batch_size = 64
        '''evaluation batch size'''
        self.eval_batch_size = 4
        '''number of supports to compute center of each class'''
        self.eval_n_support = 3
        '''number of training steps per epoch'''
        self.nstep = 2
        '''total number of training step'''
        self.t_total = self.batch_size*self.nstep
        '''path to save reference sentences for each class'''
        self.corpus_path = './model/shenyuan/data_dict.json'
        '''if you need to use other pretrained encoder in BERT file format, pls specify dir path, otherwise default bert-base-chinese will be used'''
#        self.encoder_initialization_path = './model/ernie/'
