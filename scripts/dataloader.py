from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from dataprocessor import *
from batch_generator import *
from conf.config import *
import random
class FewShotDataLoader:
    def __init__(self, config):
        self.config = config
        if hasattr(self.config,'bert_initialization_path'):
            print('loading specified tokenizer from ', self.config.bert_initialization_path, '...')
            self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_initialization_path)
            print('tokenizer loaded')
        else:
            print('loading tokenzier from transformers')
            print('if this hangs too long, pls restart program')
            self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_type)
            print('tokenizer loaded')
        self.dp = DataPreProcessor(if_del_serial=True)
    
    def transform_labels(self, Y):
        labels = list(set(Y))
        labels = list(sorted(labels))
        label_map = OrderedDict({y:i for i,y in enumerate(labels)})
        self.class_map = label_map
        newY = [label_map[y] for y in Y]
        return newY

    def get_dataloader(self, lines, trainer='protonet'):
        config = self.config
        rawX = [self.dp.proc_sent(x[0]) for x in lines]

        max_len = np.max([len(s) for s in rawX])
        
        print("max sentence length: ", max_len)
        labels = [x[1] for x in lines]

        Y = self.transform_labels(labels)
        print(self.class_map, len(self.class_map))
        X, testX, Y, testY = train_test_split(rawX, Y, test_size=self.test_num, shuffle=False)

        if trainer == 'protonet':
            config = ProtoNetConfig()
            dataloader = ProtoNetBatchGenerator(X=X, Y=Y, tokenizer=self.tokenizer,class_map=self.class_map, config=config)
            eval_dataloader = ProtoNetEvalBatchGenerator(X=X, Y=Y,class_map=self.class_map, tokenizer=self.tokenizer, testX=testX, testY=testY, config=config)

        # elif trainer == 'matching':
        #     config =
        #     dataloader = RandomBatchGenerator(X=X, Y=Y, tokenizer=self.tokenizer, n_supports=self.config.n_support, k=self.config.k, n_batch=self.config.nstep, n_pos=self.config.batch_pos, n_neg=self.config.batch_neg)
        #     eval_dataloader = ProtoNetEvalBatchGenerator(X=X, Y=Y,class_map=self.class_map, tokenizer=self.tokenizer, testX=testX, testY=testY, n_supports=self.config.eval_n_support, batch_size=self.config.eval_batch_size)
        elif trainer == 'bert':
            trainX = texts_to_indices(X, config.max_sent_len, self.tokenizer)
            train_dataset = TensorDataset(T.LongTensor(trainX), T.LongTensor(Y))
            dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            testX = texts_to_indices(testX, config.max_sent_len, self.tokenizer)
            test_dataset = TensorDataset(T.LongTensor(testX), T.LongTensor(testY))
            eval_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        elif trainer == 'shenyuan':
            dataloader = ShenyuanBatchGenerator(X=X, Y=Y, tokenizer=self.tokenizer, class_map=self.class_map, config=config)
            eval_dataloader = ShenyuanEvalBatchGenerator(X=X, Y=Y, testX=testX, tokenizer=self.tokenizer, testY=testY, class_map=self.class_map, config=config)
        else:
            raise NotImplementedError
        return dataloader, eval_dataloader



    def load_data(self):
        with open(self.config.train_path, encoding='utf-8') as f:
            train_lines = f.readlines()[:]
            random.shuffle(train_lines)
        with open(self.config.test_path, encoding='utf-8') as f:
            test_lines = f.readlines()[:]
        self.test_num = len(test_lines)
        lines = train_lines + test_lines
        lines = [x.strip().split('\t') for x in lines]

        return lines








if __name__ == '__main__':
    dl = DataLoader()
    dl.load_data()

