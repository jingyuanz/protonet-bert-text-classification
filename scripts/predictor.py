import torch as T
from transformers import *
from utils import *
from dataprocessor import DataPreProcessor
import pickle
import json
from torch import save, load

class Predictor:
    def __init__(self, config):
        self.config = config
        self.dp = DataPreProcessor(if_del_serial=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def load_model_components(self):
        encoder_path = self.config.encoder_load_path
        self.encoder = BertModel.from_pretrained(encoder_path)
        self.encoder.eval()
        class_map_path = self.config.class_map_load_path
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)


    def _preprocess(self, sent):
        sent = self.dp.proc_sent(sent)
        sent = self.tokenizer.encode(sent)
        sent = pad_sequences([sent], maxlen=self.config.max_sent_len, padding='post', truncating='post').tolist()
        return sent

    def _preprocess_pair(self, a, b):
        senta = self.dp.proc_sent(a)
        sentb = self.dp.proc_sent(b)
        senta = senta[:min([self.config.max_sent_len, len(senta)])]
        sentb = sentb[:min([self.config.max_sent_len, len(sentb)])]
        sent = self.tokenizer.encode(senta, text_pair=sentb, add_special_tokens=True)
        return sent

    def predict(self, sent):
        pass

class ShenyuanPredictor(Predictor):
    def load_model_components(self):
        super(ShenyuanPredictor, self).load_model_components()
        corpus_path = self.config.corpus_path
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.data_dict = json.load(f)
        self.classes = list(self.data_dict.keys())
        self.encoder.to(self.config.device)
        self.matcher = load(self.config.matcher_load_path)
        self.matcher.to(self.config.device)
        self.matcher.eval()

    def predict(self,sent):
        topns = self._sample_topN_supports(self.config.eval_n_support, sent)
        paired_topns = []
        for i, topn_cls in enumerate(topns):
            topn_cls = [self._preprocess_pair(sent,self.class_map[str(i)]+a) for a in topn_cls]
            topn_cls = pad_sequences(topn_cls, maxlen=self.config.max_sent_len, padding='post', truncating='post').tolist()
            paired_topns.append(topn_cls)
        topn = T.LongTensor(paired_topns).to(self.config.device)
        topn = topn.view(len(self.class_map)*self.config.eval_n_support, self.config.max_sent_len)
        topn = self.encoder(topn)[1]
        logits = self.matcher(topn).squeeze()
        logits = logits.view(len(self.class_map), self.config.eval_n_support)
        logits = T.mean(logits, dim=-1).squeeze()
        pred = T.argmax(logits, dim=-1).squeeze().item()
        max_score = T.max(logits,dim=-1)[0].squeeze().item()
        if max_score < .5:
            print("OTHERS")
        else:
            print(self.class_map[str(pred)])



         
    def _sample_topN_supports(self, n, x):
        topn_all_class = []
        for cls in self.classes:
            candidates = self.data_dict[cls]
            if n > len(candidates):
                sims = [(y, char_jaccard(x, y)) for y in candidates]
                topn = list(sorted(sims, key=lambda x: x[1]))[-n:]
                topn = [x[0] for x in topn]
            else:
                topn = sample(candidates, n, replace=True)
            topn_all_class.append(topn)
        return topn_all_class

class ProtoNetPredictor(Predictor):

    def load_model_components(self):
        super(ProtoNetPredictor, self).load_model_components()
        center_path = self.config.center_load_path
        with open(center_path, 'rb') as f:
            self.centers = pickle.load(f)
        self.centers = T.FloatTensor(self.centers)

    def predict(self, sent):
        sent = self._preprocess(sent)
        sent = T.LongTensor(sent)
        sent = self.encoder(sent)[1]        
        dists = T.cdist(sent,self.centers).squeeze()
        pred = dists.argmin(dim=-1).item()
        raw_pred = self.class_map[str(pred)]
        print(raw_pred)

class BertPredictor(Predictor):
    def load_model_components(self):
        super(BertPredictor,self).load_model_components()
        matcher_path = self.config.matcher_path
        self.matcher = load(matcher_path)
        self.matcher.to(self.config.cpu)
        self.matcher.eval()

    def predict(self, sent):
        sent = self._preprocess(sent)
        sent = T.LongTensor(sent)
        sent = self.encoder(sent)[1]
        pred = T.argmax(self.matcher(sent),dim=-1).item()
        raw_pred = self.class_map[str(pred)]
        print(raw_pred)

        






