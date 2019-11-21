from collections import OrderedDict
# from matcher import *
import torch as T
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from tqdm import tqdm
from torch import save, load
from utils import *
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam 
from sklearn.metrics import classification_report
import pickle
import json
import torch.nn as nn
from pytorch_pretrained_bert.optimization import BertAdam

#from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from matcher import init_bert_weights

def pool_from_encoder(encoding, pooling='cls'):
    if pooling == 'max':
        encoding = T.max(encoding[0], dim=1)[0]
    elif pooling == 'mean':
        encoding = T.mean(encoding[0], dim=1)
    elif pooling == 'cls':
        encoding = encoding[1]
    else:
        encoding = encoding[0]
    return encoding

class Trainer:
    def __init__(self, encoder, matcher, dl, config, loss_fn, *args, **kwargs):
        self.device = config.device
        self.config = config
        self.dl = dl
        self.matcher = matcher
        print(self.device)
        self.loss_fn = loss_fn
        self.encoder = encoder
        #init_bert_weights(self.encoder)
        self.encoder.to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.params = list(self.encoder.named_parameters())
        if self.matcher:
            self.matcher.to(self.device)
            self.params += list(self.matcher.named_parameters())
        param_optimizer = self.params
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.params = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        if self.config.warmup>0:
            self.optimizer = BertAdam(lr=self.config.lr, params=self.params, t_total=self.config.t_total, warmup=self.config.warmup)
        else:
            self.optimizer = BertAdam(lr=self.config.lr, params=self.params)
        #self.optimizer = Adam(lr=self.config.lr, params=self.params)
        self.steplr = None
        if hasattr(config, 'lr_decay'):
            if config.lr_decay:
                self.steplr = StepLR(self.optimizer, step_size=config.lr_decay_step_size, gamma=config.lr_decay)
        self.best_report = ''
        self.best_acc = 0.
        self.best_epoch = 0.

    def load_data(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass


    def save_model(self):
        encoder_path = self.config.encoder_save_path
        try:
            encoder_path = '/'.join(encoder_path.split('/')[:-1])
            self.encoder.save_pretrained(encoder_path)

        except:
            save(self.encoder, encoder_path)
        if self.matcher and self.matcher.parameters():
            matcher_path = self.config.matcher_save_path
            save(self.matcher, matcher_path)

    def load_model(self):
        encoder_path = self.config.encoder_load_path
        try:
            encoder_path = '/'.join(encoder_path.split('/')[:-1])
            self.encoder = BertModel.from_pretrained(encoder_path)
        except:
            self.encoder = load(encoder_path)
        try:
            matcher_path = self.config.matcher_load_path
            self.matcher = load(matcher_path)
        except:
            print('no need to load matcher or trained matcher does not exist')


class MatchingTrainer(Trainer):

    def load_data(self):
        self.lines = self.dl.load_data()
        self.training_data_generator, self.eval_data_generator = self.dl.get_dataloader(self.lines, trainer='matching')


    def train(self):
        for i in range(self.config.epoch):
            print("{}/{} epoch".format(i, self.config.epoch))
            epoch_auc = 0.
            self.encoder.train()
            self.matcher.train()
            for j, (pos_batch, neg_batch, supports_dict) in tqdm(
                    enumerate(self.training_data_generator.sample_batch()), total=len(self.training_data_generator)):
                labels = [1]*len(pos_batch)+[0]*len(neg_batch)
                batch_data = pos_batch+neg_batch
                queries = [x[1] for x in batch_data]
                domains = [x[0] for x in batch_data]
                encoded_center_dict, encoded_queries = self._encode_batch(queries, supports_dict)
                loss, auc = self._get_loss(encoded_center_dict, encoded_queries, labels, domains)
                self.optimizer.zero_grad()
                
                loss.backward()
                # nn.utils.clip_grad_norm_(self.params, 10)
                self.optimizer.step()
                epoch_auc += auc
            print("EPOCH AVG AUC: {}".format(epoch_auc / len(self.training_data_generator)))
            if i % self.config.eval_interval == 1:
                with T.no_grad():
                    eval_ACC, eval_report = self.evaluate()
                if eval_ACC > self.best_acc:
                    self.best_acc = eval_ACC
                    self.best_report = eval_report
                    self.best_epoch = i
                    self.save_model()
                    print(eval_report)
                    print('saved model...')

                print("current best ACC: ", self.best_acc)


    def evaluate(self):
        print("Evaluating...")
        self.encoder.eval()
        self.matcher.eval()
        supports_dict = self.eval_data_generator.get_support_dict()
        encoded_center_dict = OrderedDict()
        for cls in supports_dict:
            supports = supports_dict[cls]
            encoded_supports = self.encoder(T.LongTensor(supports).to(self.device))
            encoded_supports = pool_from_encoder(encoded_supports)
            encoded_center_dict[cls] = encoded_supports.mean(0).unsqueeze(0)
        all_preds = []
        all_Y = []
        for k, (batch_X, batch_Y) in enumerate(self.eval_data_generator.sample_batch()):
            preds = self._encode_eval_batch(batch_X, encoded_center_dict)
            all_preds += preds
            all_Y += batch_Y
        acc = calc_acc(all_preds, all_Y)
        print("EPOCH ACC : ", acc)
        report = classification_report(all_Y, all_preds)
        return acc, report

    def _encode_batch(self, queries, supports_dict):
        center_dict = OrderedDict()
        classes = supports_dict.keys()
        for c in classes:
            supports = supports_dict[c]
            encoded_supports = self.encoder(T.LongTensor(supports).to(self.device))
            encoded_supports = pool_from_encoder(encoded_supports)

            center_dict[c] = encoded_supports.mean(0).unsqueeze(0)
        queries = np.asarray(queries)
        
        encoded_queries = self.encoder(T.LongTensor(queries).to(self.device))
        encoded_queries = pool_from_encoder(encoded_queries)
        return center_dict, encoded_queries

    def _encode_eval_batch(self, queries, center_dict):
        encoded_queries = self.encoder(T.LongTensor(queries).to(self.device))
        encoded_queries = pool_from_encoder(encoded_queries)
        # encoded_queries = encoded_queries.tolist()
        repeated_queries = T.cat(len(center_dict)*[encoded_queries])
        classes = list(center_dict.keys())
        target_centers = T.cat(len(queries)*[center_dict[x] for x in classes])
        scores = self.matcher(repeated_queries, target_centers).view(len(queries), len(classes))
        preds = T.argmax(scores, dim=-1).squeeze().tolist()
        #print(preds, '\n', labels)
        
        return preds

    def _get_loss(self, encoded_center_dict, encoded_queries, labels, domains):
        # centers = list(encoded_center_dict.values())
        # centers = T.cat(centers, dim=0)
        centers = T.cat([encoded_center_dict[x] for x in domains], dim=0).to(self.device)
        scores = self.matcher(encoded_queries, centers).squeeze()
        # print(scores, truth)
        with T.no_grad():
            auc = roc_auc_score(labels, scores.detach().tolist())
        loss = self.loss_fn(scores, T.FloatTensor(labels).to(self.device))
        return loss, auc

class ProtoNetTrainer(Trainer):

    def load_data(self):
        self.lines = self.dl.load_data()
        self.training_data_generator, self.eval_data_generator = self.dl.get_dataloader(self.lines, trainer='protonet')
                                                                                        
    def train(self):
        for i in range(self.config.epoch):
            # if self.steplr:
            #     self.steplr.step()
            self.encoder.train()
            print("{}/{} epoch".format(i, self.config.epoch))
            print("TRAINING...")
            for j, (supports_dict, queries_dict) in enumerate(self.training_data_generator.sample_batch()):
                center_dict, q_dict = self._process_batch(supports_dict, queries_dict)
                centers = list(center_dict.values())
                centers = T.cat(centers, dim=0)
                loss = 0.
                for ind, cls in enumerate(supports_dict):
                    qs = q_dict[cls]
                    loss += self.loss_fn(qs, centers, ind)
                loss /= len(supports_dict)
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(self.params, 10)
                self.optimizer.step()
                print('TRAINING loss: ', loss.item())

            if i % self.config.eval_interval == 1:
                with T.no_grad():
                    eval_ACC, eval_report = self.evaluate()
                if eval_ACC > self.best_acc:
                    self.best_acc = eval_ACC
                    self.best_report = eval_report
                    self.best_epoch = i
                    self.save_model()
                    print(eval_report)
                    print('saved model...')

                print("current best ACC: ", self.best_acc)
        return self.best_acc, self.best_report, self.best_epoch

    def evaluate(self):
        print("EVALUATING...")
        self.encoder.eval()
        all_preds = []
        all_Y = []
        supports_dict = self.eval_data_generator.get_support_dict()
        encoded_center_dict = OrderedDict()
        order_keys = list(supports_dict.keys())
        self.classes = order_keys
        for cls in supports_dict:
            supports = supports_dict[cls]
            encoded_supports = self.encoder(T.LongTensor(supports).to(self.device))
            encoded_supports = pool_from_encoder(encoded_supports)
            encoded_center_dict[cls] = encoded_supports.mean(0).unsqueeze(0)
        centers = T.cat(list(encoded_center_dict.values()), dim=0)
        self.centers = centers.detach().tolist()
        total = len(self.eval_data_generator)
        for k, (batch_X, batch_Y) in tqdm(enumerate(self.eval_data_generator.sample_batch()), total=total):
            preds = self._process_eval_batch(centers, batch_X)
            preds = [order_keys[m] for m in preds]
            all_preds += preds
            all_Y += batch_Y
        accs = calc_acc(all_preds, all_Y)
        report = classification_report(all_Y, all_preds)
        print("EPOCH evaluation ACC: ", accs)
        return accs, report

    def _process_batch(self, supports_dict, queries_dict):
        center_dict = OrderedDict()
        q_dict = OrderedDict()
#        print(supports_dict.keys())
        for i in supports_dict:
            samples = supports_dict[i]
            assert samples, (i, samples)
            samples = self.encoder(T.LongTensor(samples).to(self.device))
            qsamples = queries_dict[i]
            qsamples = self.encoder(T.LongTensor(qsamples).to(self.device))
            samples = pool_from_encoder(samples)
            qsamples = pool_from_encoder(qsamples)
            center = samples.mean(0)
            center_dict[i] = center.unsqueeze(0)
            q_dict[i] = qsamples
        return center_dict, q_dict

    def _process_eval_batch(self, centers, queries):
        encoded_batchX = self.encoder(T.LongTensor(queries).to(self.device))
        encoded_batchX = pool_from_encoder(encoded_batchX)
        dists = T.cdist(encoded_batchX, centers)
        
        lbs = T.argmin(dists, dim=-1).detach().tolist()
        #print()
        #print("PREDS: ", lbs)
        
        return lbs

    def _get_loss(self):
        pass

    def save_model(self):
        super(ProtoNetTrainer, self).save_model()
        with open(self.config.center_save_path, 'wb') as f:
            pickle.dump(self.centers, f)
        with open(self.config.class_map_save_path, 'w', encoding='utf-8') as f:
            inverse_class_map = {y:x for x,y in self.dl.class_map.items()}
            json.dump(inverse_class_map, f)

class BertClassificationTrainer(Trainer):
    def load_data(self):
        self.lines = self.dl.load_data()
        self.training_data_generator, self.eval_data_generator = self.dl.get_dataloader(self.lines, trainer='bert')
        #init_bert_weights(self.matcher)

    def train(self):
        for i in range(self.config.epoch):
            self.encoder.train()
            self.matcher.train()
            print("{}/{} epoch".format(i, self.config.epoch))
            print("TRAINING...")
            total = len(self.training_data_generator)
            accs = 0.
            losses = 0.
            for j, (batchX, batchY) in tqdm(enumerate(self.training_data_generator), total=total):
                batchX = batchX.to(self.device)
                batchY = batchY.to(self.device)
                encoded_X = self.encoder(batchX)[1]
                logits = self.matcher(encoded_X)
   #             loss = self.loss_fn(logits, batchY)
                loss = F.cross_entropy(logits, batchY)
                preds = T.argmax(logits, dim=-1).detach().tolist()
                acc = calc_acc(preds,batchY.detach().tolist())
                accs += acc
                losses+=loss
                self.optimizer.zero_grad()
                loss.backward()
                #                nn.utils.clip_grad_norm_(self.params, 10)
                self.optimizer.step()
            print('TRAINING loss: ', losses.item()/total)
            print("Training acc: ", accs/total)
            if i % self.config.eval_interval == 1:
                with T.no_grad():
                    eval_ACC, eval_report = self.evaluate()
                if eval_ACC > self.best_acc:
                    self.best_acc = eval_ACC
                    self.best_report = eval_report
                    self.best_epoch = i
                    self.save_model()
                    print(eval_report)
                    print('saved model...')

                print("current best ACC: ", self.best_acc)
        return self.best_acc, self.best_report, self.best_epoch

    def evaluate(self):
        self.encoder.eval()
        self.matcher.eval()
        print("Evaluating...")
        total = len(self.eval_data_generator)
        all_preds = []
        all_Y = []
        for j, (batchX, batchY) in tqdm(enumerate(self.eval_data_generator), total=total):
            batchX = batchX.to(self.device)
            batchY = batchY.to(self.device)
            encoded_X = self.encoder(batchX)[1]
            logits = self.matcher(encoded_X)
            preds = T.argmax(logits, dim=-1).detach().tolist()
            all_preds += preds
            all_Y += batchY.detach().tolist()
        accs = calc_acc(all_preds, all_Y)
        report = classification_report(all_Y, all_preds)
        print("EVALUATION ACC: ", accs)
        return accs, report

    def save_model(self):
        super(BertClassificationTrainer, self).save_model()
        with open(self.config.class_map_save_path, 'w', encoding='utf-8') as f:
            inverse_class_map = {y:x for x,y in self.dl.class_map.items()}
            json.dump(inverse_class_map, f)


class ShenyuanTrainer(Trainer):
    # def __init__(self, *args, **kwargs):
    #     super(ShenyuanTrainer, self).__init__(*args, **kwargs)

    def load_data(self):
        self.lines = self.dl.load_data()
        self.training_data_generator, self.eval_data_generator = self.dl.get_dataloader(self.lines, trainer='shenyuan')

    def train(self):
        self.eval_batches = []
        self.eval_Y = []
        for i in range(self.config.epoch):
            #if self.steplr:
            #    self.steplr.step()
            self.encoder.train()
            self.matcher.train()
            print("{}/{} epoch".format(i, self.config.epoch))
            print("TRAINING...")
            total = len(self.training_data_generator)
            accs = 0.
            losses = 0.
            for j, (batchX, batchY) in tqdm(enumerate(self.training_data_generator.sample_batch()), total=total):
                batchX = T.LongTensor(batchX)
                batchY = T.FloatTensor(batchY)
                batchX = batchX.to(self.device)
                batchY = batchY.to(self.device)
                encoded_X = self.encoder(batchX)[1]
                logits = self.matcher(encoded_X).squeeze()
                loss = self.loss_fn(logits, batchY)
                # print(logits.size())
                # preds = T.argmax(logits, dim=-1).detach().tolist()
                # acc = calc_acc(preds, batchY.detach().tolist())
                # accs += acc
                losses += loss
                self.optimizer.zero_grad()
                loss.backward()
                #                nn.utils.clip_grad_norm_(self.params, 10)
                self.optimizer.step()
            print('TRAINING loss: ', losses.item() / total)
            # print("Training acc: ", accs / total)
            with T.no_grad():
                eval_acc, report = self.evaluate()
            if eval_acc > self.best_acc:
                self.best_acc = eval_acc
                self.best_report = report
                self.best_epoch = i
                self.save_model(self.config.corpus_path, self.config.class_map_save_path)
            print("Current Best ACC: ", self.best_acc)
        return self.best_acc, self.best_report, self.best_epoch

    def evaluate(self):
        self.encoder.eval()
        self.matcher.eval()
        print("Evaluating...")
        total = len(self.eval_data_generator)
        classes = self.eval_data_generator.classes
        save_flag = not self.eval_batches
        all_Y = []
        all_preds = []
        if save_flag:
            for j, (batchX, batchY) in tqdm(enumerate(self.eval_data_generator.sample_batch()), total=total):
                batch_size = len(batchX)//len(classes)
                batchX = T.LongTensor(batchX)
                #batchX = batchX.to(self.device)
                batchX = batchX.view(batch_size*len(classes)*self.config.eval_n_support, self.config.max_sent_len)
                self.eval_batches.append(batchX)
                self.eval_Y.append(batchY)
        for i, batchX in enumerate(self.eval_batches):
            batchY = self.eval_Y[i]
            if i % 10 ==0:
                print("{}/{}".format(i, len(self.eval_batches)))
            batch_size = batchX.size(0)//(len(classes)*self.config.eval_n_support)
            batchX = batchX.to(self.device)
            encoded_X = self.encoder(batchX)[1]
            logits = self.matcher(encoded_X).squeeze()
            logits = logits.view(batch_size, len(classes), self.config.eval_n_support)
            logits = T.mean(logits, dim=-1).squeeze()
            preds = T.argmax(logits, dim=-1).squeeze().detach().tolist()
            all_preds += preds
            all_Y += batchY
        accs = calc_acc(all_preds, all_Y)
        report = classification_report(all_Y, all_preds)
        print("EVALUATION ACC: ", accs / len(self.eval_Y))
        return accs, report

    def save_model(self):
        class_map_path = self.config.class_map_save_path
        super(ShenyuanTrainer, self).save_model()
        with open(class_map_path, 'w', encoding='utf-8') as f:
            inverse_class_map = {y:x for x,y in self.dl.class_map.items()}
            json.dump(inverse_class_map, f, ensure_ascii=False)

if __name__ == '__main__':
    matcher = None
    # matcher = RE2()
    # matcher = SimpleMatcher()
    matcher = CosMatcher()
    # matcher = DotMatcher()
    trainer = Trainer(matcher=matcher)
    print("(((((((((((((((((((((((((((")
    trainer.load_data()
    # trainer.train_classification()
    trainer.train()
