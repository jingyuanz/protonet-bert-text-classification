from collections import defaultdict, OrderedDict
from random import choice
from utils import *
from sim import *
import itertools
import json
class BatchGenerator:
    def __init__(self, X, Y, tokenizer, class_map, config, indiced_data_dict=True):
        self.class_map = class_map
        self.inverse_map = {y: x for x, y in self.class_map.items()}
        self.classes = list(class_map.values())
        self.config = config
        self.tokenizer = tokenizer
        self.indiced_data_dict = indiced_data_dict
        self.X = X
        self.Y = Y
        self.data_dict = self._compute_data_dict(X, Y, index=indiced_data_dict)
        self.labelled_data_dict = self._compute_labelled_data_dict(X, Y, index=indiced_data_dict)
        self.count_dict = {x: len(y) for x, y in self.data_dict.items()}
        print(self.count_dict)

    def sample_batch(self):
        pass

    def sample_support_dict(self, classes, n_support):
        support_dict = defaultdict(list)
        for cls in classes:
            supports = sample(from_array=self.labelled_data_dict[cls], n_sample=n_support, replace=False)
            assert supports, (cls, len(self.labelled_data_dict[cls]))
            support_dict[cls] = supports
        return support_dict

    def __len__(self):
        pass

    def _compute_labelled_data_dict(self, X, Y, index=True):
        X = [self.inverse_map[y] + '=' + x for x, y in list(zip(X, Y))]
        if index:
            X = texts_to_indices(X, self.config.max_sent_len, self.tokenizer)
        data_dict = defaultdict(list)
        for i, (x, y) in enumerate(zip(X, Y)):
            data_dict[y].append(x)
        return data_dict

    def _compute_data_dict(self, X, Y, index=True):
        if index:
            X = texts_to_indices(X, self.config.max_sent_len, self.tokenizer)
        data_dict = defaultdict(list)
        for i, (x, y) in enumerate(zip(X, Y)):
            data_dict[y].append(x)
        return data_dict


''' need to pass in a dictionary with keys being classes(int), and values being positive samples of that class, data in format [sent, class] '''
'''
    data_dict: {class_nm:instances} dict
    n_pos, n_neg: num of pos/neg samples
    all_sents: in case not all sents are in data_dict (e.g. some sents belong to no classes)
'''


class RandomBatchGenerator(BatchGenerator):
    def __init__(self, *args, **kwargs):
        super(RandomBatchGenerator, self).__init__(*args, **kwargs)
        self.k = self.config.k
        self.n_batch = self.config.n_batch
        self.classes = list(self.data_dict.keys())
        self.n_pos = self.config.n_pos
        self.neg_ratio = self.config.neg_ratio
        self.n_support = self.config.n_support

    def sample_batch(self):
        for i in range(self.n_batch):
            chosen_classes = sample(self.classes, self.k, replace=False)
            batch_pos_data = []
            batch_neg_data = []
            for cls in chosen_classes:
                cls_pos_data = sample(self.data_dict[cls], self.n_pos, replace=True)
                corr_classes = list(set(chosen_classes).difference([cls])) * self.n_pos
                tmp = list(itertools.chain.from_iterable(itertools.repeat(x, self.k - 1) for x in cls_pos_data))
                cls_pos_data = list(zip([cls] * self.n_pos, cls_pos_data))
                cls_neg_data = list(zip(corr_classes, tmp))
                cls_neg_data = sample(cls_neg_data, self.neg_ratio * self.n_pos, replace=False)
                batch_pos_data += cls_pos_data
                batch_neg_data += cls_neg_data
            supports_dict = self.sample_support_dict(chosen_classes, self.n_support)
            if batch_pos_data:
                yield batch_pos_data, batch_neg_data, supports_dict

    def __len__(self):
        return self.n_batch


class ProtoNetBatchGenerator(BatchGenerator):
    def __init__(self, *args, **kwargs):
        super(ProtoNetBatchGenerator, self).__init__(*args, **kwargs)
        self.k = self.config.k
        self.n_batch = self.config.n_batch
        self.shot = self.config.shot
        self.n_support = self.config.n_support
        self.class_weight = []
        for cls in self.classes:
            self.class_weight.append(self.count_dict[cls])
        self.class_weight = np.asarray(self.class_weight)
        self.class_weight = (self.class_weight / self.class_weight.sum(0)).tolist()
        print(self.class_weight)

    def sample_batch(self):
        for i in range(self.n_batch):
            query_dict = OrderedDict()
            if self.k < len(self.classes):
                p = self.class_weight
            else:
                p = None
            chosen_classes = sample(self.classes, self.k, replace=False, p=p)
            assert len(set(chosen_classes).difference(self.classes)) == 0
            for cls in chosen_classes:
                cls_pos_data = sample(self.data_dict[cls], self.shot, replace=False)
                query_dict[cls] = cls_pos_data
            # all_y = list(set([x[0] for x in batch_neg_data+batch_pos_data]))
            supports_dict = self.sample_support_dict(chosen_classes, self.n_support)
            assert query_dict.keys() == supports_dict.keys()
            if query_dict and supports_dict:
                yield supports_dict, query_dict

    def __len__(self):
        return self.n_batch


class ProtoNetEvalBatchGenerator(BatchGenerator):
    def __init__(self, testX, testY, *args, **kwargs):
        super(ProtoNetEvalBatchGenerator, self).__init__(*args, **kwargs)
        self.testX = texts_to_indices(testX, self.config.max_sent_len, self.tokenizer)
        self.testY = testY
        print("Length of Test Data", len(testY))
        self.batch_size = self.config.eval_batch_size
        if len(self.testY)%self.batch_size==0:
            self.n_batch = len(self.testY) // self.batch_size
        else:
            self.n_batch = len(self.testY) // self.batch_size + 1

    def __len__(self):
        return self.n_batch

    def get_support_dict(self):
        return self.sample_support_dict(self.classes, self.config.eval_n_support)

    def sample_batch(self):
        for i in range(self.n_batch):
            head = i * self.batch_size
            tail = min(len(self.testY), head + self.batch_size)
            batch_testX = self.testX[head:tail]
            batch_testY = self.testY[head:tail]
            if batch_testX:
                yield batch_testX, batch_testY


class ShenyuanBatchGenerator(BatchGenerator):
    def __init__(self, *args, **kwargs):
        super(ShenyuanBatchGenerator, self).__init__(indiced_data_dict=False, *args, **kwargs)
        self.batch_size = self.config.batch_size
        if len(self.X) % self.batch_size == 0:
            self.n_batch = len(self.X) // self.batch_size
        else:
            self.n_batch = len(self.X) // self.batch_size + 1
        with open(self.config.corpus_path, 'w', encoding='utf-8') as f:
            json.dump(self.data_dict, f, ensure_ascii=False)

    def __len__(self):
        return self.n_batch

    def sample_batch(self):
        for i in range(self.n_batch):
            head = i * self.batch_size
            tail = min(len(self.X), head + self.batch_size)
            batchX = self.X[head:tail]
            batchY = self.Y[head:tail]
            batch_paired_X = []
            batch_match_Y = []
            for x, y in zip(batchX, batchY):
                other_classes = list(set(self.classes).difference([y]))
                random_class = choice(other_classes)
                sampled_neg = sample(self.data_dict[random_class], 1)[0]
                sampled_pos = sample(self.data_dict[y], 1)[0]
                pos = self._join(x, sampled_pos)
                neg = self._join(x, sampled_neg)
                batch_paired_X.append(pos)
                batch_paired_X.append(neg)
                batch_match_Y += [1, 0]
            if batch_match_Y:
                batch_paired_X = texts_to_indices(batch_paired_X, max_len=self.config.max_sent_len,
                                                  tokenizer=self.tokenizer)
                yield batch_paired_X, batch_match_Y

    def _join(self, a, b):
        cut_len_a = min(self.config.max_sent_len // 2, len(a))
        cut_len_b = min(self.config.max_sent_len // 2, len(b))
        return a[:cut_len_a] + '[SEP]' + b[:cut_len_b]


class ShenyuanEvalBatchGenerator(BatchGenerator):
    def __init__(self, testX, testY, *args, **kwargs):
        super(ShenyuanEvalBatchGenerator, self).__init__(indiced_data_dict=False, *args, **kwargs)
        self.batch_size = self.config.eval_batch_size
        self.testX = testX
        self.testY = testY
        self.n_support = self.config.eval_n_support
        if len(self.testX) % self.batch_size == 0:
            self.n_batch = len(self.testX) // self.batch_size
        else:
            self.n_batch = len(self.testX) // self.batch_size + 1
            

    def sample_batch(self):
        for i in range(self.n_batch):
            head = i * self.batch_size
            tail = min(len(self.testY), head + self.batch_size)
            batchX = self.testX[head:tail]
            batchY = self.testY[head:tail]
            batch_paired_X = []
            batch_labels = []
            for x, y in zip(batchX, batchY):
                supports = self._sample_topN_supports(self.n_support, x)
                pairs = [[self._join(x, b) for b in ss] for ss in supports]
                batch_paired_X += [texts_to_indices(p, max_len=self.config.max_sent_len, tokenizer=self.tokenizer) for p
                                   in pairs]
                batch_labels.append(y)
            if batch_labels:
                yield batch_paired_X, batch_labels

    def __len__(self):
        return self.n_batch

    def _join(self, a, b):
        cut_len_a = min(self.config.max_sent_len // 2, len(a))
        cut_len_b = min(self.config.max_sent_len // 2, len(b))
        return a[:cut_len_a] + '[SEP]' + b[:cut_len_b]

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
