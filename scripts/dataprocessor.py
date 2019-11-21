import re
import string


class DataPreProcessor(object):
    """Runs end-to-end pre-processor."""

    def __init__(self, if_del_serial=False, clean_ratio=0.6, keep_punc=True):
        self.punc = "！？。％（）＋，－：∶；≤＜＝＞≥＠［］｛｜｝～、》「」『』【】〔〕〖〗《》〝〞–—‘’‛“”„‟‧﹏①②③④⑤⑥⑦⑧⑨⑩"
        self.punc += string.punctuation
        self.if_del_serial = if_del_serial
        self.clean_ratio = clean_ratio
        self.keep_punc = keep_punc

    def clear_rare_char(self, input_char):
        if u'\u4e00' <= input_char <= u'\u9fa5' \
                or input_char in self.punc \
                or u'\u0030' <= input_char <= u'\u0039' \
                or u'\u0041' <= input_char <= u'\u005A' \
                or u'\u0061' <= input_char <= u'\u007A':
            return input_char
        return ''

    @staticmethod
    def strQ2B(ustring):
        """把字符串全角转半角"""
        ss = []
        for s in ustring:
            rstring = ""
            for uchar in s:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248
                rstring += chr(inside_code)
            ss.append(rstring)
        return ''.join(ss)

    @staticmethod
    def del_serial(in_str):
        re_str = re.sub('^\d+(\.\d+)+', '', in_str)
        re_str = re.sub(
            '^[第(（]*[a-zA-Z①②③④⑤⑥⑦⑧⑨⑩一二三四五六七八九十\d]+[、.：:)）。条节章点](?!\d+)', '', re_str)
        match_flag = re.match('^\d+(个|年|月|日|岁|天|小时|点|张|寸|分|：\d+|:\d+)', re_str)
        if match_flag == None:
            re_str = re.sub('^\d+', '', re_str)
        re_str = re.sub('^[第\(（]*[①②③④⑤⑥⑦⑧⑨⑩][、.:：\)）。条节章点]*', '', re_str)
        # re_str = re.sub('\d\.\d\.\d', '', re_str)
        return re_str

    @staticmethod
    def format_sent(sent):
        sent = sent.replace(' ', '')
        sent = sent.replace('　', '')
        sent = sent.replace(' ', '')
        sent = sent.replace('\t', '')
        sent = sent.replace('工作描述:', '')
        sent = sent.replace('工作经历:', '')
        sent = sent.replace('工作范围:', '')
        sent = sent.replace('工作职责:', '')
        sent = sent.replace('工作业绩:', '')
        sent = sent.replace('主要业绩:', '')
        sent = sent.replace('主要职责:', '')
        sent = sent.replace('主要工作:', '')
        sent = sent.rstrip('\n\r')
        sent = sent.lstrip('+，。；:’‘“”：】、·）)？?>,.-—_=+!~`@#$%^&*')
        return sent

    @staticmethod
    def split_sent(sents, keep_punc=True):
        """split sentences(output has no ending punc)"""
        sents = sents.replace('[SEP]', '。')
        sents = re.split('(。|；|\?|？|\\n|\\\\n)', sents)
        if keep_punc:
            sents.append('')
            sents = ["".join(i) for i in zip(sents[0::2], sents[1::2])]
        sents = list(filter(lambda x: len(x) > 1, sents))
        return sents

    @classmethod
    def split_doc_file(cls, doc_path, del_empty_line=False):
        split_txt = []
        for line in open(doc_path, encoding='utf8'):
            # split_line = cls.split_sent(line.strip())
            # split_txt += split_line
            split_txt.append(line.strip())
        if del_empty_line:
            split_txt = list(filter(lambda x: len(x) > 1, split_txt))
        return split_txt

    @classmethod
    def split_doc_list(cls, doc_list, del_empty_line=False):
        split_txt = []
        for line in doc_list:
            split_line = cls.split_sent(line.strip())
            split_txt += split_line
        if del_empty_line:
            split_txt = list(filter(lambda x: len(x) > 0, split_txt))
        return split_txt

    def clean_sent(self, sent):
        cnt = 0
        clean_sent = ''
        for char in sent:
            if char in self.punc:
                cnt += 1
            clean_sent += self.clear_rare_char(char)
        if len(clean_sent) == 0:
            return ''
        if float(cnt) / float(len(clean_sent)) > self.clean_ratio:
            return ''
        return clean_sent

    def proc_sent(self, sent):
        """process single sentence"""
        sent = self.strQ2B(sent)
        sent = self.clean_sent(sent)
        sent = self.format_sent(sent)
        if len(sent) > 0:
            if self.if_del_serial:
                sent = self.format_sent(self.del_serial(sent))
        return sent

    def proc_sents(self, sents):
        """split and process sentence[s]"""
        sents = self.split_sent(sents, self.keep_punc)
        sents = [self.proc_sent(sent) for sent in sents]
        sents = list(filter(lambda x: len(x) > 0, sents))
        return sents


if __name__ == '__main__':
    # =====测试======
    t1 = "● 2017.7-2017.9，轮岗管理医疗部门（12人），打通医疗部门与业务部门的合作"
    t2 = "2:轮岗至百度直通车团队，管理百度直通车团队，对手机百度，百度地图，百度糯米产品的新客户开发以及老客户维护；"
    t3 = "工作描述：销售部门轮岗（1）完成每月公司的分销目标及其他KPI的达成"
    t4 = "*任职期间曾在公司资产管理部轮岗，熟悉如何利用资管工具为客户提供更为综合的金融服务"
    # print(DataPreProcessor.split_sent(test_str))

    proc = DataPreProcessor(if_del_serial=True)  # if_del_serial 是否删除开头序号
    print(proc.proc_sent(t3))
    # print(proc.proc_sents(test_str))
