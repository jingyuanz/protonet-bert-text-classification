

def word_jaccard(seg1, seg2):
    a = list(set(seg1).intersection(set(seg2)))
    b = list(set(seg1).union(set(seg2)))
    return float(len(a) / len(b))


def char_jaccard(sen1, sen2):
    a = list(set(list(sen1)).intersection(set(list(sen2))))
    b = list(set(list(sen1)).union(set(list(sen2))))
    return float(len(a) / len(b))

if __name__ == '__main__':
    a = '你 是 谁啊'
    b = '我 是 什么人呢'
    print(word_jaccard(a.split(),b.split()))
    print(char_jaccard(a,b))