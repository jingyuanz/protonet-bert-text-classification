from sklearn.model_selection import train_test_split


def swap_sent_label(path_train, path_test):
    with open(path_test, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.strip().split('\t') for x in lines]
        lines = ['\t'.join([x[1],x[0]])+'\n' for x in lines]
    with open(path_test,'w', encoding='utf-8') as f:
        f.writelines(lines)
    with open(path_train, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.strip().split('\t') for x in lines]
        lines = ['\t'.join([x[1],x[0]])+'\n' for x in lines]
    with open(path_train,'w', encoding='utf-8') as f:
        f.writelines(lines)

def matching_to_classification(path_train, path_test):
    with open(path_test, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.strip().split('\t') for x in lines]
        lines = ['\t'.join(x[:2])+'\n' for x in lines if x[-1]=="Y"]
    with open(path_test+'.formatted','w', encoding='utf-8') as f:
        f.writelines(lines)
    with open(path_train, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.strip().split('\t') for x in lines]
        lines = ['\t'.join(x[:2])+'\n' for x in lines if x[2]=="Y"]
    with open(path_train+'.formatted','w', encoding='utf-8') as f:
        f.writelines(lines)

def matching_to_classification_with_other(path_train, path_test):
    with open(path_test, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.strip().split('\t') for x in lines]
        lines = ['\t'.join(x[:2])+'\n' if x[2]=='Y' else '\t'.join([x[0],'其他'])+'\n' for x in lines]
    with open(path_test+'.o','w', encoding='utf-8') as f:
        f.writelines(lines)
    with open(path_train, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.strip().split('\t') for x in lines]
        lines = ['\t'.join(x[:2])+'\n' if x[2]=='Y' else '\t'.join([x[0],'其他'])+'\n' for x in lines]
    with open(path_train+'.o','w', encoding='utf-8') as f:
        f.writelines(lines)

def split_corpus(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        X = [line[0] for line in lines]
        Y = [line[1] for line in lines]
        trainX, testX, trainY, testY = train_test_split(X, Y, stratify=Y, test_size=0.1)
        train_lines = ['\t'.join([x[0],x[1]])+'\n' for x in zip(trainX,trainY)]
        test_lines = ['\t'.join([x[0],x[1]])+'\n' for x in zip(testX, testY)]
    with open('data/ptrain.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open('data/ptest.txt', 'w', encoding='utf-8') as f:
        f.writelines(test_lines)

def down_sampling(fn, sampling_rate=0.01):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        X = [line[0] for line in lines]
        Y = [line[1] for line in lines]
        trainX, testX, trainY, testY = train_test_split(X, Y, stratify=Y, test_size=sampling_rate)
        from collections import defaultdict
        data_dict = defaultdict(list)
        for x, y in zip(testX, testY):
            data_dict[y].append(x)
        data_dict = {x:len(y) for x, y in data_dict.items()}
        print(data_dict)
        print(min(data_dict.values()))
        test_lines = ['\t'.join([x[0],x[1]])+'\n' for x in zip(testX,testY)]
        with open(fn+'.small', 'w', encoding='utf-8') as f:
            f.writelines(test_lines)

if __name__ == '__main__':
    # swap_sent_label('data/train_intent.txt','data/test_intent.txt')
    # matching_to_classification_with_other('data/train.tsv','data/test.tsv')
    # split_corpus('data/new_perfor1020')
    down_sampling('../data/train_intent.txt')
