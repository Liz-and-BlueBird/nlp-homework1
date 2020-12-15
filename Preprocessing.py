import json
from tqdm import tqdm


def Make(fileName: str):
    """
    处理语料文件
    :param fileName: 要处理的语料文件
    :return:None
    """
    with open(fileName, 'r', encoding='utf-8') as in_set, open(fileName[fileName.find('-') + 1:-4] + ' set.txt', 'w',
                                                               encoding='utf-8') as out_set:
        lines = in_set.readlines()
        for line in tqdm(lines):
            words = line.strip('\n').split('\t')
            words_pos = list()
            for word in words:
                if word.find(']/') == -1:
                    if word.find('[') == -1:
                        words_pos.append(word)
                    else:
                        words_pos.append(word[1:])
                else:
                    words_pos.append(word[:word.find(']/')])
            out_set.write(' '.join(words_pos) + '\n')


def setMake():
    """
    根据语料制作训练集和测试集
    语料：199801-train.txt，199801-test.txt（由人民日报裁剪而来）
    :return:None
    """
    fileList = ['199801-train.txt', '199801-test.txt']
    for file in fileList:
        Make(file)


def dictionaryMake():
    """
    从词典文件中提取词典
    词典文件：CoreNatureDictionary.mini.txt（HanLP里的一个小型词典）
    :return:None
    """
    dic = list()
    # 提取字典为列表
    with open('CoreNatureDictionary.mini.txt', 'r', encoding='utf-8') as CoreNatureDictionary:
        lines = CoreNatureDictionary.readlines()
        for line in tqdm(lines):
            dic.append(line[0:line.find('\t')])
    # 保存为json格式的文件
    with open('dictionary.json', 'w', encoding='utf-8') as dictionaryFile:
        json.dump(dic, dictionaryFile, ensure_ascii=False)


def wordMatrixMake():
    """
    根据语料统计二元语法词频
    语料：train set.txt
    :return:None
    """
    dic = ['<BOS>', '<EOS>']
    sentences = list()
    with open('train set.txt', 'r', encoding='utf-8') as train:
        lines = train.readlines()
        for line in tqdm(lines):
            words = [word[:word.find('/')] for word in line.strip('\n').split(' ')]
            sentence = ['<BOS>']
            for word in words:
                if word not in dic:
                    dic.append(word)
                sentence.append(word)
            sentence.append('<EOS>')
            sentences.append(sentence)

    wordMatrix = dict({'<N>': {'<sum>': 0, '<maxLength>': 0}})
    for word in dic:
        behind = {'<sum>': 0}
        wordMatrix[word] = behind
    for sentence in tqdm(sentences):
        length = len(sentence)
        point = 0
        while point < length:
            wordMatrix['<N>']['<sum>'] += 1
            wordMatrix[sentence[point]]['<sum>'] += 1
            if len(sentence[point]) > wordMatrix['<N>']['<maxLength>']:
                wordMatrix['<N>']['<maxLength>'] = len(sentence[point])
            if point + 1 != length:
                if sentence[point + 1] not in wordMatrix[sentence[point]].keys():
                    wordMatrix[sentence[point]][sentence[point + 1]] = 1
                else:
                    wordMatrix[sentence[point]][sentence[point + 1]] += 1
            point += 1

    with open('wordMatrix.json', 'w', encoding='utf-8') as wordMatrixFile:
        json.dump(wordMatrix, wordMatrixFile, ensure_ascii=False)


def posProbabilityMake():
    """
    根据语料构建初始状态概率向量、状态转移矩阵和发射概率矩阵
    语料：train set.txt
    :return:None
    """
    # 初始状态概率向量
    start_probability = dict({'<sum>': 0})
    # 状态转移矩阵
    transition_probability = dict()
    # 发射概率矩阵
    emission_probability = dict()

    with open('train set.txt', 'r', encoding='utf-8') as train:
        lines = train.readlines()
        for line in tqdm(lines):
            words = line.strip('\n').split(' ')
            # 统计初始状态
            if words[0][words[0].find('/') + 1:] not in start_probability.keys():
                start_probability[words[0][words[0].find('/') + 1:]] = 1
            else:
                start_probability[words[0][words[0].find('/') + 1:]] += 1
            start_probability['<sum>'] += 1
            # 统计状态转移
            length = len(words)
            point = 0
            while point < length:
                frontPos = words[point][words[point].find('/') + 1:]
                if point < length - 1:
                    behindPos = words[point + 1][words[point + 1].find('/') + 1:]
                    if frontPos not in transition_probability.keys():
                        transition_probability[frontPos] = {'<sum>': 0}
                    if behindPos not in transition_probability[frontPos].keys():
                        transition_probability[frontPos][behindPos] = 1
                    else:
                        transition_probability[frontPos][behindPos] += 1
                    transition_probability[frontPos]['<sum>'] += 1
                # 统计发射概率
                word = words[point][:words[point].find('/')]
                if frontPos not in emission_probability.keys():
                    emission_probability[frontPos] = {'<sum>': 0}
                if word not in emission_probability[frontPos].keys():
                    emission_probability[frontPos][word] = 1
                else:
                    emission_probability[frontPos][word] += 1
                emission_probability[frontPos]['<sum>'] += 1
                point += 1

    with open('startProbability.json', 'w', encoding='utf-8') as startProbabilityFile:
        json.dump(start_probability, startProbabilityFile, ensure_ascii=False)
    with open('transitionProbability.json', 'w', encoding='utf-8') as transitionProbabilityFile:
        json.dump(transition_probability, transitionProbabilityFile, ensure_ascii=False)
    with open('emissionProbability.json', 'w', encoding='utf-8') as emissionProbabilityFile:
        json.dump(emission_probability, emissionProbabilityFile, ensure_ascii=False)


if __name__ == '__main__':
    setMake()
    dictionaryMake()
    wordMatrixMake()
    posProbabilityMake()
