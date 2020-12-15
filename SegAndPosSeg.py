import json
import math


# --------------------------------------------------------
# 字典分词

def loadDictionary():
    """
    加载词典
    :return:词典
    """
    with open('dictionary.json', 'r', encoding='utf-8') as dictionaryFile:
        rtv = json.load(dictionaryFile)
    return rtv


# 字典
dic = loadDictionary()


def getMaxLength():
    """
    获取词典中最长词的长度
    :return: 最长词的长度
    """
    rtv = len(max(dic, key=len))
    return rtv


# 词典中最长词的长度
maxLength = getMaxLength()


def forwardSegment(sentence: str):
    """
    正向最长匹配
    :param sentence: 待切分的语句
    :return: 正向最长匹配的切分结果
    """
    rtv = list()
    while sentence != "":
        point = min(maxLength, len(sentence))
        while point > 1:
            if sentence[:point] in dic:
                break
            else:
                point -= 1
        rtv.append(sentence[:point])
        sentence = sentence[point:]
    return rtv


def backwardSegment(sentence: str):
    """
    逆向最长匹配
    :param sentence:待切分的语句
    :return: 逆向最长匹配的切分结果
    """
    rtv = list()
    while sentence != "":
        point = min(maxLength, len(sentence))
        while point > 1:
            if sentence[-point:] in dic:
                break
            else:
                point -= 1
        rtv.append(sentence[-point:])
        sentence = sentence[:-point]
    rtv.reverse()
    return rtv


def singleWordCount(sentence: list):
    """
    统计切分好的语句中单字的个数
    :param sentence: 切分好的语句
    :return: 单字个数
    """
    count = 0
    for word in sentence:
        if len(word) == 1:
            count += 1
    return count


def bidirectionalSegment(sentence: str):
    """
    双向最长匹配
    :param sentence:待切分的语句
    :return: 双向最长匹配的切分结果
    """
    forwardRtv = forwardSegment(sentence)
    backwardRtv = backwardSegment(sentence)
    forwardRtvWords = len(forwardRtv)
    backwardRtvWords = len(backwardRtv)
    if forwardRtvWords == backwardRtvWords:
        return forwardRtv if singleWordCount(forwardRtv) < singleWordCount(backwardRtv) else backwardRtv
    else:
        return forwardRtv if forwardRtvWords < backwardRtvWords else backwardRtv


# --------------------------------------------------------
# 二元语法分词


def loadWordMatrix():
    """
    加载词频矩阵
    :return: 词频矩阵
    """
    with open('wordMatrix.json', 'r', encoding='utf-8') as wordMatrixFile:
        rtv = json.load(wordMatrixFile)
    return rtv


# 词频矩阵
wordMatrix = loadWordMatrix()


def wordNetGet(sentence: str):
    """
    由语句生成词网
    :param sentence: 待切分语句
    :return: 词网
    """
    wordNet = list([['<BOS>']])
    length = len(sentence)
    point = 0
    while point < length:
        words = list()
        offset = point
        while offset < length + 1 and (offset - point + 1) < wordMatrix['<N>']['<maxLength>']:
            if sentence[point:offset] in wordMatrix.keys():
                words.append(sentence[point:offset])
            offset += 1
        if len(words) == 0:
            words.append(sentence[point])
        wordNet.append(words)
        point += 1
    wordNet.append(list(['<EOS>']))
    return wordNet


def possibility(backNode: str, node: str):
    """
    计算经前一个结点到当前节点路径的概率
    :param backNode: 前一个结点
    :param node: 当前节点
    :return: 到当前节点路径概率的负对数
    """
    m = 0.99
    if backNode not in wordMatrix.keys():
        p1 = 0
        if node not in wordMatrix.keys():
            p2 = 0
        else:
            p2 = wordMatrix[node]['<sum>'] / wordMatrix['<N>']['<sum>']
    else:
        if node not in wordMatrix.keys():
            p1 = 0
            p2 = 0
        else:
            if node not in wordMatrix[backNode].keys():
                p1 = 0
            else:
                p1 = wordMatrix[backNode][node] / wordMatrix[backNode]['<sum>']
            p2 = wordMatrix[node]['<sum>'] / wordMatrix['<N>']['<sum>']
    return -math.log(m * p1 + (1 - m) * (m * p2 + (1 - m) * (1 / wordMatrix['<N>']['<sum>'])))


def listInit(wordNet: list):
    """
    生成用于维特比算法的图
    :param wordNet: 语句词网
    :return: 用于维特比算法的有向无环图
    """
    rtv = list()
    for words in wordNet:
        if len(words) == 1 and words[0] == '<BOS>':
            rtv.append([])
        else:
            wordsInfo = list()
            for _ in words:
                wordsInfo.append([0, 0, float('inf')])
            rtv.append(wordsInfo)
    return rtv


def viterbi(wordNet: list):
    """
    利用维特比算法求解最短路径
    :param wordNet: 语句词网
    :return: 作为最短路径的分词结果
    """
    length = len(wordNet)
    if length == 2:
        return list()
    path = listInit(wordNet)
    point = 0
    while point < length - 1:
        if point == 0:
            nextPoint = point + 1
            count = len(wordNet[nextPoint])
            i = 0
            while i < count:
                path[nextPoint][i][0] = point
                path[nextPoint][i][1] = 0
                path[nextPoint][i][2] = possibility('<BOS>', wordNet[nextPoint][i])
                i += 1
        else:
            for node in wordNet[point]:
                nextPoint = point + len(node)
                count = len(wordNet[nextPoint])
                i = 0
                while i < count:
                    pos = possibility(node, wordNet[nextPoint][i])
                    if pos + path[point][wordNet[point].index(node)][2] < path[nextPoint][i][2]:
                        path[nextPoint][i][0] = point
                        path[nextPoint][i][1] = wordNet[point].index(node)
                        path[nextPoint][i][2] = pos + path[point][wordNet[point].index(node)][2]
                    i += 1
        point += 1
    rtv = list()
    point = path[-1][0][0:-1]
    while point[0] > 0:
        rtv.append(wordNet[point[0]][point[1]])
        point = path[point[0]][point[1]][0:-1]
    rtv.reverse()
    return rtv


def twoGramSegment(sentence: str):
    """
    二元语法分词
    :param sentence: 待切分的语句
    :return: 二元语法分词结果
    """
    wordNet = wordNetGet(sentence)
    rtv = viterbi(wordNet)
    return rtv


# --------------------------------------------------------
# HMM序列标注词性

def loadStartProbability():
    """
    加载初始状态概率向量
    :return: 初始状态概率向量
    """
    with open('startProbability.json', 'r', encoding='utf-8') as startProbabilityFile:
        start_probability_tmp = json.load(startProbabilityFile)
    return start_probability_tmp


def loadTransitionProbability():
    """
    加载状态转移矩阵
    :return: 状态转移矩阵
    """
    with open('transitionProbability.json', 'r', encoding='utf-8') as transitionProbabilityFile:
        transition_probability_tmp = json.load(transitionProbabilityFile)
    return transition_probability_tmp


def loadEmissionProbability():
    """
    加载发射概率矩阵
    :return:发射概率矩阵
    """
    with open('emissionProbability.json', 'r', encoding='utf-8') as emissionProbabilityFile:
        emission_probability_tmp = json.load(emissionProbabilityFile)
    return emission_probability_tmp


# 初始状态概率向量
start_probability = loadStartProbability()
# 状态转移矩阵
transition_probability = loadTransitionProbability()
# 发射概率矩阵
emission_probability = loadEmissionProbability()


def possHMM(front_pos: str, behind_pos: str, word: str):
    """
    计算词性节点概率
    :param front_pos: 前一词性
    :param behind_pos: 当前词性
    :param word: 当前词
    :return: 标注为该词性的概率
    """
    m = 0.99
    if front_pos == '<EOS>':
        if behind_pos not in start_probability.keys():
            p_t = math.log((1 - m) * (1 / start_probability['<sum>']))
        else:
            p_t = math.log(m * (start_probability[behind_pos] / start_probability['<sum>']) + (1 - m) * (
                    1 / start_probability['<sum>']))
    else:
        if behind_pos not in transition_probability[front_pos].keys():
            p_t = math.log((1 - m) * (1 / transition_probability[front_pos]['<sum>']))
        else:
            p_t = math.log(
                m * (transition_probability[front_pos][behind_pos] / transition_probability[front_pos]['<sum>']) + (
                        1 - m) * (1 / transition_probability[front_pos]['<sum>']))
    if word not in emission_probability[behind_pos].keys():
        p_e = math.log((1 - m) * (1 / emission_probability[behind_pos]['<sum>']))
    else:
        p_e = math.log(
            m * (emission_probability[behind_pos][word] / emission_probability[behind_pos]['<sum>']) + (1 - m) * (
                    1 / emission_probability[behind_pos]['<sum>']))
    return -(p_t + p_e)


def findMaxKey(dicTmp: dict):
    """
    找出最后一个词最可能的词性
    :param dicTmp: 最后一个词的词性标注概率列表
    :return: 该词最可能的词性
    """
    max_p = float('inf')
    max_key = list()
    for key in dicTmp.keys():
        if dicTmp[key][1] < max_p:
            max_p = dicTmp[key][1]
            max_key.append(key)
    return max_key[-1]


def HMMPos(words: list):
    """
    利用隐马尔可夫模型进行词性标注
    :param words: 已完成分词的语句
    :return: 标注好词性的语句
    """
    # 词性标注图，记录路径和概率
    path = list()
    # 利用Viterbi算法完成图
    length = len(words)
    point = 0
    while point < length:
        pos_dic = dict()
        if point == 0:
            for behind_pos in transition_probability.keys():
                pos_dic[behind_pos] = ['<EOS>', possHMM('<EOS>', behind_pos, words[point])]
        else:
            for behind_pos in transition_probability.keys():
                pos_dic[behind_pos] = ['', float('inf')]
                for front_pos in transition_probability.keys():
                    p_t_e = path[point - 1][front_pos][1] + possHMM(front_pos, behind_pos, words[point])
                    if p_t_e < pos_dic[behind_pos][1]:
                        pos_dic[behind_pos] = [front_pos, p_t_e]
        path.append(pos_dic)
        point += 1
    # 回溯
    point -= 1
    pos = list([findMaxKey(path[point])])
    while point > 0:
        pos.append(path[point][pos[-1]][0])
        point -= 1
    pos.reverse()
    # 拼装
    result = list()
    while point < length:
        result.append(words[point] + '/' + pos[point])
        point += 1
    return result


# --------------------------------------------------------
# 评估函数


def correctWordsCount(resultTmp: list, originalTmp: list):
    """
    统计语句中正确的分词个数
    :param resultTmp: 分词结果
    :param originalTmp: 语料分词结果
    :return:
    """
    rtv = 0
    resultPoint = list([0])
    point = list([0])
    count = 0
    for word in resultTmp:
        count += len(word)
        resultPoint.append(count)
    count = 0
    for word in originalTmp:
        count += len(word)
        if count in resultPoint:
            point.append(resultPoint.index(count))
    for i in point:
        if i + 1 in point:
            rtv += 1
    return rtv


def correctPosCount(resultTmp: list, originalTmp: list):
    """
    统计语句中正确标注词性的个数
    :param resultTmp: HMM词性标注结果
    :param originalTmp: 语料标注词性结果
    :return:
    """
    rtv = 0
    length = len(resultTmp)
    point = 0
    while point < length:
        if resultTmp[point] == originalTmp[point]:
            rtv += 1
        point += 1
    return rtv


def correctWordsAndPosCount(resultTmp: list, originalTmp: list):
    """
    统计语句中正确分词和词性个数
    :param resultTmp: 分词及词性标注结果
    :param originalTmp: 语料分词及词性标注结果
    :return:
    """
    rtv = 0
    resultPoint = list([[0, '']])
    point = list([[0, '']])
    count = 0
    for word in resultTmp:
        count += len(word[:word.find('/')])
        resultPoint.append([count, word])
    count = 0
    for word in originalTmp:
        count += len(word[:word.find('/')])
        tmp = [i[0] for i in resultPoint]
        if count in tmp:
            point.append([tmp.index(count), word])

    point_index = [i[0] for i in point]
    length = len(point)
    i = 0
    while i < length:
        if (point[i][0] + 1 in point_index) and (point[i + 1][1] == resultPoint[point[i][0] + 1][1]):
            rtv += 1
        i += 1

    return rtv


def evaluate(sample: int, predict: int, correct: int, size: float, start: float, end: float):
    # 精确率P
    precision = correct / predict
    # 召回率R
    recall = correct / sample
    # F1
    f1 = (2 * precision * recall) / (precision + recall)
    # 效率
    efficiency = size / (end - start)

    print('精确率P:%.2f' % (precision * 100))
    print('召回率R:%.2f' % (recall * 100))
    print('F1:%.2f' % (f1 * 100))
    print('效率:%.2f' % efficiency + 'kb/s')
