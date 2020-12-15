from SegAndPosSeg import *
from tqdm import tqdm

import time
import os


def dicSegmentTest():
    """
    字典分词测试
    :return: None
    """
    sampleWords = 0
    predictWords = 0
    correctWords = 0
    size = os.path.getsize('test set.txt') / 1024

    start = time.perf_counter()

    with open('test set.txt', 'r', encoding='utf-8') as testSet:
        lines = testSet.readlines()
        for line in tqdm(lines):
            # 语句恢复成待切分的状态
            sentence = ''.join([word[:word.find('/')] for word in line.strip('\n').split(' ')])
            # 双向最长匹配分词结果
            result = bidirectionalSegment(sentence)
            # 语料的分词结果
            original = [word[:word.find('/')] for word in line.strip('\n').split(' ')]
            # 统计语料分词个数
            sampleWords += len(original)
            # 统计双向最长匹配分词个数
            predictWords += len(result)
            # 统计正确分词的个数
            correctWords += correctWordsCount(result, original)

    end = time.perf_counter()

    # 评估
    print('字典分词：')
    evaluate(sampleWords, predictWords, correctWords, size, start, end)


def twoGramSegmentTest():
    """
    二元语法分词测试
    :return: None
    """
    sampleWords = 0
    predictWords = 0
    correctWords = 0
    size = os.path.getsize('test set.txt') / 1024

    start = time.perf_counter()

    with open('test set.txt', 'r', encoding='utf-8') as testSet:
        lines = testSet.readlines()
        for line in tqdm(lines):
            # 语句恢复成待切分的状态
            sentence = ''.join([word[:word.find('/')] for word in line.strip('\n').split(' ')])
            # 双向最长匹配分词结果
            result = twoGramSegment(sentence)
            # 语料的分词结果
            original = [word[:word.find('/')] for word in line.strip('\n').split(' ')]
            # 统计语料分词个数
            sampleWords += len(original)
            # 统计双向最长匹配分词个数
            predictWords += len(result)
            # 统计正确分词的个数
            correctWords += correctWordsCount(result, original)

    end = time.perf_counter()

    # 评估
    print('二元语法分词：')
    evaluate(sampleWords, predictWords, correctWords, size, start, end)


def HMMPosTest():
    """
    HMM词性标注测试
    :return: None
    """
    samplePos = 0
    predictPos = 0
    correctPos = 0
    size = os.path.getsize('test set.txt') / 1024

    start = time.perf_counter()

    with open('test set.txt', 'r', encoding='utf-8') as test:
        lines = test.readlines()

        for line in tqdm(lines):
            sentence = line.strip('\n')
            # 提取词
            words = [word[:word.find('/')] for word in sentence.split(' ')]
            # HMM标注词性
            result = HMMPos(words)
            # 语料的词性标注结果
            original = sentence.split(' ')
            # 统计语料词性标注个数
            samplePos += len(original)
            # 统计HMM标注词性个数
            predictPos += len(result)
            # 统计正确标注词性的个数
            correctPos += correctPosCount(result, original)

    end = time.perf_counter()

    # 评估
    print('HMM词性标注：')
    evaluate(samplePos, predictPos, correctPos, size, start, end)


def twoGramAndHMMTest():
    """
    在二元语法分词的基础上进行HMM词性标注测试
    :return: None
    """
    sampleWords = 0
    predictWords = 0
    correctWords = 0
    size = os.path.getsize('test set.txt') / 1024

    start = time.perf_counter()

    with open('test set.txt', 'r', encoding='utf-8') as testSet:
        lines = testSet.readlines()
        for line in tqdm(lines):
            # 带词性标注的语料分词结果
            original = line.strip('\n').split(' ')
            # 语句恢复成待切分待标注的状态
            sentence = ''.join([word[:word.find('/')] for word in original])
            # 双向最长匹配分词结果
            words = twoGramSegment(sentence)
            # 在分词结果基础上进行词性标注
            result = HMMPos(words)
            # 统计语料词语个数
            sampleWords += len(original)
            # 统计双向最长匹配分词和HMM词性标注后词语个数
            predictWords += len(result)
            # 统计正确分词及词性标注的个数
            correctWords += correctWordsAndPosCount(result, original)

    end = time.perf_counter()

    # 评估
    print('二元语法分词+HMM词性标注：')
    evaluate(sampleWords, predictWords, correctWords, size, start, end)


if __name__ == '__main__':
    dicSegmentTest()
    twoGramSegmentTest()
    HMMPosTest()
    twoGramAndHMMTest()
