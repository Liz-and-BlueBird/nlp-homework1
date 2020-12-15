from SegAndPosSeg import *

if __name__ == '__main__':
    sentence = '他认为，研究生物的前景很光明'

    print(forwardSegment(sentence))
    print(backwardSegment(sentence))
    print(bidirectionalSegment(sentence))

    print(twoGramSegment(sentence))

    print(HMMPos(twoGramSegment(sentence)))
