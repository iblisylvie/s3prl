class Phonemizer:
    def __init__(self, lexicon_files):
        self.word2phonemes = {}
        for lexicon_file in lexicon_files:
            with open(lexicon_file, 'rt') as fin:
                for line in fin:
                    cols = line.strip().split('\t')
                    if len(cols) < 2:
                        print('invalid line {}'.format(cols))
                        continue
                    word = cols[0]
                    phonemes = ' '.join(cols[1:])
                    self.word2phonemes[word] = phonemes  # 如果有多音字，选最后一种
        self.vocabs = self.word2phonemes.keys()
        self.phonemes = set()
        for key, value in self.word2phonemes.items():
            self.phonemes.update(value.split(' '))

    def get_phoneme_set(self):
      return self.phonemes

    def text_to_phonemes(self, text):
        words = self._forward_longest_segment(text)
        phonemes = []
        for word in words:
            if word in self.word2phonemes:
                phonemes.append(self.word2phonemes[word])
            else:
                print('Warning: No phoneme found for {}'.format(word))
        return ' '.join(phonemes)
    
    def segment(self, text):
        return self._forward_longest_segment(text)

    # 正向最长匹配分词
    def _forward_longest_segment(self, text):
        # print('forward_longest_segment <{}>'.format(text))
        segments = []
        if not text:
            return segments
        text_to_be_segmented = text
        while text_to_be_segmented:
            found = False
            text_len = len(text_to_be_segmented)
            for i in range(text_len, 0, -1):
                subword = text_to_be_segmented[:i]
                if len(subword) == 1 or subword in self.vocabs:
                    segments.append(subword)
                    text_to_be_segmented = text_to_be_segmented[i:]
                    found = True
                    break
            assert found
        return segments


if __name__ == '__main__':
    print(Phonemizer(['../lexicon/bpe-lexicon.txt']).text_to_phonemes('聚氧乙烯聚氧丙烯季戊四醇醚'))
    print(Phonemizer(['../lexicon/bpe-lexicon.txt']).text_to_phonemes('今天天气真好，都督'))
    phonemes = Phonemizer(['../lexicon/bpe-lexicon.txt']).get_phoneme_set()
    with open('bpe-phoneme.txt', 'wt') as fout:
      fout.write('\n'.join(phonemes))