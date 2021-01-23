import difflib
from enum import IntEnum

import keras
import sklearn_crfsuite

from training import StatisticalTraining, NeuronTraining


class WordToken(IntEnum):
    WORD = 0
    REFERENCE = 1


class SyllableToken(IntEnum):
    SYLLABLE = 0
    BOW = 1
    EOW = 2
    REFERENCE = 3


class ClarinPlStatisticalApproach(StatisticalTraining):
    """
    """
    CORRECT_DATA = 'correct_data'
    UNNECESSARY = 'unnecessary'

    @property
    def division_param(self):
        return 0.5

    @property
    def c1(self):
        return 0.1

    @property
    def c2(self):
        return 0.1

    def word2features(self, sent, i):
        syllable = sent[i][SyllableToken.SYLLABLE]

        features = {
            'bias': 1.0,
            'syllable': syllable,
            'bow': sent[i][SyllableToken.BOW],
            'eow': sent[i][SyllableToken.EOW],
        }
        margin = 1
        if i - margin > 0:
            syllable1 = sent[i - margin][SyllableToken.SYLLABLE]
            features.update({
                '-1:syllable': syllable1,
            })

            margin += 1
            if i - margin > 0:
                syllable1 = sent[i - margin][SyllableToken.SYLLABLE]
                features.update({
                    '-2:syllable': syllable1,
                })

        margin = 1
        if i + margin < len(sent):
            syllable1 = sent[i + margin][SyllableToken.SYLLABLE]
            features.update({
                '+1:syllable': syllable1,
            })

            margin += 1
            if i + margin < len(sent):
                syllable1 = sent[i + margin][SyllableToken.SYLLABLE]
                features.update({
                    '+2:syllable': syllable1,
                })

        return features

    @staticmethod
    def merge(list_of_strings):
        return ' '.join(list_of_strings)

    def similar(self, w1, w2):
        if isinstance(w1, str):
            words1 = [w1]
        else:
            words1 = w1
        if isinstance(w2, str):
            words2 = [w2]
        else:
            words2 = w2
        return difflib.SequenceMatcher(None, self.merge(words1), self.merge(words2)).ratio()

    @staticmethod
    def syllabify_word(word):
        vowels = 'a', 'ą', 'e', 'ę', 'i', 'y', 'o', 'u'
        syllables = []
        syllable = ""
        for x in word.lower():
            if x in vowels:
                syllable += x
                syllables.append(syllable)
                syllable = ""
            else:
                syllable += x
        if syllable:
            syllables.append(syllable)

        return syllables

    def syllabify(self, target):
        if isinstance(target, str):
            return self.syllabify_word(target)
        else:
            syllables = []
            for word in target:
                syllables += self.syllabify_word(word)
            return syllables

    def prepare_all_sents1(self):
        self.all_sents = []
        sents_file = open('ClarinPlTrain/1best.txt', 'r')
        sents_reference_file = open('ClarinPlTrain/reference.txt', 'r')

        for sents_line_str in sents_file.readlines():
            sents_line = [ascii(s) for s in sents_line_str.split()[1:]]
            ref_line = [ascii(s) for s in sents_reference_file.readline().split()[1:]]
            current_sentence = []
            sents_i = 0
            ref_i = 0
            while sents_i < len(sents_line) and ref_i < len(ref_line):
                oneXone = self.similar([sents_line[sents_i]], [ref_line[ref_i]])

                if ref_i + 1 < len(ref_line):
                    oneXtwo = self.similar([sents_line[sents_i]], [ref_line[ref_i], ref_line[ref_i + 1]])
                else:
                    oneXtwo = 0

                if sents_i + 1 < len(sents_line):
                    twoXone = self.similar([sents_line[sents_i], sents_line[sents_i + 1]], [ref_line[ref_i]])
                else:
                    twoXone = 0

                if sents_i + 1 < len(sents_line) and ref_i + 1 < len(ref_line):
                    twoXtwo = self.similar(
                        [sents_line[sents_i], sents_line[sents_i + 1]],
                        [ref_line[ref_i], ref_line[ref_i + 1]],
                    )
                else:
                    twoXtwo = 0

                best = max(oneXone, oneXtwo, twoXone, twoXtwo)
                if best == oneXone:
                    current_sentence.append((
                        sents_line[sents_i],
                        ref_line[ref_i],
                    ))
                    sents_i += 1
                    ref_i += 1
                elif best == oneXtwo:
                    current_sentence.append((
                        sents_line[sents_i],
                        self.merge([ref_line[ref_i], ref_line[ref_i + 1]]),
                    ))
                    sents_i += 1
                    ref_i += 2
                elif best == twoXone:
                    current_sentence.append((
                        self.merge([sents_line[sents_i], sents_line[sents_i + 1]]),
                        ref_line[ref_i],
                    ))
                    sents_i += 2
                    ref_i += 1
                elif best == twoXtwo:
                    current_sentence.append((
                        sents_line[sents_i],
                        ref_line[ref_i],
                    ))
                    sents_i += 1
                    ref_i += 1
                    current_sentence.append((
                        sents_line[sents_i],
                        ref_line[ref_i],
                    ))
                    sents_i += 1
                    ref_i += 1
            if sents_i < len(sents_line):
                current_sentence[-1] = (self.merge([current_sentence[-1][WordToken.WORD]] + sents_line[sents_i:]), current_sentence[-1][WordToken.REFERENCE])
            if ref_i < len(ref_line):
                current_sentence[-1] = (current_sentence[-1][WordToken.WORD], self.merge(
                    [current_sentence[-1][WordToken.REFERENCE]] + ref_line[ref_i:]))
            if len(current_sentence) > 0:
                self.all_sents.append(current_sentence)

    def prepare_all_sents(self):
        print('Preparing sentences...')

        self.all_sents = []
        sentences = []
        sents_file = open('ClarinPlTrain/1best.txt', 'r')
        sents_reference_file = open('ClarinPlTrain/reference.txt', 'r')

        for sents_line_str in sents_file.readlines():
            sents_line = [ascii(s) for s in sents_line_str.split()[1:]]
            ref_line = [ascii(s) for s in sents_reference_file.readline().split()[1:]]
            current_sentence = [(s, self.UNNECESSARY) for s in sents_line]

            sents_i = 0
            for index, ref in enumerate(ref_line):
                if sents_i - 2 > 0:
                    minus_two = self.similar(ref, sents_line[sents_i - 2])
                else:
                    minus_two = 0
                if sents_i - 1 > 0:
                    minus_one = self.similar(ref, sents_line[sents_i - 1])
                else:
                    minus_one = 0
                if sents_i < len(sents_line):
                    zero = self.similar(ref, sents_line[sents_i])
                else:
                    zero = 0
                if sents_i + 1 < len(sents_line):
                    one = self.similar(ref, sents_line[sents_i + 1])
                else:
                    one = 0
                if sents_i + 2 < len(sents_line):
                    two = self.similar(ref, sents_line[sents_i + 2])
                else:
                    two = 0
                best = max(zero, one, two, minus_two, minus_one)
                if best == zero:
                    pass
                elif best == minus_one:
                    sents_i = sents_i - 1
                elif best == one:
                    sents_i = sents_i + 1
                elif best == minus_two:
                    sents_i = sents_i - 2
                else:
                    sents_i = sents_i + 2
                if current_sentence[sents_i][WordToken.REFERENCE] == self.UNNECESSARY or (
                    current_sentence[sents_i][WordToken.REFERENCE] != self.CORRECT_DATA and self.similar(
                        current_sentence[sents_i][WordToken.WORD],
                        ref,
                    ) > self.similar(
                        current_sentence[sents_i][WordToken.WORD],
                        current_sentence[sents_i][WordToken.REFERENCE],
                    )
                ):
                    if current_sentence[sents_i][WordToken.WORD] == ref:
                        current_sentence[sents_i] = (current_sentence[sents_i][WordToken.WORD], self.CORRECT_DATA)
                    else:
                        current_sentence[sents_i] = (current_sentence[sents_i][WordToken.WORD], ref)
                sents_i += 1
            if len(current_sentence) > 0:
                sentences.append(current_sentence)

        sents_file.close()
        sents_reference_file.close()

        for sentence in sentences[:int(len(sentences))]:
            current_sentence = []
            for word in sentence:
                word_syllables = self.syllabify_word(word[WordToken.WORD])
                if word[WordToken.REFERENCE] in (self.UNNECESSARY, self.CORRECT_DATA):
                    for i, syl in enumerate(word_syllables):
                        bow = i == 0
                        eow = i == len(word_syllables) - 1
                        current_sentence.append((syl, bow, eow, word[WordToken.REFERENCE]))
                else:
                    ref_syllables = self.syllabify_word(word[WordToken.REFERENCE])
                    for i, ws in enumerate(word_syllables):
                        bow = i == 0
                        eow = i == len(word_syllables) - 1
                        if i < len(ref_syllables):
                            if ws == ref_syllables[i]:
                                current_sentence.append((ws, bow, eow, self.CORRECT_DATA))
                            else:
                                current_sentence.append((ws, bow, eow, ref_syllables[i]))
                        else:
                            current_sentence.append((ws, bow, eow, self.UNNECESSARY))
            self.all_sents.append(current_sentence)

    def sent2labels(self, sent):
        return [reference for word, bow, eow, reference in sent]

    def train_model(self):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=self.c1,
            c2=self.c2,
            verbose=True,
            max_iterations=5,
        )
        print('Begin training...')
        crf.fit(self.x_train, self.y_train)

        self.labels = list(crf.classes_)
        self.labels.remove('correct_data')
        print(self.labels)

        print('Predicting results...')
        self.y_predict = crf.predict(self.x_test)


# clarin_approach = ClarinPlStatisticalApproach()
# clarin_approach.prepare_all_sents()
# print(clarin_approach.all_sents[1])
# clarin_approach.prepare_train_data()
#clarin_approach.show_best_c_params()

# clarin_approach.run_training()
# print('Current f1 score:', clarin_approach.get_f1_score())
# print('Classification report\n', clarin_approach.get_classification_report())


class ClarinPlNeuronApproach(NeuronTraining):
    CORRECT_DATA = 'correct_data'
    UNNECESSARY = 'unnecessary'
    epochs = 150

    @staticmethod
    def merge(list_of_strings):
        return ' '.join(list_of_strings)

    def similar(self, w1, w2):
        if isinstance(w1, str):
            words1 = [w1]
        else:
            words1 = w1
        if isinstance(w2, str):
            words2 = [w2]
        else:
            words2 = w2
        return difflib.SequenceMatcher(None, self.merge(words1), self.merge(words2)).ratio()

    @staticmethod
    def syllabify_word(word):
        vowels = ('a', 'ą', 'e', 'ę', 'i', 'y', 'o', 'u')
        syllables = []
        syllable = ""
        for x in word.lower():
            if x in vowels:
                syllable += x
                syllables.append(syllable)
                syllable = ""
            else:
                syllable += x
        if syllable:
            syllables.append(syllable)

        return syllables

    def syllabify(self, target):
        if isinstance(target, str):
            return self.syllabify_word(target)
        else:
            syllables = []
            for word in target:
                syllables += self.syllabify_word(word)
            return syllables

    def prepare_all_sents(self):
        print('Preparing sentences...')

        self.all_sents = []
        sentences = []
        sents_file = open('ClarinPlTrain/1best.txt', 'r')
        sents_reference_file = open('ClarinPlTrain/reference.txt', 'r')

        for sents_line_str in sents_file.readlines():
            sents_line = [ascii(s) for s in sents_line_str.split()[1:]]
            ref_line = [ascii(s) for s in sents_reference_file.readline().split()[1:]]
            current_sentence = [(s, self.UNNECESSARY) for s in sents_line]

            sents_i = 0
            for index, ref in enumerate(ref_line):
                if sents_i - 2 > 0:
                    minus_two = self.similar(ref, sents_line[sents_i - 2])
                else:
                    minus_two = 0
                if sents_i - 1 > 0:
                    minus_one = self.similar(ref, sents_line[sents_i - 1])
                else:
                    minus_one = 0
                if sents_i < len(sents_line):
                    zero = self.similar(ref, sents_line[sents_i])
                else:
                    zero = 0
                if sents_i + 1 < len(sents_line):
                    one = self.similar(ref, sents_line[sents_i + 1])
                else:
                    one = 0
                if sents_i + 2 < len(sents_line):
                    two = self.similar(ref, sents_line[sents_i + 2])
                else:
                    two = 0
                best = max(zero, one, two, minus_two, minus_one)
                if best == zero:
                    pass
                elif best == minus_one:
                    sents_i = sents_i - 1
                elif best == one:
                    sents_i = sents_i + 1
                elif best == minus_two:
                    sents_i = sents_i - 2
                else:
                    sents_i = sents_i + 2
                if current_sentence[sents_i][WordToken.REFERENCE] == self.UNNECESSARY or (
                    current_sentence[sents_i][WordToken.REFERENCE] != self.CORRECT_DATA and self.similar(
                        current_sentence[sents_i][WordToken.WORD],
                        ref,
                    ) > self.similar(
                        current_sentence[sents_i][WordToken.WORD],
                        current_sentence[sents_i][WordToken.REFERENCE],
                    )
                ):
                    if current_sentence[sents_i][WordToken.WORD] == ref:
                        current_sentence[sents_i] = (current_sentence[sents_i][WordToken.WORD], self.CORRECT_DATA)
                    else:
                        current_sentence[sents_i] = (current_sentence[sents_i][WordToken.WORD], ref)
                sents_i += 1
            if len(current_sentence) > 0:
                sentences.append(current_sentence)

        sents_file.close()
        sents_reference_file.close()

        for sentence in sentences:
            current_sentence = []
            for word in sentence:
                word_syllables = [ascii(a) for a in self.syllabify_word(word[WordToken.WORD])]
                if word[WordToken.REFERENCE] in (self.UNNECESSARY, self.CORRECT_DATA):
                    for i, syl in enumerate(word_syllables):
                        bow = i == 0
                        eow = i == len(word_syllables) - 1
                        current_sentence.append((syl, bow, eow, word[WordToken.REFERENCE]))
                else:
                    ref_syllables = [ascii(a) for a in self.syllabify_word(word[WordToken.REFERENCE])]
                    for i, ws in enumerate(word_syllables):
                        bow = i == 0
                        eow = i == len(word_syllables) - 1
                        if i < len(ref_syllables):
                            if ws == ref_syllables[i]:
                                current_sentence.append((ws, bow, eow, self.CORRECT_DATA))
                            else:
                                current_sentence.append((ws, bow, eow, ref_syllables[i]))
                        else:
                            current_sentence.append((ws, bow, eow, self.UNNECESSARY))
            self.all_sents.append(current_sentence)

    def prepare_token_to_id(self):
        self.token_to_id_map[self.CORRECT_DATA] = 0
        self.token_to_id_map[self.UNNECESSARY] = 1
        self.id_to_token_map[0] = self.CORRECT_DATA
        self.id_to_token_map[1] = self.UNNECESSARY

        i = 2
        for sentence in self.all_sents:
            for syllable in sentence:
                if syllable[SyllableToken.SYLLABLE] not in self.token_to_id_map:
                    self.token_to_id_map[syllable[SyllableToken.SYLLABLE]] = i
                    self.id_to_token_map[i] = syllable[SyllableToken.SYLLABLE]
                    i += 1
                if syllable[SyllableToken.REFERENCE] not in self.token_to_id_map:
                    self.token_to_id_map[syllable[SyllableToken.REFERENCE]] = i
                    self.id_to_token_map[i] = syllable[SyllableToken.REFERENCE]
                    i += 1

        self.max_words = i
        self.num_classes = i

    def word2features(self, sent, i):
        return self.token_to_id_map[sent[i][SyllableToken.SYLLABLE]]

    def sent2labels(self, sent):
        return [self.token_to_id_map[syl[SyllableToken.REFERENCE]] for syl in sent]

    def predict(self, sentence):
        syllables = self.sent2features([ascii(a) for a in self.syllabify(sentence.split()[1:])])
        print(self.model.predict(keras.utils.to_categorical(syllables)))


clarin_neuro = ClarinPlNeuronApproach()
clarin_neuro.run_training()