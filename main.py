from enum import IntEnum

import spacy as spacy

from training import StatisticalTraining, NeuronTraining


class Token(IntEnum):
    WORD = 0
    POSTAG = 1
    TAGS = 2
    LABEL = 3
    HEAD = 4
    DEP = 5


class NKJPStatisticalApproach1(StatisticalTraining):
    """
        In this approach I'm remembering all interp symbols before and after current word and two prev and next full
        words.
        Dataset is divided into sentences and I ignore ending sentence dots.

        Results:
        F1: 0.807
        Precision: 0.752
        Recall: 0.772
    """

    C1 = 0.271
    C2 = 0.011

    def __init__(self):
        super(NKJPStatisticalApproach1, self).__init__()
        self.labels = ['persName_surname', 'placeName_settlement', 'orgName', 'geogName', 'persName_addName', 'persName_forename', 'persName', 'time', 'placeName_country', 'date', 'placeName', 'placeName_region', 'placeName_bloc', 'placeName_district']

    def word2features(self, sent, i):
        word = sent[i][Token.WORD]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word.lower()[-3:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': sent[i][Token.POSTAG],
            'word.len': len(word),
            'maintag': sent[i][Token.TAGS].split(':')[0],
        }
        margin = 1
        while i - margin > 0 and sent[i - margin][Token.TAGS] == 'interp':
            features['-{}:interp'.format(margin)] = sent[i - margin][Token.WORD]
            margin += 1
        if i - margin > 0:
            word1 = sent[i - margin][Token.WORD]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:postag': sent[i - margin][Token.POSTAG],
                '-1:word.istitle()': word1.istitle(),
                '-1:word[-3:]': word1.lower()[-3:],
                '-1:maintag': sent[i - margin][Token.TAGS].split(':')[0],
            })

            margin += 1
            while i - margin > 0 and sent[i - margin][Token.TAGS] == 'interp':
                margin += 1

            if i - margin > 0:
                word1 = sent[i - margin][Token.WORD]
                features.update({
                    '-2:word.lower()': word1.lower(),
                    '-2:word.istitle()': word1.istitle(),
                    '-2:postag': sent[i - margin][Token.POSTAG],
                    '-2:word[-3:]': word1.lower()[-3:],
                })
        else:
            features['BOS'] = True

        margin = 1
        while i + margin < len(sent) - 1 and sent[i + margin][Token.TAGS] == 'interp':
            features['+{}:interp'.format(margin)] = sent[i + margin][Token.WORD]
            margin += 1
        if i + margin < len(sent):
            word1 = sent[i + margin][Token.WORD]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:postag': sent[i + margin][Token.POSTAG],
                '+1:word[-3:]': word1.lower()[-3:],
                '+1:maintag': sent[i + margin][Token.TAGS].split(':')[0],
            })

            margin += 1
            while i + margin < len(sent) - 1 and sent[i + margin][Token.TAGS] == 'interp':
                margin += 1

            if i + margin < len(sent):
                word1 = sent[i + margin][Token.WORD]
                features.update({
                    '+2:word.lower()': word1.lower(),
                    '+2:word.istitle()': word1.istitle(),
                    '+2:postag': sent[i + margin][Token.POSTAG],
                    '+2:word[-3:]': word1.lower()[-3:],
                })
        else:
            features['EOS'] = True

        return features

    def prepare_all_sents(self):
        self.all_sents = []
        current_sentence = []
        sents_file = open('nkjp-morph-named.txt', 'r')
        sents_pos_file = open('nkjp_tab_onlypos.txt', 'r')

        for line in sents_pos_file.readlines():
            if line == '&\t&\tinterp\n' and len(current_sentence) > 0:
                if current_sentence[-1][Token.WORD] == '.':
                    current_sentence = current_sentence[:-1]
                self.all_sents.append(current_sentence)
                current_sentence = []
            else:
                current_sentence.append(sents_file.readline()[:-1].split('\t'))

        if len(current_sentence) > 0 and current_sentence[-1][Token.WORD] == '.':
            current_sentence = current_sentence[:-1]
        if len(current_sentence) > 0:
            self.all_sents.append(current_sentence)

        sents_file.close()
        sents_pos_file.close()

    def sent2labels(self, sent):
        return [label for word, lemma, tag, label in sent]


class NKJPStatisticalApproach2(StatisticalTraining):
    """
        Results:
        F1: 0.820
        Precision: 0.864
        Recall: 0.785
    """
    DIVISION_PARAM = 0.8
    C1 = 0.705
    C2 = 0.008

    def __init__(self):
        super(NKJPStatisticalApproach2, self).__init__()
        self.labels = ['persName_surname', 'placeName_settlement', 'orgName', 'geogName', 'persName_addName', 'persName_forename', 'persName', 'time', 'placeName_country', 'date', 'placeName', 'placeName_region', 'placeName_bloc', 'placeName_district']

    def word2features(self, sent, i):
        word = sent[i][Token.WORD]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word.lower()[-3:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': sent[i][Token.POSTAG],
            'word.len': len(word),
            'maintag': sent[i][Token.TAGS].split(':')[0],
            'word:head': sent[i][Token.HEAD],
            'word:dep': sent[i][Token.DEP],
        }
        margin = 1
        if i - margin > 0:
            word1 = sent[i - margin][Token.WORD]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:postag': sent[i - margin][Token.POSTAG],
                '-1:word.istitle()': word1.istitle(),
                '-1:word[-3:]': word1.lower()[-3:],
                '-1:maintag': sent[i - margin][Token.TAGS].split(':')[0],
            })

            margin += 1
            if i - margin > 0:
                word1 = sent[i - margin][Token.WORD]
                features.update({
                    '-2:word.lower()': word1.lower(),
                    '-2:word.istitle()': word1.istitle(),
                    '-2:postag': sent[i - margin][Token.POSTAG],
                })
        else:
            features['BOS'] = True

        margin = 1
        if i + margin < len(sent):
            word1 = sent[i + margin][Token.WORD]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:postag': sent[i + margin][Token.POSTAG],
                '+1:word[-3:]': word1.lower()[-3:],
                '+1:maintag': sent[i + margin][Token.TAGS].split(':')[0],
            })

            margin += 1
            if i + margin < len(sent):
                word1 = sent[i + margin][Token.WORD]
                features.update({
                    '+2:word.lower()': word1.lower(),
                    '+2:word.istitle()': word1.istitle(),
                    '+2:postag': sent[i + margin][Token.POSTAG],
                })
        else:
            features['EOS'] = True

        return features

    def parse_sentences(self):
        print('Parse sentences...')
        nlp = spacy.load("pl_core_news_lg")
        for current_sentence in self.all_sents:
            sent = nlp(' '.join(s[Token.WORD] for s in current_sentence))
            i = 0
            cached_token = ''
            cached_head = None
            cached_dep = None
            for token_index, token in enumerate(sent):
                cached_token += token.orth_
                cached_head = cached_head or token.head.lemma_
                cached_dep = cached_dep or token.dep_
                if cached_token == current_sentence[i][Token.WORD]:
                    current_sentence[i].append(cached_head)
                    current_sentence[i].append(cached_dep)
                    i += 1
                    cached_token = ''
                    cached_head = None
                    cached_dep = None

            if cached_token != '':
                print(cached_token)
                raise ValueError

    def prepare_all_sents(self):
        print('Prepare sentences...')
        self.all_sents = []
        current_sentence = []
        sents_file = open('nkjp-morph-named.txt', 'r')
        sents_pos_file = open('nkjp_tab_onlypos.txt', 'r')

        for line in sents_pos_file.readlines():
            if line == '&\t&\tinterp\n':
                if len(current_sentence) > 0:
                    self.all_sents.append(current_sentence)
                    current_sentence = []
            else:
                morph_line = ''.join(sents_file.readline().split(' '))
                current_sentence.append(morph_line[:-1].split('\t'))

        if len(current_sentence) > 0:
            self.all_sents.append(current_sentence)

        sents_file.close()
        sents_pos_file.close()

        self.parse_sentences()

    def sent2labels(self, sent):
        return [label for word, lemma, tag, label, head, dep in sent]


class NKJPNeuronApproach(NeuronTraining):
    DIVISION_PARAM = 0.5
    classes = ['O', 'persName_surname', 'placeName_settlement', 'orgName', 'geogName', 'persName_addName',
               'persName_forename', 'persName', 'time', 'placeName_country', 'date', 'placeName', 'placeName_region',
               'placeName_bloc', 'placeName_district']

    def word2features(self, sent, i):
        return self.token_to_id_map[sent[i][Token.POSTAG]]

    def prepare_all_sents(self):
        print('Prepare sentences...')
        self.all_sents = []
        current_sentence = []
        sents_file = open('nkjp-morph-named.txt', 'r')
        sents_pos_file = open('nkjp_tab_onlypos.txt', 'r')

        for line in sents_pos_file.readlines():
            if line == '&\t&\tinterp\n':
                if len(current_sentence) > 0:
                    self.all_sents.append(current_sentence)
                    current_sentence = []
            else:
                morph_line = ''.join(sents_file.readline().split(' '))
                current_sentence.append(morph_line[:-1].split('\t'))

        if len(current_sentence) > 0:
            self.all_sents.append(current_sentence)

        sents_file.close()
        sents_pos_file.close()

    def sent2labels(self, sent):
        return [self.classes.index(label) for word, lemma, tag, label in sent]

    def prepare_token_to_id(self):
        i = 0
        for sentence in self.all_sents:
            for word in sentence:
                if word[Token.POSTAG] not in self.token_to_id_map:
                    self.token_to_id_map[word[Token.POSTAG]] = i
                    self.id_to_token_map[i] = word[Token.POSTAG]
                    i += 1

        self.max_words = i
        self.num_classes = len(self.classes)


# nkjp_approach = NKJPStatisticalApproach2()
# nkjp_approach.prepare_all_sents()
# nkjp_approach.prepare_train_data()
# nkjp_approach.show_best_c_params()

# nkjp_approach.run_training()
# print('Current f1 score:', nkjp_approach.get_f1_score())
# print('Classification report\n', nkjp_approach.get_classification_report())

nkjp_nuron = NKJPNeuronApproach()
nkjp_nuron.run_training()
