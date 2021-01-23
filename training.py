from abc import ABC, abstractmethod

import keras
import sklearn
import scipy.stats
import sklearn_crfsuite
from keras_preprocessing.text import Tokenizer
from matplotlib import pyplot
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import Sequential


class StatisticalTraining(ABC):
    C1 = 0.1
    C2 = 0.1
    DIVISION_PARAM = 0.8

    def __init__(self):
        self.all_sents = []

        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []

        self.y_predict = []

        self.labels = []

    @property
    def divider(self):
        return int(len(self.all_sents) * self.DIVISION_PARAM)

    @property
    def train_sents(self):
        return self.all_sents[:self.divider]

    @property
    def test_sents(self):
        return self.all_sents[self.divider:]

    @abstractmethod
    def prepare_all_sents(self):
        pass

    @abstractmethod
    def word2features(self, sent, i):
        pass

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    @abstractmethod
    def sent2labels(self, sent):
        pass

    def prepare_train_data(self):
        self.x_train = [self.sent2features(s) for s in self.train_sents]
        self.y_train = [self.sent2labels(s) for s in self.train_sents]

        self.x_test = [self.sent2features(s) for s in self.test_sents]
        self.y_test = [self.sent2labels(s) for s in self.test_sents]

    def train_model(self):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=self.C1,
            c2=self.C2,
            verbose=True,
            max_iterations=100,
            all_possible_transitions=True
        )
        print('Begin training...')
        crf.fit(self.x_train, self.y_train)

        self.y_predict = crf.predict(self.x_test)

    def get_f1_score(self):
        return metrics.flat_f1_score(self.y_test, self.y_predict, average='weighted', labels=self.labels)

    def get_classification_report(self):
        return metrics.flat_classification_report(self.y_test, self.y_predict, labels=self.labels, digits=3)

    def run_training(self):
        self.prepare_all_sents()
        self.prepare_train_data()

        self.train_model()

    def show_best_c_params(self):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted', labels=self.labels)

        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=4,
                                n_iter=5,
                                scoring=f1_scorer)
        rs.fit(self.x_train, self.y_train)
        print('Best c params are:', rs.best_params_)


class NeuronTraining(ABC):
    DIVISION_PARAM = 0.5
    batch_size = 32
    epochs = 5

    def __init__(self):
        self.all_sents = []

        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []

        self.token_to_id_map = {}
        self.id_to_token_map = {}

        self.max_words = 0
        self.num_classes = 0

        self.model = None

    @property
    def divider(self):
        return int(len(self.all_sents) * self.DIVISION_PARAM)

    @property
    def train_sents(self):
        return self.all_sents[:self.divider]

    @property
    def test_sents(self):
        return self.all_sents[self.divider:]

    def prepare_train_data(self):
        self.x_train = [self.sent2features(s) for s in self.train_sents]
        self.y_train = [self.sent2labels(s) for s in self.train_sents]

        self.x_test = [self.sent2features(s) for s in self.test_sents]
        self.y_test = [self.sent2labels(s) for s in self.test_sents]

    @abstractmethod
    def prepare_all_sents(self):
        pass

    @abstractmethod
    def prepare_token_to_id(self):
        pass

    @abstractmethod
    def word2features(self, sent, i):
        pass

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    @abstractmethod
    def sent2labels(self, sent):
        pass

    def run_training(self):
        self.prepare_all_sents()
        self.prepare_token_to_id()
        self.prepare_train_data()

        self.train_model()

    def train_model(self):
        tokenizer = Tokenizer(num_words=self.max_words)
        x_train = tokenizer.sequences_to_matrix(self.x_train, mode='binary')
        x_test = tokenizer.sequences_to_matrix(self.x_test, mode='binary')

        y_train = tokenizer.sequences_to_matrix(self.y_train, mode='binary')
        y_test = tokenizer.sequences_to_matrix(self.y_test, mode='binary')
        # y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        # y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

        self.model = Sequential()
        self.model.add(Dense(1024, input_shape=(self.max_words,)))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(self.num_classes))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_squared_error',
                      optimizer='rmsprop', metrics=['binary_accuracy'])

        history = self.model.fit(x_train, y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=(x_test, y_test))

        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.show()
