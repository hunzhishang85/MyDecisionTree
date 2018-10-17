import pandas as pd
import numpy as np
import sklearn
from sklearn.base import BaseEstimator
import pandas.core.algorithms as algos
from tqdm import tqdm, trange
import random

xy = pd.read_csv('./train_xy.csv')

x = xy.iloc[:, 3:80]
y = xy.y


def eh(sub_label):
    total = len(sub_label)
    p0 = (sub_label == 0).sum() / total
    p1 = 1 - p0
    if p0 == 0 or p1 == 0:
        return 0
    hx = -(p0 * np.log2(p0) + p1 * np.log2(p1))
    return hx


def split_label(feature, thershold, label):
    left_labels = label[feature <= thershold]
    right_labels = label[feature > thershold]
    return left_labels, right_labels


def split_features_and_label(features, best_feature, thershold, label):
    split_feature = features[best_feature]
    left_labels = label[split_feature <= thershold]
    left_features = features[split_feature <= thershold]
    right_labels = label[split_feature > thershold]
    right_features = features[split_feature > thershold]
    return (left_features, left_labels), (right_features, right_labels)


def gain(feature, thershold, label):
    left_labels, right_labels = split_label(feature, thershold, label)
    old = eh(label)
    left_ratio = len(left_labels) / len(label)
    new = eh(left_labels) * left_ratio + eh(right_labels) * (1 - left_ratio)
    g = old - new
    return g


def search_best_split(feature, label):
    uniques = np.unique(feature)
    nunique = len(uniques)
    if nunique == 1:
        return 0, 0
    gains = {}

    # 如果稀缺值小于100个
    if nunique <= 100:
        for theshold in uniques[:-1]:
            # print(theshold)
            gains[theshold] = gain(feature, theshold, label)

    # 如果稀缺值大于100个，用百分位来计算
    else:
        for pct_theshold in range(100):
            theshold = np.percentile(feature, pct_theshold)
            gains[theshold] = gain(feature, theshold, label)

    best_split = max(gains, key=gains.get)
    best_gain = gains[best_split]
    return best_split, best_gain


def search_best_feature(x, label):
    splits = {}
    gains = {}
    for fe_id in x:
        # print(fe_id)
        feature = x[fe_id]
        # print(len(np.unique(feature)))
        fe_split, fe_gain = search_best_split(feature, label)
        splits[fe_id] = fe_split
        gains[fe_id] = fe_gain
    best_feature = max(gains, key=gains.get)
    split_thershold = splits[best_feature]
    best_gain = gains[best_feature]
    return best_feature, split_thershold, best_gain


class myDT(object):
    def __init__(self, max_depth=4, min_split_gain=0.000001, min_data_in_leaf=10, verbose=True):
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.verbose = verbose
        self.min_data_in_leaf = min_data_in_leaf
        self.root = None
        self.fitted = False
        self.settings = {'max_depth': self.max_depth, 'min_split_gain': self.min_split_gain,
                         'verbose': self.verbose, 'min_data_in_leaf': self.min_data_in_leaf}

    def fit(self, x, y):
        self.root = DTleaf(current_depth=0, settings=self.settings)

        self.root.build(x, y)
        self.fitted = True
        return self

    def predict_proba(self, x):
        probas = {}
        for id, sample in x.iterrows():
            probas[id] = self.root.predict(sample)
        yp = pd.DataFrame.from_dict(probas, 'index')[0]
        return pd.concat([1 - yp, yp], 1).values


class DTleaf(object):
    def __init__(self, current_depth, settings):
        self.settings = settings
        self.best_feature = 'wtf'
        self.split_thershold = 0
        self.best_gain = 0
        self.proba = 0
        self.builded = False
        self.left_leaf = None
        self.right_leaf = None

        self.current_depth = current_depth
        if self.current_depth < self.settings['max_depth']:
            self.bottom = False
            # self.left_leaf = DTleaf(self.current_depth + 1,max_depth,min_split_gain,verbose)
            # self.right_leaf = DTleaf(self.current_depth +1,max_depth,min_split_gain,verbose)
        else:
            self.bottom = True

    @property
    def max_depth_(self):
        if self.bottom:
            return self.current_depth
        else:
            return max(self.left_leaf.max_depth_, self.right_leaf.max_depth_)

    def __str__(self):
        if not self.builded:
            return 'not_built_yet'
        else:
            return f'深度：{self.current_depth}\n分割特征：{self.best_feature}\n信息收益：{self.best_gain}\n' + \
                   f'本层概率{self.proba}'

    def __repr__(self):
        return self.__str__()

    def build(self, features, labels):
        if self.settings['verbose']:
            print(f'正在构建第{self.current_depth}层')
        # 概率值
        self.proba = labels.mean()
        self.builded = True
        if self.bottom:
            return

        if len(labels) <= self.settings['min_data_in_leaf']:
            self.bottom = True
            return

        if self.proba == 0 or self.proba == 1:
            self.bottom = True
            return

        # 最优切分
        best_feature, split_thershold, best_gain = search_best_feature(features, labels)

        if best_gain < self.settings['min_split_gain']:
            self.bottom = True
            return

        self.best_feature = best_feature
        self.best_gain = best_gain
        self.split_thershold = split_thershold

        # 切开训练集，供子树使用
        (left_features, left_labels), (right_features, right_labels) = \
            split_features_and_label(features, best_feature, split_thershold, labels)
        if self.settings['verbose']:
            print('   ' * self.current_depth + '左侧', end='')
        self.left_leaf = DTleaf(self.current_depth + 1, self.settings)
        self.left_leaf.build(left_features, left_labels)
        if self.settings['verbose']:
            print('   ' * self.current_depth + '右侧', end='')
        self.right_leaf = DTleaf(self.current_depth + 1, self.settings)
        self.right_leaf.build(right_features, right_labels)

    def predict(self, test_sample):
        if self.bottom:
            return self.proba
        else:
            my_feature = test_sample[self.best_feature]
            if my_feature <= self.split_thershold:
                return self.left_leaf.predict(test_sample)
            else:
                return self.right_leaf.predict(test_sample)

    def show(self):
        if self.bottom:
            print(f'第{self.current_depth}层' + f'概率为{self.proba}')
        else:
            print(f'第{self.current_depth}层')
            print('   ' * self.current_depth + '左侧', end='')
            self.left_leaf.show()
            print('   ' * self.current_depth + '右侧', end='')
            self.right_leaf.show()


class MyRF(BaseEstimator):
    def __init__(self, n_estimators=10, subsample=0.5, sub_feature=0.8, max_depth=6, min_split_gain=0.001, verbose=True,
                 random_state=42):
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.sub_feature = sub_feature
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.verbose = verbose
        self.random_state = random_state
        self.trees = []
        self.masks = {}
        # for i in range(self.n_estimators):

    def fit(self, train_x, train_y):

        for i in trange(self.n_estimators):  # ,disable=~self.verbose):
            # tree=myDT(max_depth=self.max_depth, min_split_gain=self.min_split_gain,
            #                        verbose=False)
            skf = StratifiedKFold(5, True, self.random_state + i)
            sample_i = next(skf.split(train_x, train_y), )[0]
            xs = train_x[sample_i, :]
            ys = train_y[sample_i]
            if self.sub_feature < 1:
                n_sub_feature = int(train_x.shape[1] * self.sub_feature)
                mask = random.sample(range(train_x.shape[1]), n_sub_feature)
                xs = xs[:, mask]
                self.masks[i] = mask
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            self.trees.append(tree)
            tree.fit(xs, ys)

        return self

    def predict_proba(self, test_x):
        probass = []
        for i, tree in enumerate(self.trees):
            if self.sub_feature < 1:
                mask = self.masks[i]
                sub_test_x = test_x[:, mask]
            else:
                sub_test_x = test_x
            probas = tree.predict_proba(sub_test_x)
            probass.append(probas)
        return np.mean(probass, 0)


# m1.fit(x,y)
# pd.Series.rank
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

skf = StratifiedKFold(shuffle=True, random_state=42, n_splits=6)
i1, i2 = next(skf.split(x, y))
# m1=myDT(max_depth=8,min_split_gain=0.001)
# m1=DecisionTreeClassifier(max_depth=8,min_samples_split=10,)
m1 = MyRF(n_estimators=3000, subsample=0.1, max_depth=6, min_split_gain=0.001, sub_feature=1)
# m1=RandomForestClassifier(100,max_depth=6,)
x1 = x.iloc[i1, :].values
y1 = y.iloc[i1].values
x2 = x.iloc[i2, :].values
y2 = y.iloc[i2].values

m1.fit(x1, y1, )

yp1 = m1.predict_proba(x1)[:, 1]
yp2 = m1.predict_proba(x2)[:, 1]

s1 = roc_auc_score(y2, yp2)
s2 = roc_auc_score(y1, yp1)

print(s1, s2)
# feature=x
# label=y
# pd.qcut(
# )

# algos.rank(x.x_1,pct=True)
