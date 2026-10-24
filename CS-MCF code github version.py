from __future__ import division, print_function
import numpy as np
import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p
from scipy import optimize
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, f1_score, recall_score, precision_score
import copy
import math
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import lap_score
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import trace_ratio
from skfeature.function.sparse_learning_based import ll_l21
from skfeature.function.sparse_learning_based import ls_l21
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.sparse_learning_based import NDFS
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.sparse_learning_based import UDFS
from skfeature.function.statistical_based import CFS
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.statistical_based import t_score
from skfeature.function.streaming import alpha_investing
from skfeature.function.wrapper import decision_tree_backward
from skfeature.function.wrapper import decision_tree_forward
from skfeature.function.wrapper import svm_backward
from skfeature.function.wrapper import svm_forward
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking, construct_label_matrix_pan
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.over_sampling import ADASYN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



class MCDF_CLF_CF():
    def __init__(self,
                 FeatureSelector=[
                     "alpha_investing",
                     "ASFS",
                     "DISR",  # 3
                     "f_score",  # 23
                     "fisher_score",  # 10
                     "gini_index",  # 24
                     "ICAP",  # 5
                     "JMI",  # 6
                     "lap_score",  # 11
                     "ll_l21",  # 15a
                     "ls_l21",  # 16
                     "MCFS",  # 17
                     "NDFS",  # 18
                     "reliefF",  # 12
                     #        "SPEC",  # 13
                     # "t_score",  # 25 Binary
                     "trace_ratio",  # 14
                     "RFS",  # 19  #slow
                     "UDFS",  # 20

                     #     "CFS",  # 21  #slow
                     # "chi_square",  # 22
                     #     "decision_tree_backward",  # 27
                     #     "decision_tree_forward",  # 28
                     #     "svm_backward",  # 29
                     #     "svm_forward"  # 30
                 ],
                 classifiers={
                     "Logistic Regression": LogisticRegression(penalty='l2', random_state=None),
                     "KNN": KNeighborsClassifier(),
                     "SVM": SVC(probability=True, C=1),
                     # "Naive Bayes": GaussianNB(),
                     "Decision Tree": DecisionTreeClassifier(),
                     "Extra Trees": ExtraTreesClassifier(random_state=None),
                     "Random Forest": RandomForestClassifier(n_estimators=50),
                     # "Bagging": BaggingClassifier(),
                     "AdaBoost": AdaBoostClassifier(),
                     "GradientBoosting": GradientBoostingClassifier(),
                     "XGBoost": XGBClassifier(),
                     "LightGBM": LGBMClassifier(),
                     "Catboost": CatBoostClassifier(logging_level='Silent'),

                     # "LDA": LinearDiscriminantAnalysis(),
                     # "QDA": QuadraticDiscriminantAnalysis(),
                     # "MLP": MLPClassifier()
                 },
                 Fusion_num=20,
                 feature_num=20,
                 cluster_rat=0.1,
                 class_weight=None,
                 nfold=5,
                 class_num=2,
                 corresponding_threshold=1
                 ):
        self.classifiers = classifiers
        self.FeatureSelector = FeatureSelector
        self.Fusion_num = Fusion_num
        self.feature_num = feature_num
        self.base = False
        self.heatmapdata = None
        self.feaselect_id = False

        self.perf_matrix = None
        self.Basic_clf_w = None
        self.nfold = nfold
        self.class_num = class_num
        self.cluster_ratio = cluster_rat

        if corresponding_threshold == 1:
            self.corresponding_filter = True
            self.corresponding_threshold = corresponding_threshold

        else:
            self.corresponding_filter = False
            self.corresponding_threshold = corresponding_threshold

    def test_external(self, X, y, X_test, y_test, center_name, sample_weight=1):
        self.class_num = len(np.unique(y))
        # fusion step: training basic clf, to get perf_matix and w
        if self.feaselect_id is False:
            self.feature_selection(X, y)

        if self.base == False:
            self.basic_clf_validation(X, y, X_test, y_test, center_name)

        self.fea_id, self.clf_id = self.chose_sorted(self.heatmapdata[0, :, :], self.Fusion_num)
        return self

    def fit(self, X, y):
        X_cls, y_cls = self.Batch_KNN(X, y, self.cluster_ratio, imp_state=0)
        # fusion step: training basic clf, to get perf_matix and w
        if self.feaselect_id is False:
            self.feature_selection(X_cls, y_cls)

        if self.base == False:
            self.basic_clf_train(X_cls, y_cls, sample_weight=None)

        self.fea_id, self.clf_id = self.chose_sorted(self.heatmapdata[0, :, :], self.Fusion_num)

        self.perf_matrix = np.zeros([7, self.Fusion_num])

        # ad = ADASYN(ratio=float(sample_weight))  # kind = ['regular', 'borderline1', 'borderline2', 'svm']
        # X_resampled, y_resampled = ad.fit_sample(X, y)

        X_resampled, y_resampled = X, y

        for i in range(self.Fusion_num):
            # print(i)
            temp_fea_i = self.feaselect_id[self.fea_id[i], :]
            temp_fea_i = np.array(temp_fea_i[np.where(temp_fea_i == temp_fea_i)], dtype=int)
            temp_X_train = X_resampled[:, temp_fea_i]

            # creat basic clf
            temp_clf = GridSearchCV(self.classifiers[list(self.classifiers.keys())[self.clf_id[i]]], cv=3,
                                    param_grid={})
            temp_clf.fit(temp_X_train, y_resampled)
            exec('self.basic_clf_' + str(i) + ' = ' + 'temp_clf')
            exec('self.perf_matrix[:,i]' + '=' + 'self.heatmapdata[:,self.fea_id[i], self.clf_id[i]]')

        self.Basic_clf_w = self.MCDM_func(self.perf_matrix.T, np.ones([7, 1])).astype(float)

        return self

    def predict_and_prob(self, X):
        fusion_prob = np.zeros([X.shape[0], 3])
        fusion_prob_bi = np.zeros([X.shape[0], self.class_num])
        for i in range(self.Fusion_num):
            temp_fea_i = self.feaselect_id[self.fea_id[i], :]
            temp_fea_i = np.array(temp_fea_i[np.where(temp_fea_i == temp_fea_i)], dtype=int)
            temp_X_train = X[:, temp_fea_i]
            exec('temp_prob' + '=' + 'self.basic_clf_' + str(i) + ' .predict_proba(temp_X_train)')
            exec('fusion_prob' + '+=' + 'temp_prob' + ' *' + 'self.Basic_clf_w[i]')
        fusion_prob_bi = np.array([fusion_prob[:, 0], 1 - fusion_prob[:, 0]]).T
        fusion_pre = np.argmax(fusion_prob_bi, 1)
        return fusion_pre, fusion_prob_bi

    def basic_clf_train(self, X, y, sample_weight=None):
        # to get AUC heatmap from all basic clf + feature_slct combinations
        save_path = 'C:/Users/HP/Desktop/CS-MCF pkg/CS-MCF/Heatmap_data/'
        data_name = 'IV_heatmapdata_' + str(self.feature_num) + '.npy'
        if not os.path.exists(save_path + data_name):
            self.heatmapdata = np.zeros([7, len(self.FeatureSelector), len(self.classifiers), self.nfold])
            temp_fold = 0

            skf = StratifiedKFold(n_splits=self.nfold, random_state=1, shuffle=True)
            for (train_id, test_id), temp_fold in zip(skf.split(X, y), range(self.nfold)):
                X_train = X[train_id, :]
                y_train = y[train_id]
                X_test = X[test_id, :]
                y_test = y[test_id]
                for i, name2 in zip(range(self.feaselect_id.shape[0]), self.FeatureSelector):
                    j = 0
                    idx = self.feaselect_id[i, :]
                    for name1, clf in self.classifiers.items():
                        print("now training: " + name2 + " " + name1)
                        AUCb, ACCb, SPEb, SENb, PPVb, NPVb, F1b = self.heat_map_perf(name1, idx, X_train,
                                                                                            y_train, X_test, y_test,
                                                                                            clf,
                                                                                            feanum=self.feature_num,
                                                                                            FLAG_DataBalance=1)
                        self.heatmapdata[:, i, j, temp_fold] = np.array(
                            [AUCb, ACCb, SPEb, SENb, PPVb, NPVb, F1b])
                        j += 1


            self.heatmapdata = np.mean(self.heatmapdata, axis=3)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path + data_name, self.heatmapdata, allow_pickle=True)
            d = 1  # wip
            self.base = True
            self.heatmapdata = self.heatmapdata
        else:
            self.base = True
            self.heatmapdata = np.load(save_path + data_name, allow_pickle=True)
        return self

    def basic_clf_validation(self, X, y, X_test, y_test, center_name):
        # to get AUC heatmap from all basic clf + feature_slct combinations
        save_path = 'C:/Users/HP/Desktop/CS-MCF pkg/CS-MCF/Heatmap_data/'
        data_name = center_name + '_heatmapdata_' + str(self.feature_num) + '.npy'

        if not os.path.exists(save_path + data_name):

            self.heatmapdata = np.zeros([5, len(self.FeatureSelector), len(self.classifiers)])
            temp_fold = 0

            skf = StratifiedKFold(n_splits=self.nfold, random_state=1, shuffle=True)

            for i, name2 in zip(range(self.feaselect_id.shape[0]), self.FeatureSelector):
                j = 0
                idx = self.feaselect_id[i, :]
                for name1, clf in self.classifiers.items():
                    print("now training E: " + name2 + " " + name1)


                    AUCb, ACCb, SPEb, SENb, PPVb, NPVb, F1b = self.heat_map_perf(name1, idx, X, y, X_test, y_test, clf, feanum=self.feature_num, FLAG_DataBalance=1)
                    self.heatmapdata[:, i, j] = np.array([AUCb, ACCb, SPEb, SENb, PPVb, NPVb, F1b])
                    j += 1

            np.save(save_path + data_name, self.heatmapdata, allow_pickle=True)
            self.base = True
        else:
            self.base = True
            self.heatmapdata = np.load(save_path + data_name, allow_pickle=True)
        return self

    def feature_selection(self, X, y):
        save_path = 'C:/Users/HP/Desktop/CS-MCF pkg/CS-MCF/Feature saving/'

        for name2 in self.FeatureSelector:
            middle_save_path = save_path + '/' + name2 + '/'
            if not os.path.exists(middle_save_path):
                idx = self.featureid_saving(name2 + "_main", X, y, feanum=self.feature_num)
                os.makedirs(middle_save_path)
                np.save(middle_save_path + 'idx.npy', idx)

        file_name = os.listdir(save_path)
        self.feaselect_id = np.full([len(self.FeatureSelector), self.feature_num], np.nan, dtype=object)
        for i in range(len(file_name)):
            try:
                self.feaselect_id[i, :] = np.load(save_path + file_name[i] + '/idx.npy', allow_pickle=True)[
                                          :self.feature_num]
            except:
                temp_inp_num = len(np.load(save_path + file_name[i] + '/idx.npy', allow_pickle=True)[:self.feature_num])
                self.feaselect_id[i, :temp_inp_num] = np.load(save_path + file_name[i] + '/idx.npy', allow_pickle=True)
        # self.feaselect_id = self.feaselect_id.astype(int)
        d = 1

        return self

    def MCDM_func(self, dm, weights):
        '''输入一个attribute * criteria 的矩阵'''
        M = dm ** 2
        decison_metrix = dm.copy()
        A = False
        if dm.shape[0] < 3:
            for i in range(decison_metrix.shape[1]):
                decison_metrix[:, i] = decison_metrix[:, i] / math.sqrt(sum(M[:, i]))
            for i in range(decison_metrix.shape[1]):
                decison_metrix[:, i] = decison_metrix[:, i] * weights[i]

            al_weights = np.sum(decison_metrix, axis=1) / sum(sum(decison_metrix))
        else:
            '''TOPSIS_tp'''
            '''regular the decision metrix'''
            for i in range(decison_metrix.shape[1]):
                decison_metrix[:, i] = decison_metrix[:, i] / float(math.sqrt(sum(M[:, i])))

            '''regular the weights'''
            '''calcute the weighted normalised decisoin metrix'''
            for i in range(decison_metrix.shape[1]):
                decison_metrix[:, i] = decison_metrix[:, i] * weights[i]
            '''Determine the worst alternative and the best alternative'''
            best = np.zeros((1, decison_metrix.shape[1]))
            worst = np.zeros((1, decison_metrix.shape[1]))
            for i in range(decison_metrix.shape[1]):
                best[0, i] = max(decison_metrix[:, i])
                worst[0, i] = min(decison_metrix[:, i])
            '''Calculate the L2-distance between the target alternative and the worst condition'''
            Sw = np.zeros((decison_metrix.shape[0], 1))
            for i in range(decison_metrix.shape[0]):
                Sw[i] = math.sqrt(sum(sum((decison_metrix[i, :] - worst) ** 2)))
                if Sw[i] == 0:
                    Sw[i] = 0.1
            '''Calculate the L2-distance between the target alternative and the best condition'''
            Sb = np.zeros((decison_metrix.shape[0], 1))
            for i in range(decison_metrix.shape[0]):
                Sb[i] = math.sqrt(sum(sum((decison_metrix[i, :] - best) ** 2)))
            S = Sw + Sb
            al_weights = Sw / S

        classifier_w = al_weights / np.sum(al_weights)
        # y_prediction = np.zeros((n_sample, 1))
        # y_proba_final = np.zeros((n_sample, len(cls)))
        #
        # threshold_weights = [1, c1_weights]
        # for i in range(n_sample):
        #     y_proba_temp = y_proba_matrix[i]
        #     y_proba_final[i, :] = np.dot(classifier_w.T, y_proba_temp) * threshold_weights
        #     # temp = np.dot(classifier_w.T, y_proba_temp)
        #     y_prediction[i] = np.where(y_proba_final[i, :] == np.max(y_proba_final[i, :]))[0][0]
        return classifier_w

    def chose_sorted(self, heatmap, top_number):
        sortlist = np.argsort(heatmap, axis=None)[::-1][:top_number]
        i_num = [i // heatmap.shape[1] for i in sortlist]
        j_num = [j % heatmap.shape[1] for j in sortlist]

        return i_num, j_num

    def featureid_saving(self, operator, X_train, y_train, **kwargs):

        # information_theoretical_based feature selection
        if 'feanum' in kwargs.keys():
            # refer_num_fea = kwargs['feanum']
            num_fea = kwargs['feanum']
        
        ###################################################################################

        # obtain the index of selected features on the training set
        if operator == "CIFE_main":  # 1
            idx, _, _ = CIFE.cife(X_train, y_train, n_selected_features=num_fea)
        elif operator == "CMIM_main":  # 2
            idx, _, _ = CMIM.cmim(X_train, y_train, n_selected_features=num_fea)
        elif operator == "DISR_main":  # 3
            idx, _, _ = DISR.disr(X_train, y_train, n_selected_features=num_fea)
        elif operator == "FCBF_main":  # 4
            idx, _ = FCBF.fcbf(X_train, y_train, delta=0)
        elif operator == "ICAP_main":  # 5
            idx, _, _ = ICAP.icap(X_train, y_train, n_selected_features=num_fea)
        elif operator == "JMI_main":  # 6
            idx, _, _ = JMI.jmi(X_train, y_train, n_selected_features=num_fea)
        elif operator == "MIFS_main":  # 7
            idx, _, _ = MIFS.mifs(X_train, y_train, n_selected_features=num_fea)
        elif operator == "MIM_main":  # 8
            idx, _, _ = MIM.mim(X_train, y_train, n_selected_features=num_fea)
        elif operator == "MRMR_main":  # 9
            idx, _, _ = MRMR.mrmr(X_train, y_train, n_selected_features=num_fea)
        elif operator == "fisher_score_main":  # 10
            score = fisher_score.fisher_score(X_train, y_train)  # the larger the score, the more important the
            idx = fisher_score.feature_ranking(score)
        elif operator == "lap_score_main":  # 11
            kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
            W = construct_W.construct_W(X_train, **kwargs_W)
            score = lap_score.lap_score(X_train, W=W)
            idx = lap_score.feature_ranking(score)
        elif operator == "reliefF_main":  # 12
            score = reliefF.reliefF(X_train, y_train)
            # rank features in descending order according to score
            idx = reliefF.feature_ranking(score)
        elif operator == "SPEC_main":  # 13
            # specify the second ranking function which uses all except the 1st eigenvalue
            kwargs_style = {'style': 0}
            # obtain the scores of features
            score = SPEC.spec(X_train, **kwargs_style)
            # sort the feature scores in an descending order according to the feature scores
            idx = SPEC.feature_ranking(score, **kwargs_style)
        elif operator == "trace_ratio_main":  # 14
            idx, _, _ = trace_ratio.trace_ratio(X_train, y_train, num_fea, style='fisher')
        elif operator == "ll_l21_main":  # 15
            Y_train = construct_label_matrix_pan(y_train)
            Weight, obj, value_gamma = ll_l21.proximal_gradient_descent(X_train, Y_train, 0.1, verbose=False)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "ls_l21_main":  # 16
            # obtain the feature weight matrix
            Y_train = construct_label_matrix_pan(y_train)
            Weight, obj, value_gamma = ls_l21.proximal_gradient_descent(X_train, Y_train, 0.1, verbose=False)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "MCFS_main":  # 17
            # construct affinity matrix
            kwargs_W = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
            W = construct_W.construct_W(X_train, **kwargs_W)
            num_cluster = 2  # specify the number of clusters, it is usually set as the number of classes in the ground truth
            # obtain the feature weight matrix
            Weight = MCFS.mcfs(X_train, n_selected_features=num_fea, W=W, n_clusters=num_cluster)
            # sort the feature scores in an ascending order according to the feature scores
            idx = MCFS.feature_ranking(Weight)
        elif operator == "NDFS_main":  # 18
            # construct affinity matrix
            kwargs_W = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
            W = construct_W.construct_W(X_train, **kwargs_W)
            # obtain the feature weight matrix
            Weight = NDFS.ndfs(X_train, W=W, n_clusters=3)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "RFS_main":  # 19
            # obtain the feature weight matrix
            Y_train = construct_label_matrix_pan(y_train)
            Weight = RFS.rfs(X_train, Y_train, gamma=0.1)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "UDFS_main":  # 20
            # obtain the feature weight matrix
            Weight = UDFS.udfs(X_train, gamma=0.1, n_clusters=3)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "CFS_main":  # 21
            idx = CFS.cfs(X_train, y_train)
        elif operator == "chi_square_main":  # 22
            # obtain the chi-square score of each feature
            score = chi_square.chi_square(X_train, y_train)
            # rank features in descending order according to score
            idx = chi_square.feature_ranking(score)
        elif operator == "f_score_main":  # 23
            # obtain the chi-square score of each feature
            score = f_score.f_score(X_train, y_train)
            # rank features in descending order according to score
            idx = f_score.feature_ranking(score)
        elif operator == "gini_index_main":  # 24
            # obtain the chi-square score of each feature
            score = gini_index.gini_index(X_train, y_train)
            # rank features in descending order according to score
            idx = gini_index.feature_ranking(score)
        elif operator == "t_score_main":  # 25
            # obtain the chi-square score of each feature
            score = t_score.t_score(X_train, y_train)
            # rank features in descending order according to score
            idx = t_score.feature_ranking(score)
        elif operator == "alpha_investing_main":  # 26
            idx = alpha_investing.alpha_investing(X_train, y_train, 0.5, 0.5)
        elif operator == "decision_tree_backward_main":  # 27
            idx = decision_tree_backward.decision_tree_backward(X_train, y_train, num_fea)
        elif operator == "decision_tree_forward_main":  # 28
            idx = decision_tree_forward.decision_tree_forward(X_train, y_train, num_fea)
        elif operator == "svm_backward_main":  # 29
            idx = svm_backward.svm_backward(X_train, y_train, num_fea)
        elif operator == "svm_forward_main":  # 30
            idx = svm_forward.svm_forward(X_train, y_train, num_fea)
        elif operator == "_PSC_main":
            self.filter_list = [3, 5, 37, 38, 33, 9, 30, 10, 25, 26,
                                18, 15, 16, 17, 20, 69, 70, 50, 68, 72]
            idx = np.array(self.filter_list)

        return idx

    def get_class_matrix(self, y_in):
        class_num = len(np.unique(y_in))
        y_out = np.zeros([len(y_in), class_num])
        for i in range(class_num):
            y_out[np.where(y_in == i), i] = 1
        return y_out

    def cal_sen_spe(self, y_actual, y_pred):
        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0

        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == 1:
                TP += 1
        for i in range(len(y_pred)):
            if y_actual[i] == 0 and y_actual[i] != y_pred[i]:
                FP += 1
        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == 0:
                TN += 1
        for i in range(len(y_pred)):
            if y_actual[i] == 1 and y_actual[i] != y_pred[i]:
                FN += 1

        if ((y_actual == y_pred).all() and np.sum(y_actual) == len(y_actual)) or (
                (y_actual == y_pred).all() and np.sum(y_actual) == 0):  # all 1s or 0s
            Sensitivity = 1.0
            Specificity = 1.0
            NPV = 1.0
        elif (y_actual == y_pred).any() == False:
            Sensitivity = 0.0
            Specificity = 0.0
            NPV = 0.0
        else:
            Sensitivity = float(TP) / (float(TP + FN) + 0.000000001)
            Specificity = float(TN) / (float(TN + FP) + 0.000000001)
            NPV = float(TN) / (float(TN + FN) + 0.000000001)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        return Sensitivity, Specificity

    def Batch_KNN(self, X, y, cluster_ratio , imp_state=0):
        X_Cluster = []
        y_Cluster = []

        for temp_class in np.unique(y):
            X_temp = X[np.where(y == temp_class)[0], :]
            minibatch_kmeans = MiniBatchKMeans(n_clusters=round(X_temp.shape[0] * cluster_ratio),
                                               batch_size=1024, random_state=imp_state)
            minibatch_kmeans.fit_transform(X_temp)
            X_temp_Cluster = minibatch_kmeans.cluster_centers_
            y_temp_Cluster = np.full([X_temp_Cluster.shape[0], 1], temp_class)

            if len(y_Cluster) == 0:
                y_Cluster = y_temp_Cluster
                X_Cluster = X_temp_Cluster
            else:
                y_Cluster = np.append(y_Cluster, y_temp_Cluster)
                X_Cluster = np.vstack([X_Cluster, X_temp_Cluster])
        return X_Cluster, y_Cluster

    def heat_map_perf(self, name1, idx, X_train, y_train, X_test, y_test, clf, **kwargs):
        # information_theoretical_based feature selection
        if 'feanum' in kwargs.keys():
            # refer_num_fea = kwargs['feanum']
            num_fea = kwargs['feanum']

        if 'FLAG_DataBalance' in kwargs.keys():
            FLAG_DataBalance = kwargs['FLAG_DataBalance']
        else:
            FLAG_DataBalance = False
        ############## test ##############
        featuresidex = []
        #########################################################################

        # 原本的idx
        idx = np.array(idx[np.where(idx == idx)], dtype=int)
        X_train_selected = X_train[:, idx[0:min(num_fea, len(idx))]]
        X_test_selected = X_test[:, idx[0:min(num_fea, len(idx))]]
        # 存储选择出来的特征的idx
        featuresidex.append(idx[0:min(num_fea, len(idx))])

        ADA = ADASYN()
        try:
            X_selected_resampled, y_selected_resampled = ADA.fit_resample(X_train_selected, y_train)
        except:
            X_selected_resampled, y_selected_resampled = X_train_selected, y_train

        # 网格搜索
        if name1 == "Logistic Regression":
            # Grid_Dict = {'C': range(1, 10), 'solver':['liblinear','lbfgs']}
            Grid_Dict = {}
        elif name1 == "SVM":
            Grid_Dict = {}  # 'kernel':['rbf','linear','poly'],'max_iter':[50,100]
        elif name1 == "Naive Bayes":
            Grid_Dict = {}
        elif name1 == "KNN":
            # Grid_Dict = {'n_neighbors':[5, 7]} # 'n_neighbors': range(2, 15), 'p':range(1, 10)
            Grid_Dict = {}
        elif name1 == "Decision Tree":
            # Grid_Dict = {'random_state': range(0, 15)}
            Grid_Dict = {}
        elif name1 == "Extra Trees":
            # Grid_Dict = {'min_samples_split': range(2, 6)}
            Grid_Dict = {}
        elif name1 == "Bagging":
            Grid_Dict = {}
        elif name1 == "Random Forest":
            Grid_Dict = {}  # 速度很慢，'random_state': range(0, 10)
        elif name1 == "AdaBoost":
            Grid_Dict = {}  # 速度很慢，'learning_rate': [0.1, 1, 10]
        elif name1 == "GradientBoosting":
            Grid_Dict = {}  # 'learning_rate': [0.01, 0.1, 1] ，'random_state': range(0, 10)# 速度很慢
        elif name1 == "XGBoost":
            Grid_Dict = {}
        elif name1 == "LightGBM":
            Grid_Dict = {}
        elif name1 == "Catboost":
            Grid_Dict = {}
        elif name1 == "ExtraTrees":
            Grid_Dict = {}
        elif name1 == "LDA":
            Grid_Dict = {}
        elif name1 == "QDA":
            Grid_Dict = {}
        elif name1 == "MLP":
            Grid_Dict = {}

        Classifier = GridSearchCV(clf, cv=3, param_grid=Grid_Dict)
        Classifier.fit(X_selected_resampled, y_selected_resampled)

        ############### test ################
        y_predict = Classifier.predict(X_test_selected)
        y_predict_binary = copy.deepcopy(y_predict)
        y_predict_binary[np.where(y_predict >= 1)] = 1
        y_test_binary = copy.deepcopy(y_test)
        y_test_binary[np.where(y_test >= 1)] = 1

        y_predict_prob = Classifier.predict_proba(X_test_selected)
        fpr, tpr, thresholds = roc_curve(y_test_binary, y_predict_prob[:, 1] + y_predict_prob[:, 2])
        auc_o_binary = auc(fpr, tpr)
        acc_o_binary = accuracy_score(y_test_binary, y_predict_binary)
        sen_o_binary, spe_o_binary = self.cal_sen_spe(y_test_binary, y_predict_binary)
        pre_binary = precision_score(y_test_binary, y_predict_binary, average='binary')
        rec_binary = recall_score(y_test_binary, y_predict_binary, average='binary')
        f1score_binary = f1_score(y_test_binary, y_predict_binary, average='binary')

        return auc_o_binary, acc_o_binary, spe_o_binary, sen_o_binary, pre_binary, rec_binary, f1score_binary


class Data_precessing():
    def __init__(self,
                 padding_method = {# 'MEAN' : SimpleImputer(missing_values=np.nan, strategy='mean')
                                   # 'KNN' : KNNImputer(n_neighbors=5, weights="uniform")
                                   'MFI' : IterativeImputer(max_iter=300, random_state=0,tol=1e-6)},


                 ):
        self.pad_method = padding_method[list(padding_method.keys())[-1]]
        self.N_quantiles = 10


        self.basic_test= None
        self.combining_test = None
        self.boxcox_para = None
        self.datapad_pro = False
        self.standard_para = None
        self.TM_List = [2, 3, 4, 5, 6]
        self.CF_List = [32, 33, 34, 35, 36, 37]
        self.TMCF_list = [2, 3, 4, 5, 6, 32, 33, 34, 35, 36, 37]
        self.basic_test_id = 38
        self.Index_head = ['NLR','dNLR','PLR','LMR','NMLR',
                           'NLPR','SIRI','SII','AISI','NHL',
                           'NPR','PAR','PNI','HALP','NPAR',
                           'ALBI','PALBI','AGR','HPR','WHR',
                           'NAR','NMR','PWR','PMR','AGMR',
                           'dMLR','GLR','ALRI','MLPR','APRI',
                           'AFR','DAR','DFR','FPR','LAR',
                           'AAPR']

    def fit(self, X):
        self.org_data = X
        self.data_pad_and_comb(X)
        self.get_boxcox_para(self.pad_data)
        self.get_standard_para(self.boxcox_data)
        return self

    def data_pad_and_comb(self, X):
        self.distfit_para = [[] for i in range(X.shape[1])]
        self.distfit_para[0] = preprocessing.QuantileTransformer(random_state=0, n_quantiles=self.N_quantiles)
        for i in range(1, self.basic_test_id):
            temp_array = X[:, i][np.where(X[:, i] == X[:, i])]
            self.distfit_para[i] = preprocessing.QuantileTransformer(random_state=0, n_quantiles=self.N_quantiles)
            self.distfit_para[i].fit(temp_array.reshape(-1, 1))


        imp = self.pad_method
        X_pad = imp.fit_transform(X[:, :self.basic_test_id])
        for i in range(1, self.basic_test_id):
            if len(np.where(self.org_data[i, :] == np.nan)) < len(self.org_data[i, :]):
                temp_max = max(self.org_data[:, i])
                X_pad[np.where(X_pad[:, i] > temp_max), i] = temp_max
                temp_min = min(self.org_data[:, i])
                X_pad[np.where(X_pad[:, i] < temp_min), i] = temp_min
        self.basic_data = X_pad
        self.comb_data = self.data_combing_convert(X_pad)
        self.pad_data = np.hstack([self.basic_data, self.comb_data])
        self.datapad_pro = True
        return self

    def get_boxcox_para(self,X):
        self.boxcox_para = np.zeros([1, X.shape[1]])
        X_copy =copy.deepcopy(X)
        for i in range(1, X.shape[1]):
            temp_array = X[:, i]
            temp_array -= min(temp_array)
            temp_array += 1
            for temp_bound in range(1, 4):
                temp_para = boxcox_normmax(temp_array, optimizer=self.temp_optimizer)
                boxcox_array = boxcox1p(temp_array, temp_para)
                if len(np.unique(np.round(boxcox_array, 6))) > 50:
                    X_copy[:, i] = boxcox_array
                    self.boxcox_para[0, i] = temp_para
        self.boxcox_data = X_copy
        return self

    def get_standard_para(self,X):
        self.standard_para = preprocessing.StandardScaler().fit(X[:, 1:])
        self.standard_data = np.hstack([np.array([X[:, 0]]).T, self.standard_para.transform(X[:, 1:])])
        return self

    def data_combing_convert(self, Inp):
        Out_data = np.full([Inp.shape[0], len(self.Index_head)], np.nan).astype(float)
        for i in range(len(self.Index_head)):
            if self.Index_head[i] == 'NLR':
                Out_data[:, i] = Inp[:, 15] / Inp[:, 16]
            elif self.Index_head[i] == 'dNLR':
                Out_data[:, i] = Inp[:, 15] / (Inp[:, 14] - Inp[:, 16])
            elif self.Index_head[i] == 'PLR':
                Out_data[:, i] = Inp[:, 20] / Inp[:, 16]
            elif self.Index_head[i] == 'LMR':
                Out_data[:, i] = Inp[:, 16] / Inp[:, 17]
            elif self.Index_head[i] == 'NMLR':
                Out_data[:, i] = (Inp[:, 15] + Inp[:, 17]) / Inp[:, 16]
            elif self.Index_head[i] == 'NLPR':
                Out_data[:, i] = Inp[:, 15] / (Inp[:, 16] * Inp[:, 20])
            elif self.Index_head[i] == 'SIRI':
                Out_data[:, i] = (Inp[:, 15] * Inp[:, 17]) / Inp[:, 16]
            elif self.Index_head[i] == 'SII':
                Out_data[:, i] = (Inp[:, 15] * Inp[:, 20]) / Inp[:, 16]
            elif self.Index_head[i] == 'AISI':
                Out_data[:, i] = (Inp[:, 15] * Inp[:, 17] * Inp[:, 20]) / Inp[:, 16]
            elif self.Index_head[i] == 'NHL':
                Out_data[:, i] = Inp[:, 15] / (Inp[:, 18] * Inp[:, 16])
            elif self.Index_head[i] == 'NPR':
                Out_data[:, i] = Inp[:, 15] / Inp[:, 20]
            elif self.Index_head[i] == 'PAR':
                Out_data[:, i] = Inp[:, 20] / Inp[:, 9]
            elif self.Index_head[i] == 'PNI':
                Out_data[:, i] = Inp[:, 9] + 5 * Inp[:, 16]
            elif self.Index_head[i] == 'HALP':
                Out_data[:, i] = (Inp[:, 9] * Inp[:, 16] * Inp[:, 18]) / Inp[:, 20]
            elif self.Index_head[i] == 'NPAR':
                Out_data[:, i] = (Inp[:, 15] / Inp[:, 14] * 100) / (Inp[:, 9] / 10)
            elif self.Index_head[i] == 'ALBI':
                Out_data[:, i] = np.log10(Inp[:, 11].astype(float) * 0.66) - 0.0852 * Inp[:, 9]
            elif self.Index_head[i] == 'PALBI':
                Out_data[:, i] = 2.02 * np.log10(Inp[:, 11].astype(float)) - 0.37 * np.square(
                    np.log10(Inp[:, 11].astype(float))) - 0.04 * Inp[:, 9] - 3.48 * np.log10(
                    Inp[:, 20].astype(float)) + 1.01 * np.square(np.log10(Inp[:, 20].astype(float)))
            elif self.Index_head[i] == 'AGR':
                Out_data[:, i] = Inp[:, 9] / Inp[:, 10]
            elif self.Index_head[i] == 'HPR':
                Out_data[:, i] = Inp[:, 18] / Inp[:, 20]
            elif self.Index_head[i] == 'WHR':
                Out_data[:, i] = Inp[:, 14] / Inp[:, 18]
            elif self.Index_head[i] == 'NAR':
                Out_data[:, i] = Inp[:, 15] / Inp[:, 9]
            elif self.Index_head[i] == 'NMR':
                Out_data[:, i] = Inp[:, 15] / Inp[:, 17]
            elif self.Index_head[i] == 'PWR':
                Out_data[:, i] = Inp[:, 20] / Inp[:, 14]
            elif self.Index_head[i] == 'PMR':
                Out_data[:, i] = Inp[:, 20] / Inp[:, 17]
            elif self.Index_head[i] == 'AGMR':
                Out_data[:, i] = (Inp[:, 9] / Inp[:, 10]) / Inp[:, 17]
            elif self.Index_head[i] == 'dMLR':
                Out_data[:, i] = Inp[:, 17] / (Inp[:, 14] - Inp[:, 15] + 10 ** (-5))
            elif self.Index_head[i] == 'GLR':
                Out_data[:, i] = Inp[:, 21] / Inp[:, 16]
            elif self.Index_head[i] == 'ALRI':
                Out_data[:, i] = Inp[:, 8] / Inp[:, 16]
            elif self.Index_head[i] == 'MLPR':
                Out_data[:, i] = 1000 * Inp[:, 17] / (Inp[:, 16] * Inp[:, 20])
            elif self.Index_head[i] == 'APRI':
                Out_data[:, i] = Inp[:, 8] / Inp[:, 20]
            elif self.Index_head[i] == 'AFR':
                Out_data[:, i] = Inp[:, 9] / Inp[:, 36]
            elif self.Index_head[i] == 'DAR':
                Out_data[:, i] = Inp[:, 37] / Inp[:, 9]
            elif self.Index_head[i] == 'DFR':
                Out_data[:, i] = Inp[:, 37] / Inp[:, 36]
            elif self.Index_head[i] == 'FPR':
                Out_data[:, i] = Inp[:, 36] / Inp[:, 20]
            elif self.Index_head[i] == 'LAR':
                Out_data[:, i] = Inp[:, 30] / Inp[:, 9]
            elif self.Index_head[i] == 'AAPR':
                Out_data[:, i] = Inp[:, 9] / Inp[:, 29]

        return Out_data

    def transform(self, X, imp=True, boxcox=True, standered=True, distfit=True):

        if (imp == True) & (boxcox == True) & (standered == True):
            if distfit == True:
                temp_empty_list = []
                for i in range(1, self.basic_test_id):
                    if len(np.where(X[:, i] != X[:, i])[0]) < X.shape[0]:
                        quantile_transformer_temp = preprocessing.QuantileTransformer(random_state=0,
                                                                                      n_quantiles=self.N_quantiles)
                        quantile_transformer_temp.fit(X[:, i][np.where(X[:, i] == X[:, i])].reshape(-1, 1))
                        X[np.where(X[:, i] == X[:, i])[
                            0], i] = self.distfit_para[i].inverse_transform(
                            quantile_transformer_temp.transform(X[np.where(X[:, i] == X[:, i])[0], i].reshape(-1, 1)))[
                                     :, 0]
                    else:
                        temp_empty_list.append(i)
            data_w4_pad = np.vstack([self.pad_data, X])
            data_org = copy.deepcopy(X[:, :self.basic_test_id])
            mask_id = np.where(data_org == data_org)
            imp = self.pad_method
            pad_X = imp.fit_transform(data_w4_pad[:, :self.basic_test_id])[-X.shape[0]:, :]
            pad_X[mask_id] = data_org[mask_id]
            nan_id = np.where(X != X)

            if distfit == True:
                for i in temp_empty_list:
                    quantile_transformer_temp = preprocessing.QuantileTransformer(random_state=0,
                                                                                  n_quantiles=self.N_quantiles)
                    quantile_transformer_temp.fit(pad_X[:, i].reshape(-1, 1))
                    pad_X[:, i] = self.distfit_para[i].inverse_transform(
                        quantile_transformer_temp.transform(pad_X[:, i].reshape(-1, 1)))[:, 0]

            pad_X[np.where(pad_X > np.max(self.pad_data[:, :self.basic_test_id], axis=0))] = \
                np.max(self.pad_data[:, :self.basic_test_id], axis=0)[
                    np.where(pad_X > np.max(self.pad_data[:, :self.basic_test_id], axis=0))[1]]
            pad_X[np.where(pad_X < np.min(self.pad_data[:, :self.basic_test_id], axis=0))] = \
                np.min(self.pad_data[:, :self.basic_test_id], axis=0)[
                    np.where(pad_X < np.min(self.pad_data[:, :self.basic_test_id], axis=0))[1]]
            combine_X = self.data_combing_convert(pad_X)
            pad_X = np.hstack([pad_X, combine_X])

            for i in range(1, pad_X.shape[1]):
                if i == 2:
                    d = 1
                pad_X[:, i] -= np.min(pad_X[:, i])
                # pad_X[:, i] -= np.min(self.pad_data[:, i])

                pad_X[:, i] += 1
                pad_X[:, i] = boxcox1p(pad_X[:, i], self.boxcox_para[0, i])

            pad_X[:, 1:] = self.standard_para.transform(pad_X[:, 1:])

            ###260609  padding with std data ####
            pad_X[nan_id] = np.nan
            imp = IterativeImputer(max_iter=100, random_state=42, tol=1e-6)
            data_w4_pad_std = np.vstack([self.standard_data, pad_X])
            pad_X[:, 1:] = imp.fit_transform(data_w4_pad_std)[-X.shape[0]:, 1:]
            pad_X[np.where(pad_X > np.max(self.standard_data, axis=0))] = np.max(self.standard_data, axis=0)[
                np.where(pad_X > np.max(self.standard_data, axis=0))[1]]
            pad_X[np.where(pad_X < np.min(self.standard_data, axis=0))] = np.min(self.standard_data, axis=0)[
                np.where(pad_X < np.min(self.standard_data, axis=0))[1]]

        if imp == True & (boxcox == False) & (standered == False):
            data_w4_pad = np.vstack([self.standard_data, X])
            imp = self.pad_method
            pad_X = imp.fit_transform(data_w4_pad)[-X.shape[0]:, :]
            # pad_X[np.where(pad_X < 0)] = 0
            pad_X[np.where(pad_X > np.max(self.standard_data, axis=0))] = np.max(self.standard_data, axis=0)[
                np.where(pad_X > np.max(self.standard_data, axis=0))[1]]
            pad_X[np.where(pad_X < np.min(self.standard_data, axis=0))] = np.min(self.standard_data, axis=0)[
                np.where(pad_X < np.min(self.standard_data, axis=0))[1]]

        return pad_X

    def fittransform(self, X, imp=True, boxcox=True, standered=True, distfit=True):
        self.fit(X)

        if (imp == True) & (boxcox == True) & (standered == True):
            if distfit == True:
                temp_empty_list = []
                for i in range(1, self.basic_test_id):
                    if len(np.where(X[:, i] != X[:, i])[0]) < X.shape[0]:
                        quantile_transformer_temp = preprocessing.QuantileTransformer(random_state=0,
                                                                                      n_quantiles=self.N_quantiles)
                        quantile_transformer_temp.fit(X[:, i][np.where(X[:, i] == X[:, i])].reshape(-1, 1))
                        X[np.where(X[:, i] == X[:, i])[
                            0], i] = self.distfit_para[i].inverse_transform(
                            quantile_transformer_temp.transform(X[np.where(X[:, i] == X[:, i])[0], i].reshape(-1, 1)))[
                                     :, 0]
                    else:
                        temp_empty_list.append(i)
            data_w4_pad = np.vstack([self.pad_data, X])
            data_org = copy.deepcopy(X[:, :self.basic_test_id])
            mask_id = np.where(data_org == data_org)
            imp = self.pad_method
            pad_X = imp.fit_transform(data_w4_pad[:, :self.basic_test_id])[-X.shape[0]:, :]
            pad_X[mask_id] = data_org[mask_id]

            if distfit == True:
                for i in temp_empty_list:
                    quantile_transformer_temp = preprocessing.QuantileTransformer(random_state=0,
                                                                                  n_quantiles=self.N_quantiles)
                    quantile_transformer_temp.fit(pad_X[:, i].reshape(-1, 1))
                    pad_X[:, i] = self.distfit_para[i].inverse_transform(
                        quantile_transformer_temp.transform(pad_X[:, i].reshape(-1, 1)))[:, 0]

            pad_X[np.where(pad_X > np.max(self.pad_data[:, :self.basic_test_id], axis=0))] = \
                np.max(self.pad_data[:, :self.basic_test_id], axis=0)[
                    np.where(pad_X > np.max(self.pad_data[:, :self.basic_test_id], axis=0))[1]]
            pad_X[np.where(pad_X < np.min(self.pad_data[:, :self.basic_test_id], axis=0))] = \
                np.min(self.pad_data[:, :self.basic_test_id], axis=0)[
                    np.where(pad_X < np.min(self.pad_data[:, :self.basic_test_id], axis=0))[1]]
            combine_X = self.data_combing_convert(pad_X)
            pad_X = np.hstack([pad_X, combine_X])

            for i in range(1, pad_X.shape[1]):
                if i == 2:
                    d = 1
                pad_X[:, i] -= np.min(pad_X[:, i])
                # pad_X[:, i] -= np.min(self.pad_data[:, i])

                pad_X[:, i] += 1
                pad_X[:, i] = boxcox1p(pad_X[:, i], self.boxcox_para[0, i])

            pad_X[np.where(pad_X > np.max(self.boxcox_data, axis=0))] = \
                np.max(self.boxcox_data, axis=0)[np.where(pad_X > np.max(self.boxcox_data, axis=0))[1]]
            pad_X[np.where(pad_X < np.min(self.boxcox_data, axis=0))] = \
                np.min(self.boxcox_data, axis=0)[np.where(pad_X < np.min(self.boxcox_data, axis=0))[1]]

            pad_X[:, 1:] = self.standard_para.transform(pad_X[:, 1:])

            # Singular value in standard_padX
            pad_X[np.where(pad_X > 5)] = 0
            # pad_X[np.where(pad_X < np.min(self.standard_data, axis=0))] = 0

        if imp == True & (boxcox == False) & (standered == False):
            data_w4_pad = np.vstack([self.standard_data, X])
            imp = self.pad_method
            pad_X = imp.fit_transform(data_w4_pad)[-X.shape[0]:, :]
            # pad_X[np.where(pad_X < 0)] = 0
            pad_X[np.where(pad_X > np.max(self.standard_data, axis=0))] = np.max(self.standard_data, axis=0)[
                np.where(pad_X > np.max(self.standard_data, axis=0))[1]]
            pad_X[np.where(pad_X < np.min(self.standard_data, axis=0))] = np.min(self.standard_data, axis=0)[
                np.where(pad_X < np.min(self.standard_data, axis=0))[1]]
        return pad_X

    def temp_optimizer(sefl,fun):
        return optimize.minimize_scalar(fun, bounds=(-2, 5),
                                        method="bounded", options={'xatol': 1e-12})






if __name__ == '__main__':
    DP = joblib.load('C:/Users/HP/Desktop/CS-MCF pkg/DP/processing.model')
    CS_MCF_MODE =joblib.load('C:/Users/HP/Desktop/CS-MCF pkg/CS-MCF/CS_MCF.model')

    data_path = 'C:/Users/HP/Desktop/CS-MCF pkg/'
    temp_data = pd.read_excel(data_path + 'data_eg.xlsx').values
    y = temp_data[:, 0].astype(int)
    X = temp_data[:, 1:]

    y[np.where(y != 0)] = 1
    X_DP = DP.transform(X)
    y_pre, y_prob = CS_MCF_MODE.predict_and_prob(X_DP)



