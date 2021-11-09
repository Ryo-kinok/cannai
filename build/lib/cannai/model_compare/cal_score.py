import sklearn.metrics as sk_met

import numpy as np
import pandas as pd

binary_score_list = ["binary_accuracy","precision","recall","binary_f1","binary_auc","binary_f1_weighted","balanced_accuracy"
                     ,"accuracy"]

def over_half(x):
    if x >= 0.5: return 1
    else: return 0

class Cal_trained_score:
    def __init__(self, p_cannnai):
        self.parent = p_cannnai
        self.ans_df = None
        self.ans_checked = False

    def get_highest_col(self, df):
        return df.idxmax(axis=1)

    def det_answer(self):
        if self.ans_checked is False:
            self.ans_df = self.parent.get_answer()
            self.ans_checked = True
        return self.ans_df

    def get_dfline(self, df, target_line):
        if type(df) == pd.core.series.Series: #if data is series
            return df
        elif type(df) == np.ndarray:
            if df.ndim == 1:
                return df
        return df[target_line]

    def check_binary(self, e_score, ll):
        if e_score in binary_score_list:

            def zero_one(x):
                if x< 0.5: return 0
                else: return 1
            if type(ll) == np.ndarray:
                vfunc = np.vectorize(zero_one)
                ll2 = vfunc(ll)
            elif type(ll) == pd.core.series.Series: ll2 = ll.map(zero_one)
            elif type(ll) == pd.core.frame.DataFrame: ll2 = ll.applymap(zero_one)
            else: ll2 = zero_one(ll)
            return ll2
        else: return ll

    def binary_siries_to_2lines(self, ll):
        if type(ll) is pd.core.series.Series:
            ll_out = []
            for s_ll in ll:
                ll_out.append([1.0 - s_ll, float(s_ll)])
            return ll_out
        elif type(ll) == np.ndarray:
            if ll.ndim == 1:
                ll_out = []
                for s_ll in ll:
                    ll_out.append([1.0 - s_ll, float(s_ll)])
                return ll_out

            else:
                return ll
        else: return ll

    def combine_inout(self,key): #if key is -1, throw true value
        in_df = self.parent.get_input()
        if key == -1:
            out_df = self.parent.get_answer()
        else:
            out_df = self.parent.get_data(key)[1]
        return pd.concat([in_df, out_df], axis=1)

    def get_inout(self, key, target_line):
        out_df = self.parent.get_data(key)[1]

        out_l = self.get_dfline(out_df, target_line)

        ans_df = self.parent.get_answer()
        ans_l = self.get_dfline(ans_df, target_line)

        return out_l, ans_l

    def make_singledf_to_series(self,df):
        if type(df) == pd.core.frame.DataFrame:
            df = df.iloc[:,0]
        return df

    def get_roc(self,key,target_line):
        out_l, ans_l = self.get_inout(key, target_line)
        return self.get_roc_base(out_l, ans_l)

    def get_roc_base(self,out_l, ans_l):
        return sk_met.roc_curve(ans_l,out_l)

    def cal_score_single(self, key, target_line, e_score0):
        e_score = e_score0.lower()

        out_l, ans_l = self.get_inout(key, target_line)


        out_l = self.make_singledf_to_series(out_l)
        ans_l = self.make_singledf_to_series(ans_l)

        out_l = self.check_binary(e_score, out_l)
        ans_l = self.check_binary(e_score, ans_l)

        if e_score == "mae": return sk_met.mean_absolute_error(out_l, ans_l)
        elif e_score == "mse": return sk_met.mean_squared_error(out_l, ans_l)
        elif e_score == "rmse": return np.sqrt(sk_met.mean_squared_error(out_l, ans_l))
        elif e_score == "msle": return sk_met.mean_squared_log_error(out_l, ans_l)
        elif e_score == "rmsle": return np.sqrt(sk_met.mean_squared_log_error(out_l, ans_l))
        elif e_score == "r2": return sk_met.r2_score(out_l, ans_l)
        elif e_score == "accuracy": return sk_met.accuracy_score(out_l, ans_l)
        elif e_score == "precision": return sk_met.precision_score(out_l, ans_l)
        elif e_score == "recall": return sk_met.recall_score(out_l, ans_l)
        elif e_score == "binary_f1": return sk_met.f1_score(out_l, ans_l)
        elif e_score == "binary_f1_weighted": return sk_met.f1_score(out_l, ans_l, average='weighted')
        elif e_score == "balanced_accuracy": return sk_met.balanced_accuracy_score(out_l, ans_l)
        elif (e_score == "binary_auc") or (e_score == "auc"): return sk_met.roc_auc_score(ans_l, out_l )
        elif e_score == "auc_micro": return sk_met.roc_auc_score(out_l, ans_l,average='micro')
        elif e_score == "binary_accuracy": return sk_met.accuracy_score(out_l, ans_l)
        elif e_score == "accuracy": return sk_met.accuracy_score(out_l, ans_l)
        elif (e_score == "binary_logloss") or (e_score == "binary_cross_entropy"):
            out_l2 = self.binary_siries_to_2lines(out_l)
            ans_l2 = self.binary_siries_to_2lines(ans_l)
            return sk_met.log_loss(ans_l2, out_l2)
        elif (e_score == "logloss") or (e_score == "cross_entropy"): return sk_met.log_loss(ans_l, out_l)
        elif e_score == "f1": return sk_met.f1_score(out_l, ans_l)
        elif e_score == "f1_weighted": return sk_met.f1_score(out_l, ans_l, average='weighted')
        elif e_score == "auc_ovr": return sk_met.roc_auc_score(out_l, ans_l, multi_class='ovr')
        elif e_score == "auc_ovo": return sk_met.roc_auc_score(out_l, ans_l, multi_class='ovo')
        elif e_score == "auc_ovr_weighted": return sk_met.roc_auc_score(out_l, ans_l, average='weighted', multi_class='ovr')
        elif e_score == "auc_ovo_weighted": return sk_met.roc_auc_score(out_l, ans_l, average='weighted', multi_class='ovo')
        elif e_score == "average_precision": return sk_met.average_precision_score(out_l, ans_l)
        else: raise IndexError(e_score + ": can not be used")



    def cal_score_multiple(self, key_list, target_line, e_score):
        self.ans_checked = False
        score_list = []
        for key in key_list: score_list.append(self.cal_score_single(key, target_line, e_score))
        return score_list

    def cal_score_multiple_2d(self, key_list, target_line, e_score_list):
        # score[e_score][model]
        self.ans_checked = False
        score_list = []
        for e_score in e_score_list:
            score_list0 = []
            for key in key_list: score_list0.append(self.cal_score_single(key, target_line, e_score))
            score_list.append(score_list0)
        return score_list

    def cal_classtering(self, key, target_line):
        out_l0, ans_l = self.get_inout(key, target_line)
        out_l = out_l0.copy()
        if self.parent.class_type == "m": out_l = self.get_highest_col(out_l0)
        elif self.parent.class_type == "b": out_l = out_l0.map(over_half)
        else: raise IndexError("class_check can be used with classification models")

        max_n = ans_l.max()
        max_n2 = out_l.max()
        if max_n2 > max_n: max_n = max_n2
        max_n += 1
        len_l = len(ans_l)

        taisyou_list = [0 for aa1 in range(max_n*max_n)]
        taisyou_list2 = [0.0 for aa1 in range(max_n*max_n)]

        for ii in range(len_l):
            taisyou_list[out_l[ii] + ans_l[ii] * max_n] += 1
            taisyou_list2[out_l[ii] + ans_l[ii] * max_n] += 1

        al_c = ans_l.value_counts()

        for ii in range(max_n):
            if ii in al_c.index:
                alc_v = float(al_c[ii])
                for jj in range(max_n): taisyou_list2[ii * max_n + jj] /= alc_v
            else:
                for jj in range(max_n): taisyou_list2[ii * max_n + jj] = 0.0

        for ii in range(len(taisyou_list2)): taisyou_list2[ii] = round(taisyou_list2[ii], 2)

        return taisyou_list,taisyou_list2, max_n