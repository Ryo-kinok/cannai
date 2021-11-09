import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from .base import *
import sys
import pandas as pd
import seaborn as sns
import math
import lightgbm
from .. import cannai_class
import warnings

color_list = ["g", "r", "b", "y", "m", "c", "deeppink", "darkseagreen", "tomato", "darkslategray"]

def list_flatten(list_a):
    print(list_a[0],type(list_a[0]))
    if type(list_a[0]) == list:
        f_list = []
        for ll in list_a: f_list.extend(ll)
        return list_flatten(f_list)
    else: return list_a

def multiclass_base(C_mod, key_list, tl):
    tkl = type(key_list)
    if (key_list == None) or (key_list == "all"):
        if C_mod.multicmod: key_list = C_mod.model_name_dict[tl]
        else: key_list = C_mod.model_name_list
    elif (tkl is str) or (tkl is int): key_list = [key_list]
    if len(key_list) > 10:
        raise ImportError("maximum number of keys inputs is 10, but " + str(len(key_list)) + " was input")
    try:
        if C_mod.iscannai != True:
            raise ImportError("C_mod.iscannai is not true")
    except AttributeError:
        raise ImportError("C_mod is not cannai classes")

    if key_list == []:
        raise ImportError("key_list is empty")

    return key_list

def multiclass_base_group(C_mod, group_list):
    group_list = multiclass_base_group_sub(C_mod,group_list)
    key_list = []
    key_class = []
    for ii in range(len(C_mod.model_list)):
        cmgii = C_mod.model_groupid[ii]
        if cmgii in group_list:
            key_list.append(ii)
            key_class.append(cmgii)

    return key_list, key_class,group_list

def multiclass_base_group_sub(C_mod, group_list):
    tkl = type(group_list)
    if (group_list == None) or (group_list == "all"):
        group_list = [ij for ij in range(len(C_mod.group_name_list))]
    elif (tkl is str) or (tkl is int): group_list = [group_list]
    if len(group_list) > 10:
        raise ImportError("maximum number of keys inputs is 10, but " + str(len(group_list)) + " was input")
    try:
        if C_mod.iscannai != True:
            raise ImportError("C_mod.iscannai is not true")
    except AttributeError:
        raise ImportError("C_mod is not cannai classes")

    if group_list == []:
        raise ImportError("key_list is empty")

    group_list2 = []
    for gg in group_list:
        if type(gg) == int:
            group_list2.append(gg)
        else:
            group_list2.append(C_mod.group_name_list.index(gg))

    return group_list2

def get_labels(df):
    if type(df) is pd.core.series.Series:
        return df.name
    elif type(df) is pd.core.frame.DataFrame:
        return df.columns
    else:
        return ["label" + str(pp) for pp in range(df.shape[-1])]


def add_labels(df, addi):
    if type(df) is pd.core.series.Series:
        df.name = df.name + addi
    else:
        dc = df.columns
        for ii in range(len(dc)):
            dc[ii] = dc[ii] + addi
        df.columns = dc
    return df


def get_line(df, l_name):
    if type(df) is pd.core.series.Series:
        if df.name != l_name and l_name != 0:
            raise IndexError(str(l_name) + "is not included in this data")
        else:
            return df
    else:
        return df[l_name]

class multiclass_lib:
    def __init__(self, parent):
        self.parent = parent

    def bar(self, key_list, target_line, score_list,target_label=None):
        multiclass_bar(self.parent, key_list, target_line, score_list,target_label)

    def scatter(self, key_list, target_line, explanatory_line_list,target_label=None):
        multiclass_scatter(self.parent, key_list, target_line, explanatory_line_list,target_label)

    def matrix(self, key_list,target_label=None):
        multiclass_matrix(self.parent, key_list,target_label)

    def rank(self, key_list, target_line ,score_type = "abs", comvert="default", show_range="top50",target_label=None):
        multiclass_rank(self.parent, key_list, target_line, score_type, comvert, show_range,target_label)

    def radarchart(self, key_list, target_line,target_label=None):
        multiclass_radarchart(self.parent, key_list, target_line,target_label)

    def f_importance(self, key_list,bar_type=None,target_label=None, show_range="top20"):
        multiclass_f_importance(self.parent, key_list,bar_type,target_label)

    def roccurve(self, key_list,target_label=None):
        multiclass_roccurve(self.parent,key_list,target_label)

    def class_check(self, key_list, target_line, value_type="number",target_label=None):
        multiclass_class_check(self.parent,key_list, target_line,value_type,target_label)

    def boxplot(self, group_list, target_line, score_list):
        multiclass_boxplot(self.parent, group_list, target_line, score_list)

def check_multi_cmodel(C_mod, target_line, target_label):
    if C_mod.multicmod:
        if target_label == None:
            raise IndexError(
                "please input target_label")
        elif target_line != None:
            raise IndexError(
                "please input target_line as None")
        elif type(target_label) != list:
            target_label = [target_label]

        C_mod.out_is_S = True
    elif target_line != None:
        target_label = [None]
        C_mod.out_is_S = True
    else:
        target_label = [None]
        C_mod.out_is_S = False

    return target_label

def multiclass_bar(C_mod, key_list, target_line, score_list,target_label=None):
    """print bar graph for comparing models

    Parameters:
    ----------
    C_mod : cannai_model

    key_list : list of (int or str)
        key list of loading each model
    
    target_line : int or str or None
        label of column which wanted to use for plotting y
        if you are using multi_Cmodel, please set this None

    score_list : str or (list of str)
        list of evaluate score, what you want to display
        regression: MAE,MSE,RMSE,MSLE,RMSLE,R2
        binary classification(label): binary_accuracy,precision,recall,binary_f1,binary_f1_weighted,balanced_accuracy
        binary classification(rate_list): binary_cross_entropy(binary_logloss),binary_auc,auc_micro,average_precision
        multi classification(label): accuracy,cross_entropy(logloss),
        multi classification(rate_list): f1,f1_weighted,auc,auc_micro,auc_ovr,auc_ovo,auc_ovr_weighted,auc_ovo_weighted

    target_label : None or str or list
        if you are not using multi_Cmodel, do not set this input.
        you you are using it, each model group called by target_label.

    """

    target_label = check_multi_cmodel(C_mod, target_line, target_label)

    tsl = type(score_list)
    if (tsl is str) or (tsl is int): score_list = [score_list]



    len_score_l = len(score_list)

    for tl1 in target_label:
        if (C_mod.multicmod) and (tl1 not in C_mod.model_name_dict):
            warnings.warn('model are not inputed in ' + tl1 + ' group')
            continue
        key_list = multiclass_base(C_mod, key_list, tl1)
        print(key_list)
        len_key_l = len(key_list)
        fig = plt.figure(figsize=(8.0, 6.0))
        labels = C_mod.get_names(key_list, target_label=tl1)
        ax_list = []
        if tl1 != None: target_line = [tl1]
        for count in range(len_score_l):
            ax = fig.add_subplot(1, len_score_l, count + 1)
            e_score = score_list[count]
            score_out = C_mod.Cal_s.cal_score_multiple(key_list, target_line, e_score)
            left = np.array([ii + 1 for ii in range(len_key_l)])
            height = score_out
            ax.bar(left, height, tick_label=labels, color=color_list[:len_key_l], align="center")
            ax.set_title(e_score)
            ax.set_xlabel("models")
            ax.set_ylabel("score")
            ax.grid(True)
            ax_list.append(ax)
        plt.tight_layout()
        plt.show()


def multiclass_scatter(C_mod, key_list, target_line, explanatory_line_list,target_label=None):
    """ print scatter graph for comparing models

    Parameters:
    ----------
    C_mod : cannai_model

    key_list : list of (int or str)
        key list of loading each model

    target_line : int or str or None
        label of column which wanted to use for plotting y
        if you are using multi_Cmodel, please set this None

    explanatory_line_list : str or (list of str)
        labels of column which wanted to use for plotting x

    target_label : None or str or list
        if you are not using multi_Cmodel, do not set this input.
        you you are using it, each model group called by target_label.


    """
    target_label = check_multi_cmodel(C_mod, target_line, target_label)

    fig = plt.figure(figsize=(8.0, 12.0))

    ltl = len(target_label)

    for tl1 in target_label:
        if tl1 != None: target_line = tl1
        if explanatory_line_list is str: explanatory_line_list = [explanatory_line_list]
        key_list = multiclass_base(C_mod, key_list,tl1)
        labels = C_mod.get_names(key_list,target_label=tl1)

        len_key_l = len(key_list)
        len_exp_l = len(explanatory_line_list)

        out_l_list = C_mod.get_datas(key_list,target_label=tl1)

        inp_df = C_mod.get_input()
        ans_df = C_mod.get_answer(tl1)

        target_name = C_mod.get_labelname(target_line,target_label=tl1)

        ans_li = get_line(ans_df, target_name)

        ax_list = []

        for count in range(len_exp_l):

            exp_name = explanatory_line_list[count]
            inp_li = get_line(inp_df, exp_name)


            for count2 in range(len_key_l):
                ax = fig.add_subplot(len_exp_l*len_key_l, 1, count * len_key_l + count2 + 1)
                out_l0 = out_l_list[count2]
                out_l = get_line(out_l0, target_line)
                sa_l = out_l - ans_li
                ax.bar(inp_li, out_l, label=labels[count], color=color_list[count], align="center")
                ax.set_xlabel(exp_name)
                ax.set_ylabel(target_name)
                ax.grid(True)
                ax_list.append(ax)
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.show()


def multiclass_matrix(C_mod, key_list,target_label=None):

    """print matrix graph for comparing models

    Parameters:
    ----------
    C_mod : cannai_model

    key_list : list of (int or str)
        key list of loading each model
    """
    target_label = check_multi_cmodel(C_mod, None, target_label)

    for tl1 in target_label:
        key_list = multiclass_base(C_mod, key_list, tl1)
        labels = C_mod.get_names(key_list,target_label=tl1)

        len_key_l = len(key_list)

        fig = plt.figure(figsize=(8.0, 6.0))
        ax_list = []

        x_line = get_labels(C_mod.get_input())
        y_line = get_labels(C_mod.get_answer(tl1))

        ax = fig.add_subplot(1, 1, 1)

        base_df = C_mod.get_input().var()
        print(base_df)
        if C_mod.is_df_or_series(base_df) == False:
            base_df = pd.DataFrame(base_df)
        base_df.name = "distribute"

        labels = C_mod.get_names(key_list,target_label=target_label)

        for count in range(len_key_l):
            comb_df0 = C_mod.Cal_s.combine_inout(count)
            comb_df1 = comb_df0.corr()
            comb_df2 = comb_df1.loc[y_line, x_line]
            comb_df2 = add_labels(comb_df2, "_(" + labels[count] + ")")
            base_df = pd.concat([base_df, comb_df2], axis=1)

        print(base_df)
        sns.heatmap(base_df.drop(columns="distribute"), annot=True, ax=ax)

        plt.tight_layout()
        plt.show()


def multiclass_rank(C_mod, key_list, target_line ,score_type = "abs", comvert="default", show_range="top50",target_label=None):
    """print ranking graph for comparing models
    Args:

        C_mod : cannai_model

        key_list : list of (int or str)
            key list of loading each model

        score_type : str(default: "abs")
            "abs": | pred_value - true_value |
            "rel": | 1 - (pred_value / true_value) |

        comvert: str(default: "default")
            y value conversion
            "default": no change
            "log": convert to log10 value

        show_range: str
            show top / bottom X datas
            (X must be int value)
            "topX": show X datas from top
            "botX": show X datas from bottom

    """
    target_label = check_multi_cmodel(C_mod, target_line, target_label)





    for tl1 in target_label:
        if (C_mod.multicmod) and (tl1 not in C_mod.model_name_dict):
            warnings.warn('model are not inputed in ' + tl1 + ' group')
            continue
        key_list = multiclass_base(C_mod, key_list, tl1)

        labels = C_mod.get_names(key_list, target_label=tl1)
        len_key_l = len(key_list)
        if tl1 != None: target_line = tl1
        target_name = C_mod.get_labelname(target_line, target_label=tl1)
        print(target_name)

        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(1, 1, 1)

        sa_lists = []
        for count in range(len_key_l):
            key = key_list[count]

            out_l, ans_l = C_mod.Cal_s.get_inout(key, target_line)

            if C_mod.is_df_or_series(out_l):
                out2 = list_flatten(out_l.values.tolist())
            else:
                out2 = list_flatten(out_l.tolist())

            if C_mod.is_df_or_series(ans_l):
                ans2 = list_flatten(ans_l.values.tolist())
            else:
                ans2 = list_flatten(ans_l.tolist())

            max_ans_10000 = max(ans2) / 10000

            sa2 = []

            if score_type == "abs":
                for jjj in range(len(out2)) : sa2.append(abs(out2[jjj] - ans2[jjj]))
            elif score_type == "rel":
                for jjj in range(len(out2)) : sa2.append(abs(math.log(out2[jjj] / (ans2[jjj] + max_ans_10000))))

            else: raise IndexError("score_type must be diff or prod")

            for ss in sa2:
                if score_type == "abs": ss_b = abs(ss)
                elif score_type == "rel": ss_b = abs(ss)
                sa_lists.append([ss_b, count])

        sa_lists = sorted(sa_lists)
        l_sal = len(sa_lists)

        try:
            vv = int(show_range[3:])
        except ValueError:
            raise IndexError("show_range does not match topX or botX")

        if show_range[:3] == "top":
            if l_sal > vv: sa_lists = sa_lists[l_sal - vv:]
        elif show_range[:3] == "bot":
            if l_sal > vv: sa_lists = sa_lists[:vv]
        else: raise IndexError("show_range must be started from top or bot")

        l_sal = len(sa_lists)

        for count in range(len_key_l):

            v_list = []
            c_list = []
            for c2 in range(l_sal):
                c_list.append(c2)
                if sa_lists[c2][1] == count:
                    v_list.append(sa_lists[c2][0])
                else: v_list.append(0)

            if show_range[:3] == "top": c_list.reverse()

            if comvert == "log": ax.set_yscale('log')
            #ax.scatter(c_list, v_list, label=labels[count], color=color_list[count], s=8, alpha=0.4)
            ax.bar(c_list, v_list, label=labels[count], color=color_list[count], align="center")


        ax.set_ylabel(target_name + "_error")
        ax.grid(True)
        plt.tight_layout()
        if show_range[:3] == "top": plt.legend(loc='upper right')
        else: plt.legend(loc='upper left')
        plt.show()

def multiclass_roccurve(C_mod, key_list, target_line,target_label=None):
    """print roc curve in one graph
        Args:

            C_mod : cannai_model

            target_line : int or str or None
                label of column which wanted to use for plotting y
                if you are using multi_Cmodel, please set this None

            key_list : list of (int or str)
                key list of loading each model
        """
    target_label = check_multi_cmodel(C_mod, target_line, target_label)

    for tl1 in target_label:
        if (C_mod.multicmod) and (tl1 not in C_mod.model_name_dict):
            warnings.warn('model are not inputed in ' + tl1 + ' group')
            continue
        if tl1 != None: target_line = [tl1]
        key_list = multiclass_base(C_mod, key_list, tl1)

        len_key_l = len(key_list)
        if type(target_line) is int: target_line = [target_line]
        len_tar_l = len(target_line)
        fig = plt.figure(figsize=(8.0, 6.0))
        labels = C_mod.get_names(key_list, target_label=tl1)
        ax_list = []

        for count in range(len_tar_l):
            ax = fig.add_subplot(1, len_tar_l, count+1)
            for count2 in range(len_key_l):
                roc = C_mod.Cal_s.get_roc(key_list[count2], target_line[count])
                ax.plot(roc[0], roc[1], label=labels[count2], color=color_list[count2])
            ax.set_title(tl1)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.grid(True)
            ax.legend(loc='lower right')
            ax_list.append(ax)
        plt.tight_layout()
        plt.show()


def multiclass_radarchart(C_mod, key_list, target_line,target_label=None):
    """print radar chart for comparing models easily
    Args:

        C_mod : cannai_model

        key_list : list of (int or str)
            key list of loading each model

        target_line : int or str
            label of column which wanted to calculate

    """

    def log_change(in_num):
        if in_num >= 0: return math.log10(in_num)
        else: return -1.0 * math.log10(-1.0 * in_num)

    target_label = check_multi_cmodel(C_mod, target_line, target_label)

    for tl1 in target_label:
        key_list = multiclass_base(C_mod, key_list,tl1)
        labels = C_mod.get_names(key_list,target_label=target_label)

        len_key_l = len(key_list)
        c_type = C_mod.class_type

        fig = plt.figure(figsize=(20.0, 15.0))
        ax_list = []

        if c_type == "b":
            e_score_list = ["accuracy", "precision", "recall", "binary_logloss", "binary_auc"]
            e_title = ["accuracy", "precision", "recall", "logloss_inverse", "auc_inverse"]
            score_out = C_mod.Cal_s.cal_score_multiple_2d(key_list, target_line, e_score_list)

            for cc in range(len(score_out[0])):
                score_out[3][cc] = log_change(score_out[3][cc])
                score_out[4][cc] = log_change(score_out[4][cc])

            max_list = []
            min_list = []
            for so in score_out:
                max_list.append(max(so))
                min_list.append(min(so))

            for ii in range(5):
                if min_list[ii] > 0: min_list[ii] = 0

            for cc in range(len(score_out[0])):
                score_out[0][cc] /= max_list[0]
                score_out[1][cc] /= max_list[1]
                score_out[2][cc] /= max_list[2]
                score_out[3][cc] = (max_list[3] - score_out[3][cc]) / (max_list[3] - min_list[3])
                score_out[4][cc] = (max_list[4] - score_out[4][cc]) / (max_list[4] - min_list[4])

        elif c_type == "a":
            e_score_list = ["accuracy", "f1", "logloss", "auc"]
            e_title = ["accuracy", "f1", "logloss_inverse", "auc_inverse"]

            score_out = C_mod.Cal_s.cal_score_multiple_2d(key_list, target_line, e_score_list)

            for cc in range(len(score_out[0])):
                score_out[2][cc] = log_change(score_out[2][cc])
                score_out[3][cc] = log_change(score_out[3][cc])

            max_list = []
            min_list = []
            for so in score_out:
                max_list.append(max(so))
                min_list.append(min(so))

            for ii in range(4):
                if min_list[ii] > 0: min_list[ii] = 0

            for cc in range(len(score_out[0])):
                score_out[0][cc] /= max_list[0]
                score_out[1][cc] /= max_list[1]
                score_out[2][cc] = (max_list[2] - score_out[2][cc]) / (max_list[2] - min_list[2])
                score_out[3][cc] = (max_list[3] - score_out[3][cc]) / (max_list[3] - min_list[3])

        elif c_type == "r":
            e_score_list = ["rmse","r2","mae","rmsle"]
            e_title = ["rmse_inverse","r2_inverse","mae_inverse","rmsle_inverse"]

            score_out = C_mod.Cal_s.cal_score_multiple_2d(key_list, target_line, e_score_list)

            for cc in range(len(score_out[0])):
                score_out[0][cc] = log_change(score_out[0][cc])
                score_out[1][cc] = log_change(score_out[1][cc])
                score_out[2][cc] = log_change(score_out[2][cc])
                score_out[3][cc] = log_change(score_out[3][cc])

            max_list = []
            min_list = []
            for so in score_out:
                max_list.append(max(so))
                min_list.append(min(so))

            for ii in range(4):
                if min_list[ii] > 0: min_list[ii] = 0

            for cc in range(len(score_out[0])):
                score_out[0][cc] = (max_list[0] - score_out[0][cc]) / (max_list[0] - min_list[0])
                score_out[1][cc] = (max_list[1] - score_out[1][cc]) / (max_list[1] - min_list[1])
                score_out[2][cc] = (max_list[2] - score_out[2][cc]) / (max_list[2] - min_list[2])
                score_out[3][cc] = (max_list[3] - score_out[3][cc]) / (max_list[3] - min_list[3])

        val_list = list(zip(*score_out))

        ax = fig.add_subplot(len_key_l+1, 1, 1, polar=True)
        for count in range(len_key_l):
            vals = val_list[count]
            angles = np.linspace(0, 2 * np.pi, len(vals) + 1, endpoint=True)
            values = np.concatenate((vals, [vals[0]]))  # 閉じた多角形にする
            ax.plot(angles, values, 'o-', color=color_list[count])  # 外枠
            ax.set_thetagrids(angles[:-1] * 180 / np.pi, e_title)  # 軸ラベル
            ax.set_rlim(0, 1)
        ax.set_title(labels[count])
        ax_list.append(ax)

        for count in range(len_key_l):
            vals = val_list[count]
            ax = fig.add_subplot(len_key_l+1, 1, count + 2, polar=True)
            angles = np.linspace(0, 2 * np.pi, len(e_title) + 1, endpoint=True)
            values = np.concatenate((vals, [vals[0]]))  # 閉じた多角形にする
            print(values)
            ax.plot(angles, values, 'o-', color=color_list[count])  # 外枠
            ax.fill(angles, values, alpha=0.25, color=color_list[count])  # 塗りつぶし
            ax.set_thetagrids(angles[:-1] * 180 / np.pi, e_title)  # 軸ラベル
            ax.set_rlim(0, 1)
            ax.set_title(labels[count])
            ax_list.append(ax)
        plt.tight_layout()
        plt.show()

def multiclass_f_importance(C_mod, key_list, bar_type = None, show_range="top20",target_label=None):
    """print bar chart about feature importance of models
    Args:

        C_mod : cannai_model

        key_list : list of (int or str)
            key list of loading each model

        bar_type: None or str
            None: show feature importance of each model
            "overlaid": make overlaid bar graph
            "lineup": sort label by score and show in one graph


    """

    def get_feature_importance(l_mod):
        #print(type(l_mod))
        if type(l_mod) == lightgbm.basic.Booster: return l_mod.feature_importance()
        else: return l_mod.feature_importances_

    target_label = check_multi_cmodel(C_mod, None, target_label)

    for tl1 in target_label:

        le_ke = len(key_list)
        labels = C_mod.get_names(key_list,target_label=tl1)

        if bar_type == None:
            fig = plt.figure(figsize=(8.0, 6.0))
            ax_list = []
            for iii in range(le_ke):
                ax = fig.add_subplot(le_ke, 1, iii+1)
                l_mod = C_mod.get_model(key_list[iii],target_label=target_label)
                lbiii = labels[iii]
                fe_i = pd.DataFrame(get_feature_importance(l_mod), index=C_mod.get_input().columns, columns=[lbiii])
                fe_i = fe_i.sort_values(lbiii, ascending=True)
                labels_b = fe_i.index
                le_la = len(labels_b)
                height_kari = fe_i[lbiii]

                try:
                    vv = int(show_range[3:])
                except ValueError:
                    raise IndexError("show_range does not match topX or botX")

                if show_range[:3] == "top":
                    if le_la > vv:
                        labels_b = labels_b[le_la - vv:]
                        height_kari = height_kari[le_la - vv:]
                elif show_range[:3] == "bot":
                    if le_la > vv:
                        labels_b = labels_b[:vv]
                        height_kari = height_kari[:vv]

                le_la = len(labels_b)
                left = np.array([ii + 1 for ii in range(le_la)])
                height = height_kari
                ax.barh(left, height, tick_label=labels_b, color=color_list[iii], align="center")
                ax.set_title(lbiii)
                ax.set_ylabel("labels")
                ax.grid(True)
                ax_list.append(ax)
            ax_list[-1].set_xlabel("importance")
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.7)
            plt.show()
        elif (bar_type == "overlaid") or (bar_type == "lineup"):
            fig = plt.figure(figsize=(8.0, 12.0))
            ax = fig.add_subplot(1, 1, 1)
            l_mod = C_mod.get_model(key_list[0])
            fei0 = get_feature_importance(l_mod)
            fe_i = pd.DataFrame(fei0/fei0.max(), index=C_mod.get_input().columns, columns=[labels[0]])
            for iii in range(1,le_ke):
                l_mod = C_mod.get_model(key_list[iii])
                lbiii = labels[iii]
                fei0 = get_feature_importance(l_mod)
                fe_i[lbiii] = fei0/fei0.max()

            fe_i["all_model_sum_akods"] = fe_i.sum(axis=1)
            fe_i = fe_i.sort_values("all_model_sum_akods", ascending=True)
            labels_b = fe_i.index
            le_la = len(labels_b)

            if bar_type == "overlaid":
                left = np.array([ii + 1 for ii in range(le_la)])
                print(left)
                sum_heig = [0 for jjj in range(le_la)]

                for iii in range(le_ke):
                    height = fe_i.iloc[:,iii]
                    ax.bar(left, height,bottom=sum_heig , color=color_list[iii], align="center", label=labels[iii])
                    hvt = height.values.tolist()
                    for iii2 in range(le_la):
                        sum_heig[iii2] += hvt[iii2]
                ax.set_ylabel("importance")
                ax.set_xlabel("labels")
                ax.set_xticks(left)
                ax.set_xticklabels(labels_b)
                ax.legend(loc="upper left")
                ax.grid(True)

            elif bar_type == "lineup":
                b_width = 0.8 / le_ke
                left = np.array([ii + 1 for ii in range(le_la)])
                for iii in range(le_ke):
                    height = fe_i.iloc[:, iii]
                    ax.bar(left + b_width*iii, height, width = b_width, color=color_list[iii], align="center", label=labels[iii])
                ax.set_ylabel("importance")
                ax.set_xlabel("labels")
                ax.set_xticks(left + b_width * 0.5 * (le_ke - 1))
                ax.set_xticklabels(labels_b)
                ax.legend(loc="upper left")
                ax.grid(True)

            plt.subplots_adjust(hspace=0.4)
            plt.tight_layout()
            plt.show()

def multiclass_class_check(C_mod,key_list, target_line, value_type="number",target_label=None):
    """print class rate 
           Args:

               C_mod : cannai_model

               target_line : int or str or None
                   label of column which wanted to use for plotting y
                   if you are using multi_Cmodel, please set this None
            
               value_type: str(default: "number")
                   set values in table.
                   "number": number of each
                   "rate": each rate of classification.
                

               key_list : list of (int or str)
                   key list of loading each model
           """
    target_label = check_multi_cmodel(C_mod, target_line, target_label)
    if value_type == "number":
        nv = 0
    elif value_type == "rate":
        nv = 1
    else:
        raise IndexError("value_type must be number or rate")

    for tl1 in target_label:
        if (C_mod.multicmod) and (tl1 not in C_mod.model_name_dict):
            warnings.warn('model are not inputed in ' + tl1 + ' group')
            continue
        if tl1 != None: target_line = [tl1]
        key_list = multiclass_base(C_mod, key_list, tl1)

        len_key_l = len(key_list)
        if type(target_line) is int: target_line = [target_line]
        len_tar_l = len(target_line)
        fig = plt.figure(figsize=(8.0, 6.0))
        labels = C_mod.get_names(key_list, target_label=tl1)
        ax_list = []

        for count in range(len_tar_l):
            ax = fig.add_subplot(1, len_tar_l, count + 1)
            class_datas = []
            class_datas2 = []
            for count2 in range(len_key_l):
                class_data,class_data2, max_n = C_mod.Cal_s.cal_classtering(key_list[count2], target_line)
                class_datas.append(class_data)
                class_datas2.append(class_data2)

            cddf_ind = []
            for ii in range(max_n):
                for jj in range(max_n):
                    cddf_ind.append("TV_class " + str(ii) + "  pred_class " + str(jj))

            if nv == 0:
                cd_df = pd.DataFrame(np.array(class_datas).T,
                                     index=cddf_ind,
                                     columns=labels)
            elif nv == 1:
                cd_df = pd.DataFrame(np.array(class_datas2).T,
                                     index=cddf_ind,
                                     columns=labels)
            ax.set_title(tl1)

            c_col = np.full_like(cd_df.values, "", dtype=object)
            for ii in range(max_n):
                akr = 1 - (class_data2[ii * max_n + jj]/2)
                for jj in range(max_n):
                    if ii == jj:
                        for gg in range(len_key_l): c_col[ii * max_n + jj, gg] = [1,akr,akr]
                    else:
                        for gg in range(len_key_l): c_col[ii * max_n + jj, gg] = [akr,akr,1]


            ax.axis('off')
            the_table = ax.table(cellText=cd_df.values, colLabels=cd_df.columns, rowLabels=cd_df.index, loc="center",cellColours=c_col)
            for pos, cell in the_table.get_celld().items():
                cell.set_height(1 / len(cd_df.values))
            ax_list.append(ax)
        plt.tight_layout()
        plt.show()


def multiclass_boxplot(C_mod, group_list, target_line, score_list):
    """print bar graph for comparing models

    Parameters:
    ----------
    C_mod : cannai_model

    key_list : list of (int or str)
        group list of loading each model

    target_line : int or str or None
        label of column which wanted to use for plotting y
        if you are using multi_Cmodel, please set this None

    score_list : str or (list of str)
        list of evaluate score, what you want to display
        regression: MAE,MSE,RMSE,MSLE,RMSLE,R2
        binary classification(label): binary_accuracy,precision,recall,binary_f1,binary_f1_weighted,balanced_accuracy
        binary classification(rate_list): binary_cross_entropy(binary_logloss),binary_auc,auc_micro,average_precision
        multi classification(label): accuracy,cross_entropy(logloss),
        multi classification(rate_list): f1,f1_weighted,auc,auc_micro,auc_ovr,auc_ovo,auc_ovr_weighted,auc_ovo_weighted

    """

    target_label = check_multi_cmodel(C_mod, target_line, None)

    tsl = type(score_list)
    if (tsl is str) or (tsl is int): score_list = [score_list]

    len_score_l = len(score_list)

    for tl1 in target_label:
        if (C_mod.multicmod) and (tl1 not in C_mod.model_name_dict):
            warnings.warn('model are not inputed in ' + tl1 + ' group')
            continue
        key_list, key_class,group_list2 = multiclass_base_group(C_mod, group_list)
        print(group_list2)
        len_key_l = len(key_list)
        fig = plt.figure(figsize=(8.0, 6.0))
        labels = [C_mod.group_name_list[gll] for gll in group_list2]
        ax_list = []
        if tl1 != None: target_line = [tl1]
        for count in range(len_score_l):
            ax = fig.add_subplot(1, len_score_l, count + 1)
            e_score = score_list[count]
            score_out = C_mod.Cal_s.cal_score_multiple(key_list, target_line, e_score)

            score_group = [[] for ii3 in range(max(key_class)+1)]
            for jj2 in range(len(score_out)):
                score_group[key_class[jj2]].append(score_out[jj2])
            score_group = [xx for xx in score_group if xx]

            ax.set_xticklabels(labels)
            ax.boxplot(score_group)
            ax.set_title(e_score)
            ax.set_xlabel("models")
            ax.set_ylabel("score")
            ax.grid(True)
            ax_list.append(ax)
        plt.tight_layout()
        plt.show()