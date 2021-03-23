import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from .base import *
import sys
import pandas as pd
import seaborn as sns

color_list = ["g", "r", "b", "y", "m", "c", "deeppink", "darkseagreen", "orengered", "darkslategray"]


def multiclass_base(C_mod, key_list):
    tkl = type(key_list)
    if (tkl is str) or (tkl is int): key_list = [key_list]
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


def get_labels(df):
    if type(df) is pd.core.series.Series:
        return df.name
    else:
        return df.columns


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


def multiclass_bar(C_mod, key_list, target_line, score_list):
    """print bar graph for comparing models

    Parameters:
    ----------
    C_mod : cannai_model

    key_list : list of (int or str)
        key list of loading each model
    
    target_line : int or str
        label of column which wanted to calculate

    score_list : str or (list of str)
        list of evaluate score, what you want to display
        regression: MAE,MSE,RMSE,MSLE,RMSLE,R2
        binary classification(label): binary_accuracy,precision,recall,binary_f1,binary_f1_weighted,balanced_accuracy
        binary classification(rate_list): binary_cross_entropy(binary_logloss),binary_auc,average_precision
        multi classification(label): accuracy,cross_entropy(logloss),
        multi classification(rate_list): f1,f1_weighted,auc_ovr,auc_ovo,auc_ovr_weighted,auc_ovo_weighted


    """
    tsl = type(score_list)
    if (tsl is str) or (tsl is int): score_list = [score_list]

    key_list = multiclass_base(C_mod, key_list)
    labels = C_mod.get_names(key_list)

    len_score_l = len(score_list)
    len_key_l = len(key_list)

    fig = plt.figure(figsize=(8.0, 6.0))
    ax_list = []

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


def multiclass_scatter(C_mod, key_list, target_line, explanatory_line_list):
    """ print scatter graph for comparing models

    Parameters:
    ----------
    C_mod : cannai_model

    key_list : list of (int or str)
        key list of loading each model

    target_line : int or str
        label of column which wanted to use for plotting y

    explanatory_line_list : str or (list of str)
        labels of column which wanted to use for plotting x

    """
    if explanatory_line_list is str: explanatory_line_list = [explanatory_line_list]
    key_list = multiclass_base(C_mod, key_list)
    labels = C_mod.get_names(key_list)

    len_key_l = len(key_list)
    len_exp_l = len(explanatory_line_list)

    fig = plt.figure(figsize=(8.0, 6.0))

    out_l_list = C_mod.get_datas(key_list)

    inp_df = C_mod.get_input()
    ans_df = C_mod.get_answer()

    target_name = C_mod.get_labelname(target_line)

    ans_li = get_line(ans_df, target_name)

    ax_list = []

    for count in range(len_exp_l):
        ax = fig.add_subplot(len_exp_l, 1, count + 1)
        exp_name = explanatory_line_list[count]
        inp_li = get_line(inp_df, exp_name)

        ax.scatter(inp_li, ans_li, label="true value", color='k', s=15, alpha=0.4)
        for count2 in range(len_key_l):
            out_l0 = out_l_list[count2]
            out_l = get_line(out_l0, target_line)
            ax.scatter(inp_li, out_l, label=labels[count2], color=color_list[count2], s=8, alpha=0.4)
        ax.set_xlabel(exp_name)
        ax.set_ylabel(target_name)
        ax.grid(True)
        ax_list.append(ax)
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.show()


def multiclass_matrix(C_mod, key_list):

    """print matrix graph for comparing models

    Parameters:
    ----------
    C_mod : cannai_model

    key_list : list of (int or str)
        key list of loading each model
    """

    key_list = multiclass_base(C_mod, key_list)
    labels = C_mod.get_names(key_list)

    len_key_l = len(key_list)

    fig = plt.figure(figsize=(8.0, 6.0))
    ax_list = []

    x_line = get_labels(C_mod.get_input())
    y_line = get_labels(C_mod.get_answer())

    ax = fig.add_subplot(1, 1, 1)

    base_df = C_mod.get_input().var()
    base_df.name = "distribute"

    labels = C_mod.get_names(key_list)

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


def multiclass_rank(C_mod, key_list, target_line ,score_type = "abs", comvert="default", show_range="top50"):
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

    key_list = multiclass_base(C_mod, key_list)
    labels = C_mod.get_names(key_list)

    len_key_l = len(key_list)

    fig = plt.figure(figsize=(8.0, 6.0))
    ax_list = []

    ax = fig.add_subplot(1, 1, 1)


    target_name = C_mod.get_labelname(target_line)
    sa_lists = []

    for count in range(len_key_l):
        key = key_list[count]
        out_l, ans_l = C_mod.Cal_s.get_inout(key, target_line)

        if score_type == "abs":
            sa_l = out_l - ans_l
        elif score_type == "rel":
            sa_l = (out_l - ans_l) / ans_l
        else: raise IndexError("score_type must be diff or prod")

        sa2 = sa_l.values.tolist()
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
