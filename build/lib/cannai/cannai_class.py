import matplotlib as mpl
import cloudpickle
import pandas as pd
import warnings
import os, sys
import copy
import datetime
from .model_compare.cal_score import *
import xgboost


__all__ = [
    "load_Cclass",
    "Cmodel",
]

def load_Cclass(set_add):
    with open(set_add + '.pkl', "rb") as f:
        r2 = cloudpickle.load(f)
    return r2

class Cannaibase:
    # get df (and dir) to make cannai class
    def __init__(self, dirct=None, save_model=True, binary_class = False):
        self.input = None  # define in instance
        self.model_list = []  # list of [model_name,model_output,date]
        self.model_name_list = []
        self.save_fla = False  # if dir is not selected, cannai_model can't be saved
        self.save_mod = False  # if false, trained model doesn't be saved
        self.warn_flag = False  # throw warning between function
        self.answer = None
        self.input_size = None
        self.output_size = None
        self.Cal_s = Cal_trained_score(self)
        self.iscannai = True # used for checking class
        self.out_is_S = False # if output is series, this become True
        self.class_binary = binary_class # if binary_classification, this become False
        if dirct is not None:
            self.save_fla = True
            self.save_mod = save_model
            self.dir = dirct
            # if folder doesn't exist, make folder
            if os.path.isdir(dirct) == False: os.mkdir(dirct)

    #save cannai_class: get pickle address
    def save_setting(self, set_add):
        with open(set_add + '.pkl', "wb") as f:
            cloudpickle.dump(self, f)


    def yesno(self):
        while True:
            ans = input()
            if ans == "y":
                return True
            elif ans == "n":
                return False

    def warning_check(self):
        if self.warn_flag:
            self.warn_flag = False
            print("warning occurred. do you want to continue?\n(y/n)")
            ans = self.yesno()
            return ans

        else: return True

    def __del__(self, inp=None):
        yesno = None
        if inp is None:
            print("save this model?(y, n)")
            yesno = self.yesno()
        if inp == "y" or yesno == True: self.save_all(None)
        elif inp == "n" or yesno == False: pass
        else: self.save_all(inp)

    def save_all(self, inp):
        if self.save_fla: self.save_input()
        if inp is None:
            print("input Cannnai model address")
            inp = input()
        while True:
            try:
                self.save_setting(inp)
            except FileNotFoundError as e:
                sys.stderr.write(e)
                print("input Cannnai model address")
                inp = input()
            except Exception as e:
                sys.stderr.write(e)
            else: break

    def save_input(self):
        pass  # define in instance

    #return length of model_list
    def __len__(self):
        return len(self.model_list)

    def __iadd__(self, other):
        Cn = self + other
        user_ok = self.warning_check()
        if user_ok:
            self = Cn

    # model can be added using + or +=
    def __add__(self, other):
        Cn = copy.deepcopy(self)
        self.warn_flag = False

        if self.input != other.input:
            self.warn_flag = True
            warnings.warn('input data of two cannai model are different\n(input data was set to first one)')

        if other.save_fla: Cn.save_fla = True

        if Cn.save_fla:
            # second one only has directory
            if self.save_fla == False: Cn.dir = other.dir
            # first one only has directory
            elif other.save_fla == False: pass
            # both has directory
            else:
                if self.dir != other.dir:
                    self.warn_flag = True
                    warnings.warn('directory of two cannai model are different\n(directory was set to first one)')

        if self.save_mod != other.save_mod:
            self.warn_flag = True
            warnings.warn('save_model flag of two cannai model are different\n(save_model flag was set to first one)')

        Cn.model_list = self.model_list + other.model_list

        return Cn

    def save_model(self, model, m_name):
        if self.save_mod:
            with open(self.dir + m_name + '.pkl', "wb") as f:
                cloudpickle.dump(model, f)

    def save_df(self, df, m_name):
        if self.save_fla: df.to_csv(self.dir + m_name + '.csv')

    def check_num(self, input):
        if (type(input) is int) or (type(input) is float) or (type(input) is np.int) or (
                type(input) is np.float) or (type(input) is np.int32) or (type(input) is np.float32) or (type(input) is np.int64) or (type(input) is np.float64): return True
        else:return False

    def prob_01_2l_to_1l(self, list):
        if self.check_num(list[0]) == True: return list
        if len(list[0]) == 1: return list
        l2 = []
        for l1 in list: l2.append(l1[1])
        return l2

    def output_to_df(self, output):
        if self.answer is not None:
            if self.out_is_S:
                if self.class_binary: output = self.prob_01_2l_to_1l(output)
                df = pd.Series(output, name=self.answer.name, index=self.answer.index)
            else:
                df = pd.DataFrame(output, columns=self.answer.columns, index=self.answer.index)

            if df.shape != self.output_size:
                raise RuntimeError("size of true-value df is",self.output_size,", but output size of this model is",df.shape)

        else:
            warnings.warn('true-value data has not been set')
            df = pd.DataFrame(output)
        return df

    def __setitem__(self, key, model, model_n=None):
        sub_ml = self.add_model_base(model,model_n, key)
        self.model_list.insert(key, sub_ml)
        self.model_name_list.insert(key, sub_ml[0])

    def __contains__(self, item):
        if item in self.model_name_list: return True
        else: return False

    def set_input(self, input):
        self.input = input
        self.input_size = input.size

    def set_answer(self, ans_df):
        self.answer = ans_df
        self.save_df(ans_df, "t_value")
        self.output_size = ans_df.shape
        if type(self.answer) is pd.core.series.Series: self.out_is_S = True
        else: self.out_is_S = False

    def set_date(self):
        dt_now = datetime.datetime.now()
        return dt_now.strftime('%Y%m%d_%H%M%S')

    def set_mname(self, model_n, ii):
        cc = 1
        if model_n is None: m_base = "model_" + str(ii)
        else: m_base = model_n

        if m_base in self.model_name_list:
            #  model name already exists
            m_base = m_base + "_"

            while True: #find not existing name
                if m_base + str(cc) in self.model_name_list: cc += 1
                else: break

        return m_base

    def add_model_base(self, model, model_n, ii):
        pass  # define in instance

#get model and model_name
    def add_model(self, model, model_n=None):
        ii = len(self.model_list)  #ii is id of next model
        sub_ml = self.add_model_base(model, model_n, ii)
        self.save_df(sub_ml[1], model_n)
        self.model_list.append(sub_ml)
        self.model_name_list.append(sub_ml[0])

    def get_id(self, key):
        t_it = type(key)

        if self.model_list == []:
            raise Exception("no model inputed")

        if t_it == int:
            return key

        if t_it == str:
            ls = len(self)
            for ii in range(ls + 1):
                if ii == ls:
                    raise IndexError(key + ": can not found in models")
                if self.model_name_list[ii] == key: break
            return ii

        else:
            raise TypeError("key must be int or string, but type of key is " + t_it)

    def __delitem__(self, key):
        id = self.get_id(key)

        try:
            del self.model_list[id]
        except IndexError:
            raise IndexError("number of model is " + str(len(self)) + ", but id is " + str(key))

    def __getitem__(self, item):
        id = self.get_id(item)

        try:
            sub_model = self.model_list[id]
        except IndexError:
            raise IndexError("number of model is " + str(len(self)) + ", but id is " + str(item))
        else:
            return sub_model

    #undo last trained-model addition
    def undo(self):
        del self[-1]

    def get_data(self, key):
        return self[key]

    #throw data list for printing graphs
    def get_datas(self, keys):
        dat_list = []
        for key in keys:
            dat = self[key]
            dat_list.append(dat[1])
        return dat_list

    # throw model name for printing graphs
    def get_names(self, keys):
        dat_list = []
        for key in keys:
            dat = self[key]
            dat_list.append(dat[0])
        return dat_list

    # throw label name for printing graphs
    def get_labelname(self, line):
        if type(line) is str: return line
        elif type(line) is int:
            if self.out_is_S:
                return self.model_list[0][1].name
            else:
                return self.model_list[0][1].columns[line]

    def get_answer(self):
        if self.answer is None:
            raise RuntimeError('true-value data has not been set')
        else:
            return self.answer

    def get_input(self):
        if self.input is None:
            raise RuntimeError('input data has not been set')
        else: return self.input

    def input_convert(self, model):
        if isinstance(model, xgboost.core.Booster):
            return xgboost.DMatrix(self.input)
        else: return self.input

class Cmodel(Cannaibase):
    def save_input(self):
        self.save_df(self.input, "input")

    def add_model_base(self, model, model_n, ii):
        model_n = self.set_mname(model_n, ii)
        dat = self.set_date()
        imp2 = self.input_convert(model)
        al_re = False
        try:
            import catboost
        except:
            pass
        else:
            if isinstance(model, catboost.core.CatBoostClassifier):
                output = model.predict_proba(imp2)
                al_re = True
        if al_re != True:
            output = model.predict(imp2)
        self.save_model(model, model_n)
        df = self.output_to_df(output)
        return [model_n, df, dat]