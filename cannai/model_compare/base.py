import matplotlib

def get_dflist(C_mod, key_list):
    df_list = []
    datas = C_mod.get_data(key_list)
    for da in datas: df_list.append(da[1])
    return df_list