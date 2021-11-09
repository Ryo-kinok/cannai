# **Cannai**

## **Overview**
機械学習用の可視化ライブラリです。
このライブラリは、google colabでの実行を想定しています。

## **Description**
学習済み機械学習モデルを格納し、管理するライブラリです。
cannaiモデル作成、学習済みモデル格納、グラフ表示のいずれも、一行のソースコードで実行可能です。

## **How to use**

ライブラリのインストールは

```python
!pip install cannai
```

で行えます。

colab等の一部環境の場合、関連するライブラリが自動でインストールされます。

**管理用ライブラリの定義**

```python
import cannai
Cnai = cannai.Cmodel("b",base_dir)
```

第一引数では文字を入力することで,対応するグラフを使用できる.
"b": 二値分類問題
"c": 分類問題
"r": 回帰問題

インポートしたライブラリから、管理用ライブラリを作成します。

ライブラリ作成の行う際, ライブラリの内容を保存するディレクトリ、
扱うモデルの出力が二値分類かどうかをbinary_classで設定します。

```python
Cnai.set_input(test_df)
Cnai.set_answer(test_df_ans)
```

テストデータについて, 説明変数はset_input関数、
目的変数はset_answer関数にそれぞれdataframe(もしくはseries)を入力してください。

**学習済みモデルの格納**

```python
xgbst = xgb.train(xgb_params,dtrain,)
Cnai.add_model(xgbst,"xgb")
```

学習済みモデルは、add_model関数を用いることで、ライブラリに追加することができます。
上のコードは, xgboostの学習済みモデルを格納している様子で、
add_model関数にモデル名("xgb")と共に格納しています。
(モデル名がない場合、自動で割り当てられます)

**グラフ表示**

```python
from cannai.model_compare.multiclass import multiclass_bar
multiclass_bar(Cnai,[1,2,3] ,0,["binary_accuracy","binary_cross_entropy"])
```

グラフを表示したい際、対応する関数をcannai.model_compareから呼び出し、
呼び出した関数にライブラリを入力してください

上のソースコードは、 multiclass内にあるmulticlass_bar関数を呼び出し実行しています。

## **Functions**

### **multiclass:**

用語:　
　モデルid: モデルは格納されたモノから順番に, 0,1,2,...とidが振られています。
　モデル名:「学習済みモデルの格納」で指定したモデル名です。

**multiclass_bar:**

<img width="414" alt="multiclass_bar" src="https://user-images.githubusercontent.com/81252378/114963734-57d1e580-9ea8-11eb-901d-e3cab5e10f71.PNG">

```python
multiclass_bar(C_mod, key_list ,target_line, explanatory_line_list)
```

複数の評価指標を同時に棒グラフで表示します

引数:
　第一引数:cannaiライブラリ　　
　第二引数:モデルid(int) または モデル名(str)のリスト
　第三引数:目的変数のラベル(str) または 列の位置(int)
　第四引数:評価指標
　
**multiclass_scatter:**

<img width="415" alt="multiclass_scatter" src="https://user-images.githubusercontent.com/81252378/114963739-59031280-9ea8-11eb-81c9-fcb6224b02b0.PNG">

```python
multiclass_scatter(C_mod, key_list ,0, ['LotArea', 'OverallQual'])
```

複数の説明変数に対し、それぞれ目的変数との二変数散布図を表示します。

引数:
　第一引数:cannaiライブラリ　　
　第二引数:モデルid(int) または モデル名(str)のリスト
　第三引数:目的変数のラベル(str) または 列の位置(int)
　第四引数:説明変数のラベル(str) または 列の位置(int)

**multiclass_matrix:**

<img width="424" alt="multiclass_matrix" src="https://user-images.githubusercontent.com/81252378/114963736-586a7c00-9ea8-11eb-8505-b91d7b023c69.PNG">

```python
multiclass_matrix(C_mod, key_list )
```

説明変数と目的変数のマトリックスを表示します。

引数:
　第一引数:cannaiライブラリ　　
　第二引数:モデルid(int) または モデル名(str)のリスト

**multiclass_rank:**

<img width="393" alt="multiclass_rank" src="https://user-images.githubusercontent.com/81252378/114963738-59031280-9ea8-11eb-9e43-98dd379d94b8.PNG">

```python
multiclass_rank(C_mod, key_list, target_line ,score_type = "abs", comvert="default", show_range="top50")
```

誤差の大きいデータから順番に棒グラフを並べます。

引数:
　第一引数:cannaiライブラリ　　
　第二引数:モデルid(int) または モデル名(str)
　第三引数:目的変数のラベル(str) または 列の位置(int)
　score_type:誤差の算出方法、absは絶対誤差の絶対値、relは相対誤差の絶対値
　comvert: logにすると対数グラフになる 
　show_range: データの表示範囲: topXは上位X個, botXは下位X個
　


## **Notice**
また、対応するグラフは順次追加してゆきます。
(ご質問、ご要望があれば教えてください)





