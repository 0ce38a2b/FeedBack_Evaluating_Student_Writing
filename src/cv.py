from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd

#https://www.kaggle.com/abhishek/creating-folds-properly-hopefully-p
def make_fold():
    df = pd.read_csv("/users10/lyzhang/opt/tiger/feedback/data/train.csv")

    df1 = pd.get_dummies(df, columns=['discourse_type']).groupby(['id'], as_index=False).sum()
    label_col = [c for c in df1.columns if c.startswith('discourse_type_') and c != 'discourse_type_num']
    col = label_col +['id']
    df1 = df1[col]
    df1.loc[:,'fold'] = -1

    mskf  = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, valid_index) in enumerate(mskf.split(df1, df1[label_col])):
        df1.loc[valid_index, 'fold'] = fold
        print(len(train_index),len(valid_index))

    df = df.merge(df1[['id', 'fold']], on='id', how='left')
    print(df.fold.value_counts())
    df.to_csv("/users10/lyzhang/opt/tiger/feedback/data/train_fold5.csv", index=False)
    exit(0)


make_fold()