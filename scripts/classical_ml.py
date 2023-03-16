import random
import time
from pathlib import Path
from collections.abc import Callable
from copy import deepcopy

import joblib
import typer
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import fasttext.util
import spacy
from sklearnex import patch_sklearn


patch_sklearn()


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


ROOT_FOLDER = Path(__file__).parent if Path(__file__).parent.name != 'scripts' else Path(__file__).parent.parent
FULL_DATASET_FILE = ROOT_FOLDER / 'dataset.csv'


nlp = spacy.load('en_core_web_sm')
fasttext_model = fasttext.load_model(ROOT_FOLDER / 'cc.en.300.bin')


app = typer.Typer(add_completion=False)


def _load_dataset():
    df = pd.read_csv(FULL_DATASET_FILE)
    df['text'].apply(lambda i: ' '.join([word.text for word in nlp(i)]).lower())
    return df


def _train_validate_split(model: BaseEstimator, data_train: list, target_train: list, data_val: list,
                          target_val: list, scorer: Callable) -> dict:
    """Fit predict model on current split and return score and prediction for validation set"""
    data_train, data_val = np.array(data_train), np.array(data_val)
    target_train, target_val = np.array(target_train), np.array(target_val)

    # Fit model in current fold
    start_time = time.time()
    model.fit(data_train, target_train)
    end_time = time.time()

    # predict for out-fold and save it for validation
    pred_val = model.predict(data_val)

    # Score for out-fold
    score_fold = scorer(target_val, pred_val)

    return {
        'pred_val': pred_val,
        'score': score_fold,
        'time': end_time - start_time,
    }


def _cv_kfold(model: BaseEstimator, data: list, target: list, scorer: Callable, k: int = 5, random_state=42) -> dict:
    """Fit predict model multiple times with k-fold cross validation"""
    random_instance = np.random.RandomState(random_state)

    data = np.array(data)
    target = np.array(target)

    pred_train = np.empty(data.shape[0], dtype=data.dtype)

    split_scores = []
    times = []

    pred_split_train = np.empty(data.shape[0], dtype=data.dtype)

    kf = KFold(n_splits=k, shuffle=True, random_state=random_instance)
    for i, (train_index, val_index) in enumerate(kf.split(data)):
        # select current train/val split
        data_train, data_val = data[train_index], data[val_index]
        target_train, target_val = target[train_index], target[val_index]

        # Fit model in current fold
        model_fold = deepcopy(model)
        fold_result = _train_validate_split(
            model_fold,
            data_train, target_train,
            data_val, target_val,
            scorer,
        )

        times.append(fold_result['time'])
        pred_val = fold_result['pred_val']
        score_fold = fold_result['score']

        # save for out-fold validation
        pred_train[val_index] = pred_val
        pred_split_train[val_index] = pred_val

        split_scores.append(score_fold)

    return {
        'train_pred': pred_train,
        'mean_split_scores': np.mean(split_scores),
        'split_scores': split_scores,
        'oof_score': scorer(target, pred_train),
        'mean_time': np.mean(times),
        'times': times,
    }


@app.command()
def main():
    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')

    # load dataset
    df = _load_dataset()
    ft_df = np.stack([fasttext_model.get_sentence_vector(sent) for sent in df['text']])

    # define models and vectorizers
    vectorizers = {
        'Count': CountVectorizer(),
        'Tfidf': TfidfVectorizer(),
        'Hashing': HashingVectorizer(n_features=2**16),  # default 2**20, but it's too much for the task
        'FastText': None,
    }
    models = {
        'LogReg': LogisticRegression(n_jobs=4),
        'SVM': SVC(),
        'LinearSVM': LinearSVC(),
        'RandomForest': RandomForestClassifier(n_jobs=4),
        'GradientBoosting': GradientBoostingClassifier(),
        'NaiveBayes': MultinomialNB(),
        'KNeighbors': KNeighborsClassifier(n_jobs=4),
    }

    df_scores = pd.DataFrame(columns=vectorizers.keys(), index=models.keys())
    df_time = pd.DataFrame(columns=vectorizers.keys(), index=models.keys())
    scorer = lambda *x: metrics.f1_score(*x, average='macro')

    print('\n', '-' * 32, 'Training...', '-' * 32, '\n')

    joblib_memory = joblib.Memory()
    for vec_name, vec in tqdm(vectorizers.items(), total=len(vectorizers), desc='vectorizer'):
        for model_name, model in tqdm(models.items(), total=len(models), desc='models'):
            if vec_name != 'FastText':
                X, y = df['text'], df['label'].to_numpy(str)
                pipeline = Pipeline(
                    steps=[
                        ('vec', vec),
                        ('cls', model)
                    ],
                    memory=joblib_memory,
                )
            else:
                X, y = ft_df, df['label'].apply(lambda x: {'negative': -1, 'neutral': 0, 'positive': 1}[x]).to_numpy(int)
                pipeline = Pipeline(
                    steps=[
                        ('cls', model)
                    ],
                    memory=joblib_memory,
                )

            try:
                kfold_result = _cv_kfold(pipeline, X, y, scorer=scorer, k=5)
                df_scores.loc[model_name, vec_name] = kfold_result['oof_score']
                df_time.loc[model_name, vec_name] = kfold_result['mean_time']
            except Exception as e:
                print('ERROR:', e)
                df_scores.loc[model_name, vec_name] = None
                df_time.loc[model_name, vec_name] = None

    print('\nScores:')
    print(df_scores)
    for r in df_scores.iterrows():
        print(r'\text{' + r[0] + '}', '&', ' & '.join(f'{i:.3f}' if i else r'\textit{\tiny (not applicable)}' for i in r[1]), r'\\')

    print('\nMean time:')
    print(df_time)
    for r in df_time.iterrows():
        print(r'\text{' + r[0] + '}', '&', ' & '.join(f'{i:.3f}' if i else r'\textit{\tiny (not applicable)}' for i in r[1]), r'\\')


if __name__ == '__main__':
    app()
