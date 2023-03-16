import random
from pathlib import Path

import pandas as pd
import numpy as np
import typer


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT_FOLDER = Path(__file__).parent  # repo root folder
FULL_DATASET_FILE = ROOT_FOLDER / 'dataset.csv'


app = typer.Typer(add_completion=False)


@app.command()
def main():
    df = pd.read_csv(FULL_DATASET_FILE)

    # print dataset distribution by classes
    for label, sub_df in df.groupby('label'):
        print(label)
        print('len', len(sub_df))
        print('text len', np.mean([len(t) for t in sub_df['text']]))

    print('total')
    print('len', len(df))
    print('text len', np.mean([len(t) for t in df['text']]))


if __name__ == '__main__':
    app()
