import json
import random
import time
from pathlib import Path

import neptune.new as neptune
import scipy
import sklearn.model_selection
import sklearn.metrics
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer, EarlyStoppingCallback, AutoConfig,
    TrainingArguments, DataCollatorWithPadding
)
from datasets import DatasetDict, ClassLabel, load_dataset, concatenate_datasets
from transformers.integrations import NeptuneCallback
import numpy as np
import evaluate
import typer
import torch


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


ROOT_FOLDER = Path(__file__).parent if Path(__file__).parent.name != 'scripts' else Path(__file__).parent.parent
FULL_DATASET_FILE = ROOT_FOLDER / 'dataset.csv'
with open(ROOT_FOLDER / 'params.json') as f:
    EDOS_EVAL_PARAMS = json.load(f)


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)


app = typer.Typer(add_completion=False)


def _make_efl_example(example, prompt: str, positive: bool = True):
    example['raw_label'] = example['label']
    example['label'] = 'entailment' if positive else 'not_entailment'
    example['prompt'] = f'It was {prompt}.'
    example['prompt_label'] = prompt
    return example


def _convert_dataset_to_efl(dataset, raw_cl, cl):
    _raw_negative_dataset = dataset.filter(lambda x: x['label'] == 'negative' or x['label'] == 0)
    _raw_neutral_dataset = dataset.filter(lambda x: x['label'] == 'neutral' or x['label'] == 1)
    _raw_positive_dataset = dataset.filter(lambda x: x['label'] == 'positive' or x['label'] == 2)

    efl_dataset = concatenate_datasets([
        _raw_negative_dataset.map(lambda x: _make_efl_example(x, 'negative', positive=True)),
        _raw_negative_dataset.map(lambda x: _make_efl_example(x, 'neutral', positive=False)),
        _raw_negative_dataset.map(lambda x: _make_efl_example(x, 'positive', positive=False)),

        _raw_neutral_dataset.map(lambda x: _make_efl_example(x, 'negative', positive=False)),
        _raw_neutral_dataset.map(lambda x: _make_efl_example(x, 'neutral', positive=True)),
        _raw_neutral_dataset.map(lambda x: _make_efl_example(x, 'positive', positive=False)),

        _raw_positive_dataset.map(lambda x: _make_efl_example(x, 'negative', positive=False)),
        _raw_positive_dataset.map(lambda x: _make_efl_example(x, 'neutral', positive=False)),
        _raw_positive_dataset.map(lambda x: _make_efl_example(x, 'positive', positive=True)),
    ])
    efl_dataset = efl_dataset.cast_column('label', cl)
    efl_dataset = efl_dataset.cast_column('raw_label', raw_cl)
    efl_dataset = efl_dataset.cast_column('prompt_label', raw_cl)

    efl_dataset = efl_dataset.shuffle(seed=SEED)

    return efl_dataset


def _efl_predictions_to_raw_labels(dataset, logist):
    pass


def _load_efl_split_dataset(tokenizer, extend=False, upsample=False):
    raw_cl = ClassLabel(names=['negative', 'neutral', 'positive'])
    cl = ClassLabel(names=['not_entailment', 'entailment'])
    label2id, id2label = {n: i for i, n in enumerate(cl.names)}, {i: n for i, n in enumerate(cl.names)}

    raw_dataset = load_dataset('csv', data_files=str(FULL_DATASET_FILE.absolute()))
    raw_dataset = raw_dataset['train'].train_test_split(test_size=0.2)

    efl_dataset = DatasetDict({
        'train': _convert_dataset_to_efl(raw_dataset['train'], raw_cl, cl),
        'test': _convert_dataset_to_efl(raw_dataset['test'], raw_cl, cl),
    })

    def tokenize_function(examples):
        examples = tokenizer(examples['text'], examples['prompt'], truncation=True, padding='do_not_pad')
        return examples
    tokenized_dataset = efl_dataset.map(tokenize_function, batched=True)

    return tokenized_dataset, label2id, id2label


def _load_k_fold_efl_dataset(tokenizer, cv_folds=5, extend=False, upsample=False):
    raw_cl = ClassLabel(names=['negative', 'neutral', 'positive'])
    cl = ClassLabel(names=['not_entailment', 'entailment'])
    label2id, id2label = {n: i for i, n in enumerate(cl.names)}, {i: n for i, n in enumerate(cl.names)}

    raw_dataset = load_dataset('csv', data_files=str(FULL_DATASET_FILE.absolute()))
    raw_dataset = raw_dataset['train']

    kf = sklearn.model_selection.KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    for fold_num, (train_index, test_index) in enumerate(kf.split(np.arange(len(raw_dataset)))):
        sub_efl_dataset = DatasetDict({
            'train': _convert_dataset_to_efl(raw_dataset.select(train_index), raw_cl, cl),
            'test': _convert_dataset_to_efl(raw_dataset.select(test_index), raw_cl, cl),
        })

        def tokenize_function(examples):
            examples = tokenizer(examples['text'], examples['prompt'], truncation=True, padding='do_not_pad')
            return examples
        tokenized_sub_efl_dataset = sub_efl_dataset.map(tokenize_function, batched=True)

        yield fold_num, tokenized_sub_efl_dataset, label2id, id2label


def _compute_f1_macro_metric(eval_pred):
    metric_f1 = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_f1.compute(predictions=predictions, references=labels, average='macro')


def _train_model(model, dataset, tokenizer, data_collator, params: dict, neptune_run, postfix: str = '', output_dir: str = '', model_save_folder=None):
    neptune_callback = NeptuneCallback(run=neptune_run, base_namespace=f'finetuning-{postfix}' if postfix else 'finetuning')

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to='none',

        learning_rate=params['learning_rate'],
        lr_scheduler_type='linear',
        weight_decay=0.01,

        auto_find_batch_size=True,  # divide by 2 in case of OOM
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        num_train_epochs=params['max_epochs'],
        warmup_ratio=params.get('warmup_ratio', 0.05),

        no_cuda=not IS_CUDA_AVAILABLE,
        fp16=IS_CUDA_AVAILABLE,  # always use fp16 on gpu
        fp16_full_eval=False,

        logging_strategy='steps',
        logging_steps=params['eval_steps'],
        evaluation_strategy='steps',
        eval_steps=params['eval_steps'],
        save_strategy='steps',
        save_steps=params['eval_steps'],

        metric_for_best_model='eval_f1',
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_f1_macro_metric,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=params.get('early_stopping_patience', 5)), neptune_callback],
    )

    trainer.train()

    if model_save_folder:
        model_save_folder.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_save_folder))

    # predict for test dataset; get probability of entailment, select max for each id
    entailment_dataset = dataset['test']
    val_data = trainer.predict(entailment_dataset)
    entailment_prob = scipy.special.softmax(val_data.predictions, axis=-1)[:, 1]

    ids = np.array(entailment_dataset['id'])
    test_labels, test_predictions = [], []
    for _id in set(ids):
        indexes = np.where(ids == _id)[0]
        max_prob_entailment_idx = np.argmax(entailment_prob[indexes])
        max_prob_example = entailment_dataset[int(indexes[max_prob_entailment_idx])]

        test_labels.append(max_prob_example['raw_label'])
        test_predictions.append(max_prob_example['prompt_label'])

    test_f1 = sklearn.metrics.f1_score(test_labels, test_predictions, average='macro')
    efl_f1 = val_data.metrics['test_f1']

    print('f1', test_f1)
    print('efl_f1', efl_f1)
    print(sklearn.metrics.classification_report(test_labels, test_predictions, digits=4))

    return test_f1, efl_f1, test_predictions, test_labels


@app.command()
def main(
        base_model: str = typer.Option('roberta-base', help='Pretrained model to finetune: HUB or Path'),
        config_name: str = typer.Option('efl', help='Config name to use: default, updated, large'),
        cross_validation: int = typer.Option(0, help='Number of folds for cross validation'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True, help='Folder to save results'),
        save_folder: Path = typer.Option(ROOT_FOLDER / 'models', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    model_name_to_save = f'efl-{base_model}-config-{config_name}'.replace('/', '-')
    output_dir = str(results_folder / model_name_to_save)
    model_save_folder = save_folder / model_name_to_save

    params = EDOS_EVAL_PARAMS[config_name.split('-')[0]]  # read base config
    params.update(EDOS_EVAL_PARAMS[config_name])  # update with specific config

    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')
    neptune_run = neptune.init_run(tags=['efl', f'model:{base_model}', f'conf:{config_name}'])
    nepture_object_id = neptune_run["sys/id"].fetch()

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
    )

    print('\n', '-' * 32, 'Training...', '-' * 32, '\n')
    if not cross_validation:
        # load dataset
        dataset, label2id, id2label = _load_efl_split_dataset(tokenizer)
        # load pretrained model
        config = AutoConfig.from_pretrained(base_model, label2id=label2id, id2label=id2label)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config, ignore_mismatched_sizes=True)
        # train model
        test_f1, efl_f1, predictions, labels = _train_model(model, dataset, tokenizer, data_collator, params, neptune_run, output_dir=output_dir, model_save_folder=model_save_folder)
    else:
        neptune_run.stop()
        time.sleep(5)
        folds_f1, folds_efl_f1, all_predictions, all_labels = [], [], [], []
        for fold_num, dataset, label2id, id2label in _load_k_fold_efl_dataset(tokenizer, cross_validation):
            print('\n', '-' * 32, f'Fold {fold_num+1}/{cross_validation}', '-' * 32, '\n')
            fold_neptune_run = neptune.init_run(with_id=nepture_object_id)
            # load pretrained model
            config = AutoConfig.from_pretrained(base_model, label2id=label2id, id2label=id2label)
            model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config, ignore_mismatched_sizes=True)
            # train model
            fold_f1, fold_efl_f1, fold_predictions, fold_labels = _train_model(model, dataset, tokenizer, data_collator, params, fold_neptune_run, postfix=f'fold-{fold_num}', output_dir=output_dir, model_save_folder=model_save_folder)
            folds_f1.append(fold_f1)
            folds_efl_f1.append(fold_efl_f1)
            all_predictions.append(fold_predictions)
            all_labels.append(fold_labels)

            fold_neptune_run.stop()
            time.sleep(5)

        efl_f1 = np.mean(folds_efl_f1)
        test_f1 = sklearn.metrics.f1_score(np.concatenate(all_labels), np.concatenate(all_predictions), average='macro')
        neptune_run = neptune.init_run(with_id=nepture_object_id)

    print('\n', '-' * 32, 'End', '-' * 32, '\n')
    print('test_f1', test_f1)
    print('efl_f1', efl_f1)
    neptune_run['finetuning/final_f1'] = test_f1
    neptune_run['finetuning/efl_f1'] = efl_f1
    neptune_run['parameters'] = {
        'base_model': base_model,
        'config_name': config_name,
    }


if __name__ == '__main__':
    app()
