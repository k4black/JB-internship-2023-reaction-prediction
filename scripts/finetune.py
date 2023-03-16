import json
import random
import time
from pathlib import Path

import neptune.new as neptune
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
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw


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


aug_synonym = naw.SynonymAug(aug_src='wordnet')
aug_spelling = naw.SpellingAug(aug_p=0.3)
aug_typo = nac.KeyboardAug(aug_char_p=0.3, aug_word_p=0.3)
aug_delete_word = naw.RandomWordAug()
aug_swap_words = naw.RandomWordAug(action='swap')
aug_crop_words = naw.RandomWordAug(action='crop')


def _augment_text(example):
    text = example['text']
    r = random.randint(1, 6)
    if r == 1:
        text = aug_synonym.augment(text)
    elif r == 2:
        text = aug_spelling.augment(text)
    elif r == 2:
        text = aug_typo.augment(text)
    elif r == 4:
        text = aug_delete_word.augment(text)
    elif r == 5:
        try:
            text = aug_crop_words.augment(text)
        except:
            pass
    else:
        text = aug_swap_words.augment(text)
    example['text'] = text[0]
    return example


def _augument_dataset(dataset, upsample=False, extend=True):

    if upsample:
        positive_examples = dataset['train'].filter(lambda x: x['label'] == 'positive' or x['label'] == 2)
        neutral_examples = dataset['train'].filter(lambda x: x['label'] == 'neutral' or x['label'] == 1)
        negative_examples = dataset['train'].filter(lambda x: x['label'] == 'negative' or x['label'] == 0)
        neutral_examples = concatenate_datasets(
            [neutral_examples for _ in range(len(positive_examples) // len(neutral_examples))], axis=0)
        negative_examples = concatenate_datasets(
            [negative_examples for _ in range(len(positive_examples) // len(negative_examples))], axis=0)
        neutral_examples = neutral_examples.map(_augment_text, load_from_cache_file=False, batched=False)
        negative_examples = negative_examples.map(_augment_text, load_from_cache_file=False, batched=False)
        dataset['train'] = concatenate_datasets([positive_examples, neutral_examples, negative_examples], axis=0)

    if extend:
        dataset['train'] = concatenate_datasets([dataset['train'] for _ in range(5)], axis=0)
        dataset['train'] = dataset['train'].map(_augment_text, load_from_cache_file=False, batched=False)

    dataset['train'] = dataset['train'].shuffle(seed=42, load_from_cache_file=False)

    return dataset


def _load_split_dataset(tokenizer, extend=False, upsample=False):
    cl = ClassLabel(names=['negative', 'neutral', 'positive'])
    label2id, id2label = {n: i for i, n in enumerate(cl.names)}, {i: n for i, n in enumerate(cl.names)}

    dataset = load_dataset('csv', data_files=str(FULL_DATASET_FILE.absolute()))
    dataset = dataset.cast_column('label', cl)
    dataset = dataset['train'].train_test_split(test_size=0.2, stratify_by_column='label')

    if extend or upsample:
        dataset = _augument_dataset(dataset, upsample=upsample, extend=extend)

    def tokenize_function(examples):
        examples = tokenizer(examples['text'], truncation=True, padding='do_not_pad')
        return examples
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset, label2id, id2label


def _load_k_fold_dataset(tokenizer, cv_folds=5, extend=False, upsample=False):
    cl = ClassLabel(names=['negative', 'neutral', 'positive'])
    label2id, id2label = {n: i for i, n in enumerate(cl.names)}, {i: n for i, n in enumerate(cl.names)}

    dataset = load_dataset('csv', data_files=str(FULL_DATASET_FILE.absolute()))
    dataset = dataset.cast_column('label', cl)
    dataset = dataset['train']

    kf = sklearn.model_selection.KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for fold_num, (train_index, test_index) in enumerate(kf.split(np.arange(len(dataset)))):
        sub_dataset = DatasetDict({
            'train': dataset.select(train_index),
            'test': dataset.select(test_index),
        })

        if extend or upsample:
            sub_dataset = _augument_dataset(sub_dataset, upsample=upsample, extend=extend)

        def tokenize_function(examples):
            examples = tokenizer(examples['text'], truncation=True, padding='do_not_pad')
            return examples
        tokenized_sub_dataset = sub_dataset.map(tokenize_function, batched=True)

        yield fold_num, tokenized_sub_dataset, label2id, id2label


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

    val_data = trainer.predict(dataset['test'])
    test_f1 = val_data.metrics['test_f1']
    test_predictions = val_data.predictions.argmax(axis=-1)
    test_labels = val_data.label_ids
    print('f1', test_f1)
    print(sklearn.metrics.classification_report(test_labels, test_predictions, digits=4))

    return test_f1, test_predictions, test_labels


@app.command()
def main(
        base_model: str = typer.Option('roberta-base', help='Pretrained model to finetune: HUB or Path'),
        config_name: str = typer.Option('default', help='Config name to use: default, updated, large'),
        cross_validation: int = typer.Option(0, help='Number of folds for cross validation'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True, help='Folder to save results'),
        save_folder: Path = typer.Option(ROOT_FOLDER / 'models', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    model_name_to_save = f'finetuning-{base_model}-config-{config_name}'.replace('/', '-')
    output_dir = str(results_folder / model_name_to_save)
    model_save_folder = save_folder / model_name_to_save

    params = EDOS_EVAL_PARAMS[config_name.split('-')[0]]  # read base config
    params.update(EDOS_EVAL_PARAMS[config_name])  # update with specific config
    do_extend = 'extend' in params.get('augment', 'none')
    do_upsample = 'upsample' in params.get('augment', 'none')

    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')
    neptune_run = neptune.init_run(tags=[f'model:{base_model}', f'conf:{config_name}'])
    nepture_object_id = neptune_run['sys/id'].fetch()

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
    )

    print('\n', '-' * 32, 'Training...', '-' * 32, '\n')
    if not cross_validation:
        # load dataset
        dataset, label2id, id2label = _load_split_dataset(tokenizer, extend=do_extend, upsample=do_upsample)
        # load pretrained model
        config = AutoConfig.from_pretrained(base_model, label2id=label2id, id2label=id2label)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)
        # train model
        test_f1, predictions, labels = _train_model(model, dataset, tokenizer, data_collator, params, neptune_run, output_dir=output_dir, model_save_folder=model_save_folder)
    else:
        neptune_run.stop()
        time.sleep(5)
        folds_f1, all_predictions, all_labels = [], [], []
        for fold_num, dataset, label2id, id2label in _load_k_fold_dataset(tokenizer, cross_validation, extend=do_extend, upsample=do_upsample):
            print('\n', '-' * 32, f'Fold {fold_num+1}/{cross_validation}', '-' * 32, '\n')
            fold_neptune_run = neptune.init_run(with_id=nepture_object_id)
            # load pretrained model
            config = AutoConfig.from_pretrained(base_model, label2id=label2id, id2label=id2label)
            model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)
            # train model
            fold_f1, fold_predictions, fold_labels = _train_model(model, dataset, tokenizer, data_collator, params, fold_neptune_run, postfix=f'fold-{fold_num}', output_dir=output_dir, model_save_folder=model_save_folder)
            folds_f1.append(fold_f1)
            all_predictions.append(fold_predictions)
            all_labels.append(fold_labels)

            fold_neptune_run.stop()
            time.sleep(5)

        test_f1 = sklearn.metrics.f1_score(np.concatenate(all_labels), np.concatenate(all_predictions), average='macro')
        neptune_run = neptune.init_run(with_id=nepture_object_id)

    print('\n', '-' * 32, 'End', '-' * 32, '\n')
    print('test_f1', test_f1)
    neptune_run['finetuning/final_f1'] = test_f1
    neptune_run['parameters'] = {
        'base_model': base_model,
        'config_name': config_name,
    }


if __name__ == '__main__':
    app()
