from typing import Optional

from .common import Dataset, Hyperparameters, \
    mk_checkpoint_callback, mk_tensorboard_callback

from .multilabel import evaluate, mk_model, train
from .multiclass_single_label import mk_model as sl_mk_model, \
    evaluate as sl_evaluate, \
    train as sl_train

from tensorflow.keras.models import Model


PATH_PREFIX = '/Users/chrisw/Projects/pmp/fsdp'
LOG_DIR = '{}/tb_logs'.format(PATH_PREFIX)
CHECKPOINT_DIR = '{}/checkpoints'.format(PATH_PREFIX)

EMBEDDING_DIM = 768
NUM_FOS_CLASSES = 19
NUM_CATEGORIES_CLASSES = 176


def _mk_model_name(class_type: str, hyperparameters: Hyperparameters) -> str:
    components = [class_type]

    for k, v in sorted(hyperparameters.__dict__().items(), key=lambda x: x[0]):
        components.append("{}={}".format(k, v))

    return '_'.join(components)


def _mk_checkpoint_path(experiment_name: str) -> str:
    return CHECKPOINT_DIR + '/' + experiment_name + '/' + '.ckpt'


def _load_model_from_checkpoint(class_type: str, hyperparameters: Hyperparameters) -> Model:
    model_name = _mk_model_name(class_type, hyperparameters)
    checkpoint_path = _mk_checkpoint_path(model_name)

    if class_type == 'pcat':
        model_fn = sl_mk_model
    else:
        model_fn = mk_model

    model = model_fn(
        EMBEDDING_DIM,
        NUM_FOS_CLASSES if class_type == "fos" else NUM_CATEGORIES_CLASSES,
        hyperparameters
    )
    model.load_weights(checkpoint_path)

    return model


def train_fos(
        hyperparameters: Hyperparameters,
        num_epochs: int = 20
):
    train_path = '{}/data/all_papers/train.jsonl'.format(PATH_PREFIX)
    eval_path = '{}/data/all_papers/eval.jsonl'.format(PATH_PREFIX)

    train_set = Dataset(train_path, 'emb', 'fos')
    eval_set = Dataset(eval_path, 'emb', 'fos')

    print("Size of train_set={}".format(len(train_set)))
    print("Size of eval_set={}".format(len(eval_set)))

    print("Building tb callbacks...")
    model_name = _mk_model_name("fos", hyperparameters)

    tb_callback = mk_tensorboard_callback(LOG_DIR, model_name)
    ckpt_callback = mk_checkpoint_callback(_mk_checkpoint_path(model_name))

    print("Building model...")
    model = mk_model(EMBEDDING_DIM, NUM_FOS_CLASSES, hyperparameters)

    print("Training model...")
    train(
        model,
        train_set,
        eval_set,
        hyperparameters=hyperparameters,
        num_epochs=num_epochs,
        callbacks=[ckpt_callback, tb_callback]
    )


def train_cats(
        hyperparameters: Hyperparameters,
        num_epochs: int = 20,
        train_set: Optional[Dataset] = None,
        eval_set: Optional[Dataset] = None
):
    if not train_set:
        train_path = '{}/data/train.jsonl'.format(PATH_PREFIX)
        train_set = Dataset(train_path, 'emb', 'cats')
    if not eval_set:
        eval_path = '{}/data/eval.jsonl'.format(PATH_PREFIX)
        eval_set = Dataset(eval_path, 'emb', 'cats')

    print("Size of train_set={}".format(len(train_set)))
    print("Size of eval_set={}".format(len(eval_set)))

    print("Building tb callback...")
    model_name = _mk_model_name("cats", hyperparameters)

    tb_callback = mk_tensorboard_callback(LOG_DIR, model_name)
    ckpt_callback = mk_checkpoint_callback(_mk_checkpoint_path(model_name))

    print("Building model...")
    model = mk_model(EMBEDDING_DIM, NUM_CATEGORIES_CLASSES, hyperparameters)
    model.summary()

    print("Training model...")
    train(
        model,
        train_set,
        eval_set,
        hyperparameters=hyperparameters,
        num_epochs=num_epochs,
        callbacks=[ckpt_callback, tb_callback]
    )

def train_pcat(
        hyperparameters: Hyperparameters,
        num_epochs: int = 20,
        train_set: Optional[Dataset] = None,
        eval_set: Optional[Dataset] = None
):
    if not train_set:
        train_path = '{}/data/primary_cats/train.jsonl'.format(PATH_PREFIX)
        train_set = Dataset(train_path, 'emb', 'pcat')
    if not eval_set:
        eval_path = '{}/data/primary_cats/eval.jsonl'.format(PATH_PREFIX)
        eval_set = Dataset(eval_path, 'emb', 'pcat')

    print("Size of train_set={}".format(len(train_set)))
    print("Size of eval_set={}".format(len(eval_set)))

    print("Building tb callback...")
    model_name = _mk_model_name("pcat", hyperparameters)

    tb_callback = mk_tensorboard_callback(LOG_DIR, model_name)
    ckpt_callback = mk_checkpoint_callback(_mk_checkpoint_path(model_name))

    print("Building model...")
    model = sl_mk_model(EMBEDDING_DIM, NUM_CATEGORIES_CLASSES, hyperparameters)
    model.summary()

    print("Training model...")
    sl_train(
        model,
        train_set,
        eval_set,
        hyperparameters=hyperparameters,
        num_epochs=num_epochs,
        callbacks=[ckpt_callback, tb_callback]
    )


def evaluate_fos(hyperparameters: Hyperparameters):
    train_path = '{}/data/all_papers/train.jsonl'.format(PATH_PREFIX)
    train_set = Dataset(train_path, 'emb', 'fos')
    eval_path = '{}/data/all_papers/eval.jsonl'.format(PATH_PREFIX)
    eval_set = Dataset(eval_path, 'emb', 'fos')
    test_path = '{}/data/all_papers/test.jsonl'.format(PATH_PREFIX)
    test_set = Dataset(test_path, 'emb', 'fos')

    print("Size of test_set={}".format(len(test_set)))
    model = _load_model_from_checkpoint('fos', hyperparameters)

    evaluate(model, train_set, eval_set, test_set)


def evaluate_cats(hyperparameters: Hyperparameters):
    train_path = '{}/data/train.jsonl'.format(PATH_PREFIX)
    train_set = Dataset(train_path, 'emb', 'cats')
    eval_path = '{}/data/eval.jsonl'.format(PATH_PREFIX)
    eval_set = Dataset(eval_path, 'emb', 'cats')
    test_path = '{}/data/test.jsonl'.format(PATH_PREFIX)
    test_set = Dataset(test_path, 'emb', 'cats')

    print("size of test_set={}".format(len(test_set)))
    model = _load_model_from_checkpoint('cats', hyperparameters)

    evaluate(model, train_set, eval_set, test_set)


def evaluate_pcat(hyperparameters: Hyperparameters):
    train_path = '{}/data/primary_cats/train.jsonl'.format(PATH_PREFIX)
    train_set = Dataset(train_path, 'emb', 'pcat')
    eval_path = '{}/data/primary_cats/eval.jsonl'.format(PATH_PREFIX)
    eval_set = Dataset(eval_path, 'emb', 'pcat')
    test_path = '{}/data/primary_cats/test.jsonl'.format(PATH_PREFIX)
    test_set = Dataset(test_path, 'emb', 'pcat')

    print("size of test_set={}".format(len(test_set)))
    model = _load_model_from_checkpoint('pcat', hyperparameters)

    sl_evaluate(model, train_set, eval_set, test_set)
