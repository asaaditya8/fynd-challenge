import sys
sys.path.append("..")

import argparse
import os
from keras.models import load_model
from src.data.loader import BATCH_SIZE, get_data
from src.data.make_dataset import TEST_SIZE
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

N_SAMPLES = {
 'train': int(3370 * (1 - TEST_SIZE)),
 'test': int(3370 * TEST_SIZE)
}

def evaluate(data_dir, model_ckpt, output_path):
    # Checkpoint already contains model definition
    model = load_model(os.path.abspath(model_ckpt))
    gen_dict = get_data(data_dir)

    # All results are going to go into this dict
    metrics = {}

    # Evaluating on both training as well as testing data
    # Want to find multiple metrics using scikit-learn
    for c in N_SAMPLES:
        y_pred = []
        y_test = []

        # For obtaining y_true manual iteration is needed
        for i in tqdm(range(N_SAMPLES[c] // BATCH_SIZE)):
            x_batch, y_batch = next(gen_dict[c])
            y_test.append( y_batch )
            y_pred.append( model.predict_on_batch(x_batch) )

        y_pred = np.concatenate(y_pred, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        # 'average' means 'class average', it handles class imbalance
        metrics[c + '_accuracy'] = accuracy_score(y_test.argmax(axis=-1), y_pred.argmax(axis=-1))[0],
        metrics[c + '_f1-score'] = f1_score(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), average='weighted')[0],
        metrics[c + '_roc-auc-score'] = roc_auc_score(y_test, y_pred, average='weighted')[0],
        metrics[c + '_log-loss'] = log_loss(y_test.argmax(axis=-1), y_pred)

    # Output json will be tracked by version control
    with open(os.path.abspath(output_path), 'w') as f:
        json.dump(metrics, f)


def main(args):
    evaluate(args.data_dir, args.ckpt_path, args.metric_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('data_dir', type=str, help='path for dataset')
    parser.add_argument('ckpt_path', type=str, help='saved model path')
    parser.add_argument('metric_path', type=str, help='path for output metric')

    args = parser.parse_args()

    main(args)