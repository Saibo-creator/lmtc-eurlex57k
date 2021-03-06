import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pickle
import json
import re
import time
import tempfile
import glob
import tqdm
import pdb
import sys

import torch
import torch.utils.data as data_utils
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss, BCELoss
from transformers import AutoModelForSequenceClassification
from tqdm.auto import trange

import numpy as np
from tempfile import TemporaryFile, NamedTemporaryFile
from copy import deepcopy
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.utils import Sequence
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_score, recall_score
# import tensorflow as tf
from json_loader import JSONLoader
from vectorizer import HgBERTVectorizer
from data import DATA_SET_DIR, MODELS_DIR
from configuration import Configuration
from metrics import mean_recall_k, mean_precision_k, mean_ndcg_score, mean_rprecision_k
from neural_networks.lmtc_networks.document_classification import DocumentClassification

LOGGER = logging.getLogger(__name__)


class LMTC:

    def __init__(self):
        super().__init__()

        self.vectorizer = HgBERTVectorizer()
        print("Loaded tokenizer from ", Configuration["model"]["uri"])

        self.load_label_descriptions()

    def load_label_descriptions(self):
        LOGGER.info('Load labels\' data')
        LOGGER.info('-------------------')

        # Load train dataset and count labels
        train_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'train', '*.json'))
        train_counts = Counter()
        for filename in tqdm.tqdm(train_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    train_counts[concept] += 1

        train_concepts = set(list(train_counts))

        frequent, few = [], []
        for i, (label, count) in enumerate(train_counts.items()):
            if count > Configuration['sampling']['few_threshold']:
                frequent.append(label)
            else:
                few.append(label)

        # Load dev/test datasets and count labels
        rest_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'dev', '*.json'))
        rest_files += glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'test', '*.json'))
        rest_concepts = set()
        for filename in tqdm.tqdm(rest_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    rest_concepts.add(concept)

        # Load label descriptors
        with open(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'],
                               '{}.json'.format(Configuration['task']['dataset']))) as file:
            data = json.load(file)
            none = set(data.keys())

        none = none.difference(train_concepts.union((rest_concepts)))
        parents = []
        for key, value in data.items():
            parents.extend(value['parents'])
        none = none.intersection(set(parents))

        # Compute zero-shot group
        zero = list(rest_concepts.difference(train_concepts))
        true_zero = deepcopy(zero)
        zero = zero + list(none)

        # Compute margins for frequent / few / zero groups
        self.label_ids = dict()
        self.margins = [(0, len(frequent) + len(few) + len(true_zero))]
        k = 0
        for group in [frequent, few, true_zero]:
            self.margins.append((k, k + len(group)))
            for concept in group:
                self.label_ids[concept] = k
                k += 1
        self.margins[-1] = (self.margins[-1][0], len(frequent) + len(few) + len(true_zero))

        # Compute label descriptors representations
        label_terms = []
        self.label_terms_text = []
        for i, (label, index) in enumerate(self.label_ids.items()):
            label_terms.append([token for token in word_tokenize(data[label]['label']) if re.search('[A-Za-z]', token)])
            self.label_terms_text.append(data[label]['label'])
        self.label_terms_ids = self.vectorizer.produce_label_term_ids(label_terms,
                                                                      max_sequence_size=Configuration['sampling'][
                                                                          'max_label_size'],
                                                                      features=['word'])

        # Eliminate labels out of scope (not in datasets)
        self.labels_cutoff = len(frequent) + len(few) + len(true_zero)

        self.label_terms_ids = self.label_terms_ids[:self.labels_cutoff]
        label_terms = label_terms[:self.labels_cutoff]
        # pdb.set_trace()

        LOGGER.info('Labels shape:    {}'.format(len(label_terms)))
        LOGGER.info('Frequent labels: {}'.format(len(frequent)))
        LOGGER.info('Few labels:      {}'.format(len(few)))
        LOGGER.info('Zero labels:     {}'.format(len(true_zero)))

    def load_dataset(self, dataset_name):
        """
        Load dataset and return list of documents
        :param dataset_name: the name of the dataset,model_type:BERT ou sth 
        :return: list of Document objects
        """
        filenames = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], dataset_name, '*.json'))
        loader = JSONLoader()

        documents = []
        for filename in tqdm.tqdm(sorted(filenames)):
            document = loader.read_file(filename)
            documents.append(document)

        return documents

    def process_dataset(self, documents):
        """
         Process dataset documents (samples) and create targets
         :param documents: list of Document objects
         :return: samples, targets
         """
        samples = []
        targets = []
        for document in tqdm.tqdm(documents):
            samples.append(document.tokens)
            targets.append(document.tags)

        del documents
        return samples, targets

    def encode_dataset(self, sequences, tags):
        samples = self.vectorizer.vectorize_inputs(sequences=sequences,
                                                   max_sequence_size=Configuration['sampling']['max_sequence_size'])

        targets = np.zeros((len(sequences), len(self.label_ids)), dtype=np.int32)
        for i, (document_tags) in enumerate(tags):
            for tag in document_tags:
                if tag in self.label_ids:
                    targets[i][self.label_ids[tag]] = 1

        del sequences, tags

        return samples, targets  # bert&BERT,

    def train(self, only_create_new_generator, not_save_new_generator, use_torch):
        if use_torch:
            LOGGER.info("########### Use Pytorch as Backend #############")
        else:
            LOGGER.info("########### Use Tensorflow  as Backend #############")
        LOGGER.info(Configuration)
        LOGGER.info('\n---------------- Train Starting ----------------')
        for param_name, value in Configuration['model'].items():
            LOGGER.info('\t{}: {}'.format(param_name, value))
        for param_name, value in Configuration['sampling'].items():
            LOGGER.info('\t{}: {}'.format(param_name, value))

        # Load training/validation data
        LOGGER.info('Load training/validation data')
        LOGGER.info('------------------------------')

        train_documents_fn = "data/generators/train_documents.pickle"
        val_documents_fn = "data/generators/val_documents.pickle"
        test_documents_fn = "data/generators/test_documents.pickle"
        if (os.path.exists(train_documents_fn) and os.path.exists(val_documents_fn) and os.path.exists(
                test_documents_fn)):
            print("train val test documents alreay exist, load them now.")
            with open(train_documents_fn, "rb") as f:
                train_documents = pickle.load(f)
            with open(val_documents_fn, "rb") as f:
                val_documents = pickle.load(f)
            with open(test_documents_fn, "rb") as f:
                test_documents = pickle.load(f)
        else:
            val_documents = self.load_dataset('dev')
            train_documents = self.load_dataset('train')
            test_documents = self.load_dataset('test')
            with open(train_documents_fn, "wb") as f:
                pickle.dump(train_documents, f)
            with open(test_documents_fn, "wb") as f:
                pickle.dump(test_documents, f)
            with open(val_documents_fn, "wb") as f:
                pickle.dump(val_documents, f)

        start_time = time.time()

        if use_torch:
            val_samples, val_tags = self.process_dataset(val_documents)
            val_samples, val_targets = self.encode_dataset(val_samples, val_tags)
            input_ids = val_samples['input_ids']
            attention_masks = val_samples['attention_mask']
            # val_inputs = torch.tensor(input_ids)
            # val_masks = torch.tensor(attention_masks)
            val_targets = torch.tensor(val_targets)
            val_dataset = data_utils.TensorDataset(input_ids, attention_masks, val_targets)
            val_dataloader = data_utils.DataLoader(val_dataset, batch_size=Configuration['model']['batch_size'],
                                                   shuffle=True)
            print("finished vectorize val")


        else:  # tensorflow

            vectorized_data_fn = "data/generators/vectorized_data_{}_{}_{}.pickle".format(
                Configuration['model']['uri'].split("/")[::-1][0],
                Configuration['model']['batch_size'], Configuration["sampling"]["max_sequence_size"])

            if os.path.exists(vectorized_data_fn):

                print("vectorized data alreay exist, load them now.")

                with open(vectorized_data_fn, "rb") as f:

                    val_samples, val_targets, val_generator, train_generator, test_samples, test_targets = pickle.load(
                        f)

            else:

                val_samples, val_tags = self.process_dataset(val_documents)

                val_generator = SampleGenerator(val_samples, val_tags, experiment=self,

                                                batch_size=Configuration['model']['batch_size'])

                val_samples, val_targets = self.encode_dataset(val_samples, val_tags)

                # for eval

                print("finished vectorize val")

                train_samples, train_tags = self.process_dataset(train_documents)

                train_generator = SampleGenerator(train_samples, train_tags, experiment=self,

                                                  batch_size=Configuration['model']['batch_size'])

                print("finished vectorize train")

                test_samples, test_tags = self.process_dataset(test_documents)

                test_samples, test_targets = self.encode_dataset(test_samples, test_tags)

                print("finished vectorize test")

                with open(vectorized_data_fn, "wb") as f:
                    pickle.dump((val_samples, val_targets, val_generator, train_generator, test_samples, test_targets),
                                f)

            total_time = time.time() - start_time

            LOGGER.info('\nTotal vectorization Time: {} hours'.format(total_time / 3600))

        if only_create_new_generator:
            sys.exit()

        if use_torch:
            LOGGER.info('----------Compile Pytorch model--------------------')
            NUM_LABELS = list(self.label_ids.keys())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
            print("torch.cuda.is_available: ", torch.cuda.is_available(), "with", n_gpu)

            model = AutoModelForSequenceClassification.from_pretrained(Configuration['model']['uri'])

            # parallel_model = torch.nn.DataParallel(model) # Encapsulate the model
            parallel_model = model
            parallel_model.cuda()

            # setting custom optimization parameters. You may implement a scheduler here as well.
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=Configuration['model']['lr'])
            # Store our loss and accuracy for plotting
            train_loss_set = []

            # Number of training epochs (authors recommend between 2 and 4)
            epochs = Configuration['model']['epochs']

            for epoch__ in trange(epochs, desc="Epoch"):

                # Training

                # Set our model to training mode (as opposed to evaluation mode)
                parallel_model.train()

                # Tracking variables
                tr_loss = 0  # running loss
                nb_tr_steps = 0

                # Train the data for one epoch
                for step, batch in enumerate(val_dataloader):
                    # Add batch to GPU
                    batch = tuple(t.to(device) for t in batch)
                    # Unpack the inputs from our dataloader
                    train_input_ids, train_input_mask, train_tags = batch
                    """
                    train_samples=torch.tensor(train_samples).to(torch.int64)
                    train_tags=torch.tensor(train_tags).to(torch.int64)
                    """
                    # Clear out the gradients (by default they accumulate)
                    optimizer.zero_grad()

                    # # Forward pass for multiclass classification
                    # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    # loss = outputs[0]
                    # logits = outputs[1]
                    pdb.set_trace()
                    # Forward pass for multilabel classification
                    outputs = parallel_model(train_input_ids)
                    logits = outputs[0]
                    loss_func = BCEWithLogitsLoss()
                    loss = loss_func(logits.view(-1, NUM_LABELS), train_tags.type_as(logits).view(-1,
                                                                                                  NUM_LABELS))  # convert labels to float for calculation
                    # loss_func = BCELoss() 
                    # loss = loss_func(torch.sigmoid(logits.view(-1,NUM_LABELS)),b_labels.type_as(logits).view(-1,NUM_LABELS)) #convert labels to float for calculation
                    train_loss_set.append(loss.item())

                    # Backward pass
                    loss.mean().backward()
                    # Update parameters and take a step using the computed gradient
                    optimizer.step()
                    # scheduler.step()
                    # Update tracking variables
                    tr_loss += loss.item()
                    nb_tr_steps += 1

                    print("Train loss: {}".format(tr_loss / nb_tr_steps))

                ###############################################################################

                # Validation

                # Put model in evaluation mode to evaluate loss on the validation set
                parallel_model.eval()

                # Variables to gather full output
                logit_preds, true_labels, pred_labels = [], [], []

                # Predict
                for i, batch in enumerate(val_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    # Unpack the inputs from our dataloader
                    val_samples, val_tags = batch
                    val_samples = torch.tensor(val_samples).to(torch.int64)
                    val_tags = torch.tensor(val_tags).to(torch.int64)
                    with torch.no_grad():
                        # Forward pass
                        outs = parallel_model(val_samples, val_tags)
                        b_logit_pred = outs[0]
                        val_predictions = torch.sigmoid(b_logit_pred)

                        b_logit_pred = b_logit_pred.detach().cpu().numpy()
                        val_predictions = val_predictions.to('cpu').numpy()
                        val_tags = val_tags.to('cpu').numpy()

                    # tokenized_texts.append(b_input_ids)
                    logit_preds.append(b_logit_pred)
                    true_labels.append(val_tags)
                    pred_labels.append(val_predictions)

                # Flatten outputs
                val_predictions = [predication for batch in pred_labels for predication in batch]
                val_targets = [label for batch in true_labels for label in batch]

                ###############################  eval  phase  ######################################

                LOGGER.info('Calculate performance on valid data')
                LOGGER.info('------------------------------')
                # val_predictions = model.predict(val_samples,
                #                             batch_size=Configuration['model']['batch_size']
                #                             if Configuration['model']['architecture'] == 'BERT'
                #                                 or Configuration['model']['token_encoding'] == 'elmo' else None)

                self.calculate_performance(predictions=val_predictions, true_targets=val_targets)




        else:  # Tensorflow as backend
            # strategy = tf.distribute.MirroredStrategy()
            # with strategy.scope():

            LOGGER.info('Compile neural network')
            LOGGER.info('------------------------------')

            network = DocumentClassification(self.label_terms_ids)

            network.compile(lr=Configuration['model']['lr'],
                            freeze_pretrained=Configuration["model"]["freeze_pretrained"])

            network.model.summary(line_length=200, print_fn=LOGGER.info)

            with tempfile.NamedTemporaryFile(delete=True) as w_fd:
                weights_file = w_fd.name

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint(filepath=weights_file, monitor='val_loss', mode='auto',
                                               verbose=1, save_best_only=True, save_weights_only=True)

            # Fit model
            LOGGER.info('Fit model')
            LOGGER.info('-----------')
            start_time = time.time()

            try:
                fit_history = network.model.fit_generator(train_generator,
                                                          validation_data=val_generator,  # To speed up the validation
                                                          workers=os.cpu_count() // 2,
                                                          steps_per_epoch=len(train_generator),
                                                          epochs=Configuration['model']['epochs'],
                                                          callbacks=[early_stopping,
                                                                     Calculate_performance(val_samples, val_targets,
                                                                                           self)],
                                                          # callbacks=[early_stopping, Calculate_performance(val_generator,list(self.label_ids.keys()))],
                                                          verbose=True)
                best_epoch = np.argmin(fit_history.history['val_loss']) + 1
                n_epochs = len(fit_history.history['val_loss'])
                val_loss_per_epoch = '- ' + ' '.join(
                    '-' if fit_history.history['val_loss'][i] < np.min(fit_history.history['val_loss'][:i])
                    else '+' for i in range(1, len(fit_history.history['val_loss'])))
                LOGGER.info('\nBest epoch: {}/{}'.format(best_epoch, n_epochs))
                LOGGER.info('Val loss per epoch: {}\n'.format(val_loss_per_epoch))

            except KeyboardInterrupt:
                LOGGER.info("skip rest of training")

            # network.model.save_pretrained(os.path.join(MODELS_DIR))

            del train_generator

            ###############################  eval  phase  ######################################

            LOGGER.info('Calculate performance on valid data')
            LOGGER.info('------------------------------')
            val_samples_array = [val_samples[input_type] for input_type in val_samples.keys()]
            val_predictions = network.model.predict(val_samples_array,
                                                    batch_size=Configuration['model']['batch_size'])

            self.calculate_performance(predictions=val_predictions, true_targets=val_targets)

            LOGGER.info('Calculate performance on test data')
            LOGGER.info('------------------------------')
            test_samples_array = [test_samples[input_type] for input_type in test_samples.keys()]
            test_predictions = network.model.predict(test_samples_array,
                                                     batch_size=Configuration['model']['batch_size'])
            self.calculate_performance(predictions=test_predictions, true_targets=test_targets)

        total_time = time.time() - start_time
        LOGGER.info('\nTotal Training Time: {} hours'.format(total_time / 3600))

    def calculate_performance(self, predictions, true_targets, verbose=True):
        pred_targets = (predictions > 0.5).astype('int32')

        template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'
        true_targets = np.array(true_targets)
        # Overall
        for labels_range, frequency, message in zip(self.margins,
                                                    ['Overall', 'Frequent', 'Few', 'Zero'],
                                                    ['Overall', 'Frequent Labels (>=50 Occurrences in train set)',
                                                     'Few-shot (<=50 Occurrences in train set)',
                                                     'Zero-shot (No Occurrences in train set)']):
            start, end = labels_range
            LOGGER.info(message)
            LOGGER.info('----------------------------------------------------')
            for average_type in ['micro', 'macro', 'weighted']:
                p = precision_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                r = recall_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                f1 = f1_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))

            if verbose:
                for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                    r_k = mean_recall_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                    p_k = mean_precision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                    rp_k = mean_rprecision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                    ndcg_k = mean_ndcg_score(true_targets[:, start:end], predictions[:, start:end], k=i)
                    LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
                LOGGER.info('----------------------------------------------------')


class Calculate_performance(Callback):
    """

  Arguments:
  """

    def __init__(self, true_samples, true_targets, lmtc):
        # def __init__(self, val_generator, class_):
        super(Calculate_performance, self).__init__()
        self.true_samples = true_samples
        self.true_targets = true_targets
        self.lmtc = lmtc

    def on_epoch_end(self, epoch, logs=None):
        # predictions = self.model.predict(self.true_samples)
        true_samples_array = [self.true_samples[input_type] for input_type in self.true_samples.keys()]
        with torch.no_grad():
            predictions = self.model.predict(true_samples_array,
                                             batch_size=Configuration['model']['batch_size'])

            self.lmtc.calculate_performance(predictions, self.true_targets)


class SampleGenerator(Sequence):
    '''
    Generates data for Keras
    :return: x_batch, y_batch
    '''

    def __init__(self, samples, targets, experiment, batch_size=32, shuffle=True):
        """Initialization"""
        self.data_samples = samples
        self.targets = targets  # actually tags
        self.batch_size = batch_size
        self.indices = np.arange(len(samples))
        self.experiment = experiment
        self.shuffle = shuffle

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of batch's sequences + tags
        samples = [self.data_samples[k] for k in indices]
        targets = [self.targets[k] for k in indices]
        # Vectorize inputs (x,y)
        x_batch, y_batch = self.experiment.encode_dataset(samples, targets)  # targets are actually tags

        input_ids = x_batch["input_ids"].numpy()
        attention_mask = x_batch["attention_mask"].numpy()

        return [input_ids, attention_mask], np.array(y_batch, dtype=np.int32)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser()
    # 首先是mandatory parameters
    parser.add_argument('--only_create_new_generator', action='store_true', help='create_new_generator')
    parser.add_argument('--not_save_new_generator', action='store_true', help='not_save_new_generator')
    parser.add_argument('--t', action='store_true', help='use torch as backend')
    args = parser.parse_args()

    not_save_new_generator = args.not_save_new_generator
    only_create_new_generator = args.only_create_new_generator
    use_torch = args.t
    Configuration.configure()
    LMTC().train(only_create_new_generator, not_save_new_generator, use_torch)
