
class Label_descriptions:
    def __init__(self, train_fn,test_fn):
    LOGGER.info('Load labels\' data')
    LOGGER.info('-------------------')

    def load(data_fn):# train_fn /test_fn
        # Load train dataset and count labels
        df = pd.read_csv(data_fn)
        cols = df.columns
        label_cols = list(cols[1:])

        # df = df.sample(frac=1).reset_index(drop=True) #shuffle rows 会不会是因为这里？
        df['one_hot_labels'] = list(df[label_cols].values)

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
        self.margins = [(0, len(frequent)+len(few)+len(true_zero))]
        k = 0
        for group in [frequent, few, true_zero]:
            self.margins.append((k, k+len(group)))
            for concept in group:
                self.label_ids[concept] = k
                k += 1
        self.margins[-1] = (self.margins[-1][0], len(frequent)+len(few)+len(true_zero))


        # Compute label descriptors representations
        label_terms = []
        self.label_terms_text = []
        for i, (label, index) in enumerate(self.label_ids.items()):
            label_terms.append([token for token in word_tokenize(data[label]['label']) if re.search('[A-Za-z]', token)])
            self.label_terms_text.append(data[label]['label'])
        self.label_terms_ids = self.vectorizer.vectorize_inputs(label_terms,
                                                                max_sequence_size=Configuration['sampling']['max_label_size'],
                                                                features=['word'])

        # Eliminate labels out of scope (not in datasets)
        self.labels_cutoff = len(frequent) + len(few) + len(true_zero)

        self.label_terms_ids = self.label_terms_ids[:self.labels_cutoff]
        label_terms = label_terms[:self.labels_cutoff]

        pdb.set_trace()
        LOGGER.info('Labels shape:    {}'.format(len(label_terms)))
        LOGGER.info('Frequent labels: {}'.format(len(frequent)))
        LOGGER.info('Few labels:      {}'.format(len(few)))
        LOGGER.info('Zero labels:     {}'.format(len(true_zero)))