
from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam

from transformers import TFBertForSequenceClassification


from configuration import Configuration
import pdb

class DocumentClassification:
    def __init__(self, label_terms_ids):
        super().__init__()
        self._decision_type = Configuration['task']['decision_type']
        self.n_classes = len(label_terms_ids)
        self.freeze_pretrained=Configuration['model']['freeze_pretrained']

    def __del__(self):
        K.clear_session()
        # del self.model


    def compile(self, lr, freeze_pretrained):

        loss = 'binary_crossentropy'

        # Wrap up model + Compile with optimizer and loss function
        # self.model = Model(inputs=word_inputs, outputs=[outputs])
        self.model = TFBertForSequenceClassification.from_pretrained(Configuration['model']['uri'], num_labels=4271)

        self.model.compile(optimizer=Adam(lr=lr), loss=loss)



