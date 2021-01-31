
from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam

from transformers import TFAutoModelForSequenceClassification
from transformers import TFLongformerForSequenceClassification


from configuration import Configuration
import pdb
import tensorflow as tf

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
        '''self.model = TFBertForSequenceClassification.from_pretrained(Configuration['model']['uri'], num_labels=4271)'''

        # Import the needed model(Bert, Roberta or DistilBert) with output_hidden_states=True
        if "longformer" in Configuration["model"]["uri"]:
            transformer_model = TFLongformerForSequenceClassification.from_pretrained(Configuration['model']['uri'],num_labels=4271)

        else:
            transformer_model = TFAutoModelForSequenceClassification.from_pretrained(Configuration['model']['uri'], num_labels=4271)

        if freeze_pretrained:
            """
            [ < transformers.modeling_tf_bert.TFBertMainLayer
            at
            0x169a23e10 >,
            < tensorflow.python.keras.layers.core.Dropout
            at
            0x169abde90 >,
            < tensorflow.python.keras.layers.core.Dense
            at
            0x169ac31d0 >]"""
            transformer_model.layers[0].trainable = False

        input_ids = tf.keras.Input(shape=(Configuration['sampling']['max_sequence_size'],), dtype='int32')
        attention_mask = tf.keras.Input(shape=(Configuration['sampling']['max_sequence_size'],), dtype='int32')

        transformer = transformer_model([input_ids, attention_mask],training=True)
        hidden_states = transformer[0]  # get output_hidden_states

        output = tf.keras.activations.sigmoid(hidden_states)
        self.model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)

        self.model.compile(optimizer=Adam(lr=lr), loss=loss)



