### import packages ###
import tensorflow as tf
import tensorflow_datasets as tfds

import time
import numpy as np
import matplotlib.pyplot as plt

# read data
import json

import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import re

import socket
import threading
import time

import sys
import argparse

def encode(dialogue):
  result = []
  for en in dialogue:
    result.append([tokenizer_en.vocab_size] + tokenizer_en.encode(en) + [tokenizer_en.vocab_size+1])
  return result

def encode_sent(dialogue): # list as input
  result = []
  for sen in dialogue:
      re_sen = re.sub('[.,!?]','', sen)
      temp = tokenizer_sent.encode(re_sen)
      if(len(temp) < max_len):
          temp = temp + [0] * (max_len - len(temp))
          result.append(temp)
      else:
          print("Input length over the max length")
  return result

# Model
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * int(-1e9)) # 굉장히 작은 숫자로 만듬으로써 softmax에서 거의 0에 가깝게 만든다. 즉, mask하는 효과와 같다.

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model) 1 60 512
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth) 1 8 60 64
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)


    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    # two sublayers
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    # Add & Norm after each sublayer
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)


  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)


    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inp, tar, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

# util로 빼놔야하나?
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def create_masks(inp, tar): # inp = input, tar = target
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask


def evaluate(inp_sentence):
  start_token = [tokenizer_en.vocab_size]
  end_token = [tokenizer_en.vocab_size + 1]

  # inp sentence is portuguese, hence adding the start and end token
  inp_sentence = start_token + tokenizer_en.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)

  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [tokenizer_en.vocab_size]
  output = tf.expand_dims(decoder_input, 0)

  # 최대 output 길이가 어느정도 나올지는 모르겠다.
  for i in range(200):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input,
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_en.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


def translate(sentence):
  result, attention_weights = evaluate(re.sub('[!?]', '', sentence.lower()))

  predicted_sentence = tokenizer_en.decode([i for i in  result
                                            if i < tokenizer_en.vocab_size])  # if문으로 start token이랑 end token 빼내는구나

  predicted_sentence = predicted_sentence.split('.')
  predicted_sentence = [w.strip() for w in predicted_sentence]
  predicted_sentence = [w.capitalize() for w in predicted_sentence]
  predicted_sentence = '. '.join(predicted_sentence)
  #print('Input: {}'.format(sentence))
  #print('Predicted translation: {}'.format(predicted_sentence))
  return predicted_sentence

def handle_client(client_socket, addr):
    print("New Client Here!")
    while True:
        ## Model inference
        data = client_socket.recv(1024)
        bytes = 0
        sent = str(-1)
        if data != None:
            msg = data.decode()
            command = msg.split()
            if len(command) != 0:
                if command[0] == 'sent':
                    enc_msg = encode_sent([msg])
                    sent = np.argmax(emotion_classifier.predict(enc_msg), axis=1)
                    sent = str(sent[0])

            answer = translate(msg)
            answer = sent + " " + answer
            data = answer.encode()
            client_socket.sendall(data)

    print("One Client Left")
    client_socket.close()
    server_socket.close()

def accept_func():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    while 1:
        try:
            #클라이언트 함수가 접속하면 새로운 소켓을 반환한다.
            client_socket, addr = server_socket.accept()
        except KeyboardInterrupt:
            server_socket.close()
            print("Keyboard interrupt")

        print("클라이언트 핸들러 스레드로 이동 됩니다.")
        #accept()함수로 입력만 받아주고 이후 알고리즘은 핸들러에게 맡긴다.
        t = threading.Thread(target=handle_client, args=(client_socket, addr))
        t.daemon = True
        t.start()

if __name__ == "__main__":
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Training chatbot"
    )

    parser.add_argument("--training", action="store_true", default=False)

    args = parser.parse_args()

    ### You should replace HOST and PORT as your own HOST and PORT.
    ### If you want to use the application anywhere outside by internet, use public IP.
    HOST = ''
    PORT = 10004
    chatbot_checkpoint_path = "./Checkpoints/kurisu/chatbot"
    emotion_classifier_path = "./Checkpoints/kurisu/emotion_classifier"
    data_path = "./Data/dialogue_example.txt"

    print("==========================tf version = {}========================".format(tf.__version__))
    print("==========================tfds version = {}========================".format(tfds.__version__))
    data = pd.read_csv(data_path, sep = '\t', engine='python')

    dialogue = []
    sent_labels = []
    for i in range(len(data)):
      dialogue.append(re.sub('[!?]', '', data['text'][i].lower()))
      sent_labels.append(np.eye(3)[data['sentiment'][i]])

    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    dial_len = [len(doc) for doc in dialogue]
    max_len = max(dial_len)

    prep_dialogue = []
    for sen in dialogue:
        prep_dialogue.append(re.sub('[.,!?]', '', sen))

    #tokenizer_sent = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #    (sent for sent in prep_dialogue), target_vocab_size = 2**13)

    tokenizer_sent = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (sent for sent in prep_dialogue), target_vocab_size = 2**13)

    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en for en in dialogue), target_vocab_size=2**13)

    dialogue_encode = encode(dialogue)
    input_dataset = []
    target_dataset = []

    for i in range(len(dialogue_encode)):
        if i == len(dialogue_encode):
            break
        if i%2 != 0:
            continue
        # As I mentioned at the beginning, one question and one answer relationship is applied.
        input_dataset.append(dialogue_encode[i])
        target_dataset.append(dialogue_encode[i+1])

    BATCH_SIZE = 4
    BUFFER_SIZE = 20000

    sum_dataset = [(input_dataset[i], target_dataset[i]) for i in range(len(input_dataset))]
    train_dataset = tf.data.Dataset.from_generator(lambda: sum_dataset, output_types= (tf.int64, tf.int64), output_shapes=([None], [None]))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print("==========================Emotion Classifier Loading========================")

    emotion_classifier = tf.keras.models.load_model(emotion_classifier_path)

    print("==========================Emotion Classifier Loading Complete========================")

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    """
    num_layers = 6
    d_model = 1024
    dff = 4096
    num_heads = 16
    dropout_rate = 0.3
    """

    input_vocab_size = tokenizer_en.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2


    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Restore trained model
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, chatbot_checkpoint_path, max_to_keep=5)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')



    if(args.training):
        EPOCHS = 200
        # The @tf.function trace-compiles train_step into a TF graph for faster
        # execution. The function specializes to the precise shape of the argument
        # tensors. To avoid re-tracing due to the variable sequence lengths or variable
        # batch sizes (the last batch is smaller), use input_signature to specify
        # more generic shapes.

        train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:] # shifted tar_input (+ 1)

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = transformer(tf.cast(inp, dtype=tf.int64), tf.cast(tar_inp,dtype=tf.int64),
                                             tf.cast(True, dtype=tf.bool),
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(tar_real, predictions)


        for epoch in range(EPOCHS):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):
                train_step(inp, tar)

                if batch % 50 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                      epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            if (epoch + 1) % 50 == 0:
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

            print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                    train_loss.result(),
                                                    train_accuracy.result()))

            print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


    else:
        # activate chatbot server
        if ckpt_manager.latest_checkpoint:
          ckpt.restore(ckpt_manager.latest_checkpoint)
          print("==========================Last checkpoint restored!========================")
        accept_func()
