import os
import numpy as np
import pickle
import tensorflow as tf


root_dir = os.path.abspath('../')
data_dir = os.path.join(root_dir, 'data')
word_embd_dir = os.path.join(data_dir, 'glove.6B')
res_dir = os.path.join(root_dir, 'res')
train_filename = os.path.join(data_dir,'en-ud-train.conllu')
dev_filename = os.path.join(data_dir,'en-ud-dev.conllu')
word_embedding_filename = os.path.join(word_embd_dir,'glove.6B.50d.txt')
pickle_dict_word_filename = os.path.join(res_dir,'word_index.pickle')
vocab_filename = os.path.join(res_dir,'vocab.npy')
embeddings_filename = os.path.join(res_dir,'embeddings.npy')

word_dict = {}
lemma_dict = {}
pos_tag_dict = {}
xpos_tag_dict = {}
x_train = []
y_train = []
x_test = []
y_test = []
vocabulary = []
word_to_index = {}
embeddings = []


learning_rate = 1e-3 # This learning rate was decided after babysitting the learning process
lr_decay = 0.5 # Again, the decay for learning rate was decided after babysitting the training loss
n_epochs = 5

X = tf.placeholder(tf.float32, shape=[None, 103]) # Input X
Y = tf.placeholder(tf.int64, shape=[None]) # Input labels for batch


keep_prob = tf.placeholder(tf.float32) # Used Dropout to add generalization, instead of L2 regularization

weights = {
    'W1': tf.Variable(tf.random_normal([103, 400])),
    'W2': tf.Variable(tf.random_normal([400, 200])),
    'W3': tf.Variable(tf.random_normal([200, 2]))
}

weights['W1'] = tf.get_variable('W1', shape=[103,400], initializer= tf.contrib.layers.xavier_initializer())
weights['W2'] = tf.get_variable('W2', shape=[400,200], initializer= tf.contrib.layers.xavier_initializer())
weights['W3'] = tf.get_variable('W3', shape=[200,2], initializer= tf.contrib.layers.xavier_initializer())

biases = {
    'b1': tf.Variable(tf.zeros([400])),
    'b2': tf.Variable(tf.zeros([200])),
    'b3': tf.Variable(tf.zeros([2]))
}

def neural_net(X, y):
    """
    3 layer feed forward neural net, using dropout instead of regularization.
    """
    fc1 = tf.nn.relu(tf.matmul(X, weights['W1']) + biases['b1'])
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.nn.relu(tf.matmul(fc1_dropout, weights['W2']) + biases['b2'])
    fc2_dropout = tf.nn.dropout(fc2, keep_prob)

    out = tf.nn.relu(tf.matmul(fc2_dropout, weights['W3']) + biases['b3'])

    # Calculate softmax loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=out))

    return out, cross_entropy_loss

out, loss = neural_net(X, Y)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) # Used Adam Optimizer
predicted_labels = tf.argmax(out, 1)
correct_predictions = tf.equal(tf.argmax(out, 1), Y)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


def loadWordEmbeddings(filename):
    vocabulary = []
    word_to_index = {}
    embeddings = []
    i = 0
    with open(filename,'r') as f:
        for line in f.readlines():
            one_record = line.strip().split(" ")
            vocabulary.append(one_record[0])
            word_to_index[one_record[0]] = len(vocabulary)-1
            embeddings.append(one_record[1:])
            i += 1
            if i%50 == 0:
                print "Embedding of {0} words done".format(str((i+1)*50))
    print "Loaded embeddings!!"
    return vocabulary, word_to_index, embeddings



def process_data(filename, mode):
    """
    Data pre-processing and training
    :param filename: File containing raw data (Train or Test)
    :param mode: TEST or TRAIN mode
    :return:
    """
    predictions_on_test = []
    lables_on_test = []
    x = []
    y = []
    i = 0
    with open(filename,'r') as f:
        for line in f.readlines():
            # Do not process comments and newlines
            if not line.startswith("#"):
                # Segregate sentences
                if not line.startswith("\n"):
                    i += 1
                    # Create a feature vector for each word in current sentence
                    data_point = np.array([])
                    one_record = line.strip().split("\t")
                    if not pos_tag_dict.has_key(one_record[3]):
                        pos_tag_dict[one_record[3]] = pos_tag_dict.__len__()
                    if not xpos_tag_dict.has_key(one_record[4]):
                        xpos_tag_dict[one_record[4]] = xpos_tag_dict.__len__()
                    if word_to_index.has_key(one_record[1]) and word_to_index.has_key(one_record[2]):
                        data_point = np.append(data_point, np.array([one_record[0]]))
                        data_point = np.append(data_point, embeddings[word_to_index[one_record[1]]])
                        data_point = np.append(data_point, embeddings[word_to_index[one_record[2]]])
                        data_point = np.append(data_point, np.array([pos_tag_dict[one_record[3]]]))
                        data_point = np.append(data_point, np.array([xpos_tag_dict[one_record[4]]]))
                        x.append(data_point)
                        if 'Number=Plur' in one_record[5].split("|"):
                            y.append(0)
                        else:
                            y.append(1)
                    else:
                        if not mode == "TRAIN":
                            if 'Number=Plur' in one_record[5].split("|"):
                                lables_on_test.append(0)
                            else:
                                lables_on_test.append(1)

                            predictions_on_test.append(1)
                else:
                    # Batching for one sentence
                    # Send for training
                    if not len(x) == 0:
                        nd_x = np.asarray(x)
                        nd_y = np.asarray(y)
                        if mode == "TRAIN":
                            if i % 200 == 0 and i > 0:
                                cur_loss = sess.run(loss, feed_dict={X:nd_x, Y:nd_y, keep_prob:0.5})
                                print "Curr loss {0}".format(str(cur_loss))
                            sess.run(optimizer, feed_dict={X:nd_x, Y:nd_y, keep_prob:0.5})
                        else:
                            prediction_labels = sess.run(predicted_labels, feed_dict={X: nd_x, Y: nd_y, keep_prob: 1.0})
                            predictions_on_test += prediction_labels.tolist()
                            lables_on_test += y

                        x = []
                        y = []

    if mode == "TEST":
        # Compute the accuracy now
        pred_arr = np.asarray(predictions_on_test)
        actual_arr = np.asarray(lables_on_test)
        assert (len(pred_arr) == len(actual_arr))
        acc = np.mean(pred_arr == actual_arr)
        print "Final Accuracy on test data: " + str(acc)

#########
# Pickle the numpy array for pre-trained word embeddings.
vocabulary, word_to_index, embeddings = loadWordEmbeddings(word_embedding_filename)
print "Now pickling word_to_index dict..."
with open(pickle_dict_word_filename, 'wb') as handle:
    pickle.dump(word_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
print "Pickling word_to_index dict complete."
print "Now pickling embeddings array..."
np.save(embeddings_filename, embeddings, allow_pickle=True)
print "Pickling embeddings array complete."

# Load pickled data
with open(pickle_dict_word_filename, 'rb') as handle:
    word_to_index = pickle.load(handle)
embeddings = np.load(embeddings_filename)

# Main training loop
for i in range(n_epochs):
    print "Starting Epoch Number %s now...." % (i)
    learning_rate *= lr_decay
    process_data(train_filename, "TRAIN")

# Get accuarcy on test
process_data(dev_filename, "TEST")

