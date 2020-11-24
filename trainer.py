import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(120, activation='relu')
    self.d2 = Dense(50, activation='relu')
    self.d3 = Dense(20, activation='relu')
    self.d4 = Dense(1, activation='sigmoid')

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)


@tf.function
def train_step(peptids, labels, model, loss_object, optimizer):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(peptids, training=True)
        loss = loss_object(labels, predictions, sample_weight=labels + 0.15)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, predictions


@tf.function
def test_step(peptids, labels, model, loss_object):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(peptids, training=False)
    t_loss = loss_object(labels, predictions)

    return t_loss, predictions


def load_train_and_test_data():
    # training and testing using labeled set
    x_train = np.loadtxt(r"data\x_train.txt")
    x_test = np.loadtxt(r"data\x_test.txt")
    y_train = np.loadtxt(r"data\y_train.txt").reshape(-1,1)
    y_test = np.loadtxt(r"data\y_test.txt").reshape(-1,1)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(128)
    # Create an instance of the model
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return train_ds, test_ds


def run_model(train_ds, test_ds):
    model = MyModel()
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_precision = tf.keras.metrics.Precision(name='train_precision')
    train_recall = tf.keras.metrics.Recall(name='train_recall')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    test_precision = tf.keras.metrics.Precision(name='test_precision')
    test_recall = tf.keras.metrics.Recall(name='test_recall')

    EPOCHS = 10

    array_test_loss = []
    array_train_loss = []

    array_test_accuracy = []
    array_train_accuracy = []

    array_test_recall = []
    array_train_recall = []

    array_test_precision = []
    array_train_precision = []

    for epoch in range(EPOCHS):

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        train_precision.reset_states()
        train_recall.reset_states()
        test_precision.reset_states()
        test_recall.reset_states()

        for peptid, labels in train_ds:
            loss, predictions = train_step(peptid, labels, model, loss_object, optimizer)
            train_loss(loss)
            train_accuracy(labels, predictions)
            train_recall(labels, predictions)
            train_precision(labels, predictions)

        array_train_loss.append(train_loss.result())
        array_train_accuracy.append(train_accuracy.result())
        array_train_recall.append(train_recall.result())
        array_train_precision.append(train_precision.result())

        for test_peptid, test_labels in test_ds:
            t_loss, predictions = test_step(test_peptid, test_labels, model, loss_object)
            test_loss(t_loss)
            test_accuracy(test_labels, predictions)
            test_recall(test_labels, predictions)
            test_precision(test_labels, predictions)

        array_test_loss.append(test_loss.result())
        array_test_accuracy.append(test_accuracy.result())
        array_test_recall.append(test_recall.result())
        array_test_precision.append(test_precision.result())

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Recall: {train_recall.result() * 100}, '
            f'Precession: {train_precision.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}, '
            f'Test Recall: {test_recall.result() * 100}, '
            f'Test Precession: {test_precision.result() * 100}'
        )

    with open (r"data\run.txt", 'a') as f:
        f.write(f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Recall: {train_recall.result() * 100}, '
        f'Precession: {train_precision.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}, '
        f'Test Recall: {test_recall.result() * 100}, '
        f'Test Precession: {test_precision.result() * 100} \n\n')

    plt.plot(array_test_loss, label="test")
    plt.plot(array_train_loss, label="train")
    plt.legend()
    plt.title("loss over epochs")
    plt.show()
    plt.plot(array_test_accuracy, label="test")
    plt.plot(array_train_accuracy, label="train")
    plt.legend()
    plt.title("Accuracy over epochs")
    plt.show()
    plt.plot(array_test_recall, label="test")
    plt.plot(array_train_recall, label="train")
    plt.legend()
    plt.title("Recall over epochs")
    plt.show()
    plt.plot(array_test_precision, label="test")
    plt.plot(array_train_precision, label="train")
    plt.legend()
    plt.title("Precision over epochs")
    plt.show()

    # results using spike protein data
    spike_protein_data = np.loadtxt('data\spike_protein_matrix.txt')
    predictions = model(spike_protein_data, training=False)

    idx_top_five = tf.make_ndarray(tf.make_tensor_proto(predictions)).flatten().argsort()[-5:]

    spike_protein_str = open('data\spike_protein_data.txt').read().replace('\n', '')
    for i in idx_top_five:
        print(spike_protein_str[i:i+9])
        print(predictions[i])


def main():
    train_ds, test_ds = load_train_and_test_data()
    run_model(train_ds, test_ds)

if __name__ == "__main__":
    main()
