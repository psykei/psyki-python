import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import tensorflow as tf
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import Session
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.adam import AdamOptimizer
from psyki.logic.lyrics import LOSS_MODE
from psyki.logic.lyrics.domain import Domain
from psyki.logic.lyrics.functions import AbstractFunction, FeedForwardNN
from psyki.logic.lyrics.logic import setTNorm, SS
from psyki.logic.lyrics.parser import constraint
from psyki.logic.lyrics.predicate import Predicate
from psyki.logic.lyrics.world import World

UNSUPERVISED_SIZE = 200
SUPERVISED_SIZE = 20
TEST_SIZE = 200
KB_LOSS_WEIGHT = 10E-2
LEARNING_RATE = 10E-3


disable_eager_execution()

allX, ally = make_circles(n_samples=UNSUPERVISED_SIZE + SUPERVISED_SIZE + TEST_SIZE, shuffle=True, noise=0.,
                          random_state=None)
X_unsup_np = allX[0:UNSUPERVISED_SIZE]
X_sup_np, y_sup_np = allX[UNSUPERVISED_SIZE:UNSUPERVISED_SIZE + SUPERVISED_SIZE], ally[
                                                                                  UNSUPERVISED_SIZE:UNSUPERVISED_SIZE + SUPERVISED_SIZE]
X_sup = tf.cast(X_sup_np, tf.float32)
y_sup = tf.reshape(tf.cast(y_sup_np, tf.float32), [-1, 1])

X_test_np, y_test_np = allX[UNSUPERVISED_SIZE + SUPERVISED_SIZE:], ally[UNSUPERVISED_SIZE + SUPERVISED_SIZE:]
X_test = tf.cast(X_test_np, tf.float32)
y_test = tf.reshape(tf.cast(y_test_np, tf.float32), [-1, 1])


class IsClose(AbstractFunction):

    def __call__(self, a: tf.Tensor, b: tf.Tensor):
        dist = tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1))
        return tf.where(dist < 0.12, tf.ones_like(dist), tf.zeros_like(dist))


is_close = IsClose()
is_A = FeedForwardNN(input_shape=[2], output_size=1, layers=[30])
X_sup_np = X_sup_np.astype(np.float32)
X = np.concatenate((X_sup_np, X_unsup_np), axis=0)
X = X.astype(np.float32)

World.reset()
World._evaluation_mode = LOSS_MODE
setTNorm(id=SS, p=1)

Points = Domain(label="Points", data=X)
SPoints = Domain(label="SPoints", data=X_sup_np, father=Points)

R1 = Predicate("A", domains=["Points", ], function=is_A)
R2 = Predicate("isClose", domains=["Points", "Points"], function=is_close)
R3 = Predicate("SA", domains=["Points", ], function=lambda x: tf.squeeze(y_sup))

c_m = constraint("forall p: forall q: isClose(p,q) -> (A(p) <-> A(q))")
c_s = constraint("forall p: SA(p) <-> A(p)", {"p": SPoints})

loss_pre_vincoli = c_s

activate_rules = tf.keras.backend.placeholder(dtype=tf.bool, shape=[])
lr = tf.keras.backend.placeholder(dtype=tf.float32, shape=[])
loss_post_vincoli = loss_pre_vincoli + KB_LOSS_WEIGHT * c_m
loss = tf.cond(activate_rules, lambda: loss_post_vincoli, lambda: loss_pre_vincoli)
train_op = AdamOptimizer(lr).minimize(loss)

test_outputs = is_A(X_test)[:, 0:1]
test_predictions = tf.where(test_outputs > 0.5, tf.ones_like(test_outputs, dtype=tf.float32),
                            tf.zeros_like(test_outputs, dtype=tf.float32))
accuracy = tf.reduce_sum(tf.cast(tf.equal(test_predictions, y_test), tf.float32)) / TEST_SIZE

sess = Session(config=ConfigProto(device_count={'GPU': 0}))
sess.run(global_variables_initializer())

epochs = 10000
flag = True
feed_dict = {activate_rules: flag, lr: LEARNING_RATE}

# while True:
for i in range(epochs):
    _, acc, ll = sess.run((train_op, accuracy, loss), feed_dict)
    if (ll < 0.1 or acc > 0.99) and flag:
        print(f"training ended at epoch {i} - Loss: {ll} - Accuracy: {acc}")
        break
    if ll < 0.3:
        xtt, pred = sess.run((X_test, test_predictions))
        pred = np.reshape(pred, [-1])
        # plt.scatter(xtt[pred == 0, 0], xtt[pred == 0, 1],color="red", marker="x", label="not A")
        # plt.scatter(xtt[pred == 1, 0], xtt[pred == 1, 1], color="green", marker="o", label="A")
        # plt.legend()
        # plt.show()
        flag = True
        feed_dict = {activate_rules: flag, lr: 0.01}
    print(f"epoch {i} - Loss: {ll} - Accuracy: {acc}")

X_test, pred = sess.run((X_test, test_predictions))
pred = np.reshape(pred, [-1])
plt.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1], color="red", marker="x", label="not A"),
plt.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color="green", marker="o", label="A"),
plt.legend()
plt.show()
