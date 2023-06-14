import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from psyki.logic.lyrics.domain import Domain
from psyki.logic.lyrics.functions import BinaryIndexFunction
from psyki.logic.lyrics.parser import constraint
from psyki.logic.lyrics.predicate import Predicate
from psyki.logic.lyrics.utils import heardEnter
from psyki.logic.lyrics.world import World

disable_eager_execution()
people_repr_size = 1


class Equal:

    def __call__(self, a, b):
        return tf.cast(tf.equal(a, b), tf.float32)


class IndexingFunction:

    def __init__(self, k):
        self.k = k
        self.var = tf.Variable(initial_value=-4 * tf.ones([k * k]))

    def call(self, a, b):
        a = tf.cast(a, tf.int32)
        b = tf.cast(b, tf.int32)
        idx = self.k * a + b
        return tf.sigmoid(tf.gather(self.var, idx))

k = 6

World.reset()
Domain(label="People", data=tf.zeros([0, 1]), size=0)
World.individuals(label="Marco", domain="People", value=[0])
World.individuals(label="Giuseppe", domain="People", value=[1])
World.individuals(label="Michelangelo", domain="People", value=[2])
World.individuals(label="Francesco", domain="People", value=[3])
World.individuals(label="Franco", domain="People", value=[4])
World.individuals(label="Andrea", domain="People", value=[5])

fo = BinaryIndexFunction("fo", k, k)
gfo = BinaryIndexFunction("gfo", k, k)
equal = Equal()

Predicate(label="fatherOf", domains=("People", "People"), function=fo)
Predicate(label="grandFatherOf", domains=("People", "People"), function=gfo)
Predicate(label="is", domains=("People", "People"), function=equal)

constraint("fatherOf(Marco, Giuseppe)")
constraint("fatherOf(Giuseppe, Michelangelo)")
constraint("fatherOf(Giuseppe, Francesco)")
constraint("fatherOf(Franco, Andrea)")
constraint("forall x: forall y: forall z: (fatherOf(x,y) and not is(x,z)) -> not fatherOf(z,y)", 0.1)
constraint("forall x: forall y: forall z: fatherOf(x,z) and fatherOf(z,y) -> grandFatherOf(x,y)", 0.1)
constraint("forall x: forall y: fatherOf(x,y) -> not grandFatherOf(x,y)", 0.1)
constraint("forall x: not fatherOf(x,x)")
constraint("forall x: not grandFatherOf(x,x)")
constraint("forall x: forall y: grandFatherOf(x,y) -> not fatherOf(x,y)", 0.1)
constraint("forall x: forall y: fatherOf(x,y) -> not fatherOf(y,x)", 0.1)
constraint("forall x: forall y: grandFatherOf(x,y) -> not grandFatherOf(y,x)", 0.1)
constraint("forall x: forall y: grandFatherOf(x,y) -> not fatherOf(y,x)", 0.1)
constraint("forall x: forall y: fatherOf(x,y) -> not grandFatherOf(y,x)", 0.1)

loss = World.loss()
train_op = GradientDescentOptimizer(1).minimize(loss)

sess = Session()
sess.run(global_variables_initializer())

epochs = 10000
for i in range(epochs):
    _, l = sess.run((train_op, loss))
    if heardEnter():
        break
    if i % 1000 == 0:
        print(l)

# print("fatherOf(Marco,Giuseppe)=%f" % (sess.run(Query("fatherOf(Marco,Giuseppe)").tensor)))
# print("fatherOf(Andrea,Giuseppe)=%f" % (sess.run(Query("fatherOf(Andrea,Giuseppe)").tensor)))
# print("fatherOf(Giuseppe,Michelangelo)=%f" % (sess.run(Query("fatherOf(Giuseppe,Michelangelo)").tensor)))
# print("grandFatherOf(Marco, Michelangelo)=%f" % (sess.run(Query("grandFatherOf(Marco, Michelangelo)").tensor)))
# print("forall x: forall y: forall z: grandFatherOf(x,z) and fatherOf(y,z) -> fatherOf(x,y)=%f" %
#       (sess.run(
#           Query("forall x: forall y: forall z: grandFatherOf(x,z) and fatherOf(y,z) -> fatherOf(x,y)").tensor)))

# learn
