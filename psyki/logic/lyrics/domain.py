import tensorflow as tf
from psyki.logic.lyrics.world import World


class Domain(object):
    def __init__(self, label, data, father=None, size=None):
        if label in World.domains:
            raise Exception("Domain %s already exists" % label)
        self.label = label
        self.tensor = tf.convert_to_tensor(data)
        self.ancestors = []
        if isinstance(self.tensor.get_shape()[1], int):
            self.columns = self.tensor.get_shape()[1]
        else:
            self.columns = self.tensor.get_shape()[1].value
        if father is not None:
            assert isinstance(father, Domain)
            self.ancestors = father.ancestors + [father]
            assert father.columns == self.columns
        try:
            self.size = len(data)
        except:
            if size is None:
                raise Exception("data has no len() and no size is provided")
            else:
                self.size = size
        World.domains[self.label] = self
