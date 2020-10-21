

import tensorflow as tf
from capsule.capsule_layer import Capsule
from capsule.em_capsule_layer import EMCapsule
from capsule.primary_capsule_layer import PrimaryCapsule
from capsule.norm_layer import Norm

layers = tf.keras.layers
models = tf.keras.models


class SignCapsNet(tf.keras.Model):

    def __init__(self, routing, layers, use_bias=False):
        super(SignCapsNet, self).__init__()

        CapsuleType = {
            "rba": Capsule,
            "em": EMCapsule
        }
        self.use_bias=use_bias
        self.capsule_layers = []
        
        # Specified through the sign dataset
        in_capsule = 1
        in_dim = 1
        for i in range(len(layers)):
            num_capsules = layers[i][0]
            out_dim = layers[i][1]
            self.capsule_layers.append(
                CapsuleType[routing](
                    in_capsules=in_capsule, 
                    in_dim=in_dim, out_capsules=num_capsules, 
                    out_dim=out_dim, use_bias=self.use_bias,
                    stdev=0.5)
            )
            in_capsule = num_capsules 
            in_dim = out_dim
        
        # Output layer
        self.capsule_layers.append(
            CapsuleType[routing](
                in_capsules=in_capsule, 
                in_dim=in_dim, out_capsules=2, 
                out_dim=10, 
                use_bias=self.use_bias,
                stdev=0.5)
        )
        
        self.norm = Norm()


    def call(self, x, y):
        x = tf.expand_dims(x, axis=1)

        for capsule in self.capsule_layers:
            x = capsule(x)
        out = self.norm(x)
        return out
