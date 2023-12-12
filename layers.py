from flax import linen as nn
from jax.nn.initializers import lecun_normal
import numpy as np
from ssm import discretize, K_conv, log_step_initializer, causal_convolution, scan_SSM
import jax

class SSMLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        """
        Initializes the necessary values for a SSM.
        """
        self.A = self.param("A", lecun_normal(), (self.N, self.N))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.D = self.param("D", nn.initializers.ones, (1,))

        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = np.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step)
        self.K = K_conv(*self.ssm. self.l_max)

        self.x_k_1 = self.variable("cache", "cache_x_k", np.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN mode
            return causal_convolution(u, self.K) + self.D * u

        else:
            # RNN mode
            x_k, y_s = scan_SSM(*self.ssm, np.expand_dims(u, axis=1), self.x_k_1.value)
            self.x_k_1.value = x_k if self.is_mutable_collection("cache") else self.x_k_1.value
            return y_s.flatten().real + self.D * u

class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    layer: dict
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True
    decode: bool = True

    def setup(self):
        self.seq = self.layer_cls(**self.layer, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)

        if self.glu:
            self.out2 = nn.Dense(self.d_model)

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic= not self.training
        )

    def __call__(self, x):
        skip = x

        if self.prenorm:
            x = self.norm(x)

        x = self.seq(x)
        x = self.drop(nn.gelu(x))

        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2)
        else:
            x = self.out(x)

        x = skip + self.drop(x)

        if not self.prenorm:
            x = self.norm
        return x


class Embedding(nn.Embed):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)


class StackedModel(nn.Module):
    layer_cls: nn.Module
    layer: dict
    d_output: int
    d_model: int
    n_layer: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False # use nn.Embed instead of nn.Dense encoder
    classification: bool = False
    training: bool = True
    decode: bool = False

    def setup(self):
        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)

        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                glu=self.glu,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0

            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])

        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)

        if self.classification:
            x = np.mean(x, axis=0)

        x = self.decoder(x)

        return nn.log_softmax(x, axis=-1)