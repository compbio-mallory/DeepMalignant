import tensorflow as tf
from tensorflow.keras import models
from gat_conv import GATConv


class GAT_AE(models.Model):
    """
    Minimal GAT autoencoder:
      Encoder: GATConv -> GATConv
      Decoder: tied weights (transpose)
    Returns:
      latent, latent, reconstruction (kept compatible with your old call pattern)
    """
    def __init__(self, in_dim: int, hidden1: int, hidden2: int,
                 dropout_rate: float = 0.0):
        super().__init__()

        # Encoder
        self.conv1 = GATConv(
            in_dim, hidden1, heads=1, activation='elu',
            skip_connection=False, _attention=True, _alpha=True,
            dropout_rate=dropout_rate
        )
        self.conv2 = GATConv(
            hidden1, hidden2, heads=1, activation=None,
            skip_connection=False, _attention=False, _alpha=True,
            dropout_rate=dropout_rate
        )

        # Decoder (tied weights)
        self.conv3 = GATConv(
            hidden2, hidden1, heads=1, activation='elu',
            skip_connection=False, _attention=True, _alpha=False,
            dropout_rate=dropout_rate
        )
        self.conv4 = GATConv(
            hidden1, in_dim, heads=1, activation=None,
            skip_connection=False, _attention=False, _alpha=True,
            dropout_rate=dropout_rate
        )

    def call(self, inputs, training=False):
        features, adj_norm, adj = inputs  # adj_norm kept for compatibility

        # NOTE: Your GATConv uses `adj` (sparse) not adj_norm.
        h1 = self.conv1([features, adj])
        h2 = self.conv2([h1, adj])

        # Tie weights safely (only if kernels exist and shapes match)
        # (This keeps your original behavior but avoids silent shape issues.)
        if hasattr(self.conv2, "kernel") and hasattr(self.conv3, "kernel"):
            if self.conv3.kernel.shape == tf.transpose(self.conv2.kernel).shape:
                self.conv3.kernel.assign(tf.transpose(self.conv2.kernel))

        if hasattr(self.conv1, "kernel") and hasattr(self.conv4, "kernel"):
            if self.conv4.kernel.shape == tf.transpose(self.conv1.kernel).shape:
                self.conv4.kernel.assign(tf.transpose(self.conv1.kernel))

        # Decoder: optionally reuse encoder attention
        h3 = self.conv3([h2, adj], tied_attention=self.conv1.attentions)
        h4 = self.conv4([h3, adj])

        return h2, h2, h4
