import jax
import jax.numpy as jnp
import haiku as hk
from typing import Callable, Optional, Union, Tuple, Any
from jaxtyping import Array, PyTree


class GaussianFourierEmbedding(hk.Module):
    def __init__(
        self,
        output_dim: int = 128,
        learnable=False,
        name: str = "gaussian_fourier_embedding",
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
            name (str, optional): Name of the module. Defaults to "gaussian_fourier_embedding".
        """
        super().__init__(name=name)
        self.output_dim = output_dim
        self.learnable = learnable

    def __call__(self, inputs):
        half_dim = self.output_dim // 2 + 1
        B = hk.get_parameter(
            "B", [half_dim, inputs.shape[-1]], init=hk.initializers.RandomNormal()
        )
        if not self.learnable:
            B = jax.lax.stop_gradient(B)
        term1 = jnp.cos(2 * jnp.pi * jnp.dot(inputs, B.T))
        term2 = jnp.sin(2 * jnp.pi * jnp.dot(inputs, B.T))
        out = jnp.concatenate([term1, term2], axis=-1)
        return out[..., : self.output_dim]
