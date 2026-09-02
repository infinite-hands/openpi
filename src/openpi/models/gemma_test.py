from flax import nnx
from flax.nnx import bridge
import jax

from openpi.models import gemma


def test_gemma_attention_flag_is_static_during_nnx_initialization():
    config = gemma.get_config("dummy")

    def create(rng):
        llm = bridge.ToNNX(
            gemma.Module(
                configs=[config, config],
                embed_dtype="float32",
                adarms=True,
            )
        )
        llm.lazy_init(
            rngs=nnx.Rngs(rng),
            method="init",
            use_adarms=[False, True],
        )
        return llm

    nnx.eval_shape(create, jax.random.key(0))
