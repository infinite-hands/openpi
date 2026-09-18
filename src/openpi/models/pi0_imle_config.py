"""Config for the IMLE-VLA action head (arXiv:2609.10915)."""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

import openpi.models.pi0_config as pi0_config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_imle import Pi0IMLE


@dataclasses.dataclass(frozen=True)
class Pi0ImleConfig(pi0_config.Pi0Config):
    """pi0.5 carrying a single-step cIMLE generator instead of the flow-matching head.

    model_type stays PI05 so the pi0.5 checkpoint, tokenizer and transforms all apply unchanged:
    the generator IS the pretrained action expert, read at a fixed timestep.
    """

    pi05: bool = True
    # The paper's sample factor m: candidates drawn per training sample for the nearest-neighbour
    # assignment. m=1 degenerates to L2 regression and collapses to the conditional mean; the paper
    # settles on 2 as the balance between mode coverage and the cost of the extra expert passes.
    imle_sample_factor: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.imle_sample_factor < 2:
            raise ValueError(
                f"imle_sample_factor must be at least 2 (got {self.imle_sample_factor}); m=1 is "
                "plain L2 regression, which is the mode collapse cIMLE exists to avoid"
            )

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0IMLE":
        from openpi.models.pi0_imle import Pi0IMLE

        return Pi0IMLE(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the whole VLM backbone; train the action expert and its projections.

        OpenPI stacks the VLM and the expert in one Gemma module and names the expert's leaves with
        an `llm ... _1` suffix, so the backbone is every llm leaf without that suffix plus the
        SigLIP tower. action_in_proj, action_out_proj and the time MLP sit outside both and stay
        trainable -- they are part of the head the paper fine-tunes.
        """
        if "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
            raise ValueError("Pi0ImleConfig trains the action expert directly; use non-LoRA variants")
        backbone = nnx.Any(nnx_utils.PathRegex(".*img.*"), nnx_utils.PathRegex(".*llm.*"))
        return nnx.All(backbone, nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))
