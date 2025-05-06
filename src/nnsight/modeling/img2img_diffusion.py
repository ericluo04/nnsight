from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import AutoPipelineForImage2Image
from transformers import BatchEncoding
from typing_extensions import Self
from ..intervention.contexts import InterventionTracer

from .. import util
from .mixins import RemoteableMixin


class Diffuser(util.WrapperModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.pipeline = AutoPipelineForImage2Image.from_pretrained(*args, **kwargs)
        
        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module):
                setattr(self, key, value)

        self.tokenizer = self.pipeline.tokenizer


class Img2ImgDiffusionModel(RemoteableMixin):
    
    __methods__ = {"generate": "_generate"}

    def __init__(self, *args, **kwargs) -> None:

        self._model: Diffuser = None

        super().__init__(*args, **kwargs)
        
    def _load_meta(self, repo_id:str, **kwargs):
        
        
        model = Diffuser(
            repo_id,
            device_map=None,
            low_cpu_mem_usage=False,
            **kwargs,
        )

        return model
        

    def _load(self, repo_id: str, device_map=None, **kwargs) -> Diffuser:

        model = Diffuser(repo_id, device_map=device_map, **kwargs)

        return model

    def _prepare_input(
        self,
        inputs: Union[str, List[str]],
    ) -> Any:

        if isinstance(inputs, str):
            inputs = [inputs]

        return ((inputs,), {}), len(inputs)

    def _batch(
        self,
        batched_inputs: Optional[Dict[str, Any]],
        prepared_inputs: BatchEncoding,
    ) -> torch.Tensor:

        if batched_inputs is None:

            return ((prepared_inputs, ), {})

        return (batched_inputs + prepared_inputs, )

    def _execute(self, prepared_inputs: Any, *args, **kwargs):

        return self._model.unet(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def _generate(
        self, prepared_inputs: Any, *args, seed: int = None, **kwargs
    ):

        if self._scanning():

            kwargs["num_inference_steps"] = 1

        generator = torch.Generator()

        if seed is not None:

            if isinstance(prepared_inputs, list):
                generator = [torch.Generator().manual_seed(seed) for _ in range(len(prepared_inputs))]
            else:
                generator = generator.manual_seed(seed)
            
        output = self._model.pipeline(
            prepared_inputs, *args, generator=generator, **kwargs
        )

        output = self._model(output)

        return output
        
    # add LoRA capability methods
    def load_lora_weights(self, pretrained_model_name_or_path, 
                         weight_name: str = None, 
                         adapter_name: str = "default", 
                         **kwargs):
        """
        Load LoRA weights into the model.
        
        Args:
            pretrained_model_name_or_path: Path to the LoRA weights or model directory
            weight_name: Name of the weight file (e.g., "pytorch_lora_weights.safetensors")
            adapter_name: Name to identify this specific LoRA adapter
            **kwargs: Additional arguments to pass to the underlying load_lora_weights method
        """
        return self._model.pipeline.load_lora_weights(
            pretrained_model_name_or_path,
            weight_name=weight_name,
            adapter_name=adapter_name,
            **kwargs
        )
    
    def fuse_lora(self, lora_scale: float = 1.0, adapter_names: Optional[List[str]] = None):
        """
        Fuse the LoRA weights into the base model weights for inference.
        
        Args:
            lora_scale: Scaling factor for the LoRA weights
            adapter_names: List of adapter names to fuse. If None, fuses all loaded adapters.
        """
        return self._model.pipeline.fuse_lora(lora_scale=lora_scale, adapter_names=adapter_names)
    
    def unfuse_lora(self):
        """
        Unfuse the LoRA weights from the base model weights.
        """
        return self._model.pipeline.unfuse_lora()
    
    def unload_lora_weights(self):
        """
        Unload LoRA weights from the model.
        
        Note: The diffusers API for unload_lora_weights does not accept adapter_names parameter.
        """
        return self._model.pipeline.unload_lora_weights()
    
    def set_lora_scale(self, lora_scale: float, adapter_names: Optional[List[str]] = None):
        """
        Set the scaling factor for the LoRA weights.
        
        Args:
            lora_scale: Scaling factor for the LoRA weights
            adapter_names: List of adapter names to scale. If None, scales all loaded adapters.
        """
        if hasattr(self._model.pipeline, "set_adapters_scale"):
            return self._model.pipeline.set_adapters_scale(lora_scale, adapter_names=adapter_names)
        else:
            raise AttributeError("This pipeline does not support setting LoRA scale dynamically.")


if TYPE_CHECKING:

    class Img2ImgDiffusionModel(Img2ImgDiffusionModel, AutoPipelineForImage2Image):

        def generate(self, *args, **kwargs) -> InterventionTracer:
            return self._model.pipeline(*args, **kwargs)
