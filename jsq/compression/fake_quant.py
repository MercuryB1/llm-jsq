import torch
from torch import nn
from functools import partial
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, w_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (w_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, w_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (w_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(x, a_bits=8):
    x_shape = x.shape
    x.view(-1, x_shape[-1])
    scales = x.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (a_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    x.div_(scales).round_().mul_(scales)
    return x


@torch.no_grad()
def quantize_activation_per_tensor_absmax(x, a_bits=8):
    x_shape = x.shape
    x.view(-1, x_shape[-1])
    scales = x.abs().max()
    q_max = 2 ** (a_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    x.div_(scales).round_().mul_(scales)
    return x


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        a_bits=8,
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features),
                    dtype=torch.float16,
                    requires_grad=False,
                ),
            )
        else:
            self.register_buffer("bias", None)
        
        if act_quant == 'per_token':
            self.act_quant_name = "per_token"
            self.a_bits = a_bits
            self.act_quant = partial(quantize_activation_per_token_absmax, a_bits=self.a_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.a_bits = a_bits
            self.act_quant = partial(quantize_activation_per_tensor_absmax, a_bits=self.a_bits)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")
        
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = None
            self.output_quant = lambda x : x
            

    def to(self, *args, **kwargs):
        super(QuantLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self
    
    
    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y
    
    
    @staticmethod
    def from_float(
        module, weight_quant="per_channel", w_bits=8, act_quant="per_token", a_bits=8, quantize_output=False, 
    ):
        assert isinstance(module, nn.Linear)
        quant_module = QuantLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            a_bits=a_bits
        )
        if weight_quant == "per_channel":
            quant_module.w_bits = w_bits
            quant_module.weight = quantize_weight_per_channel_absmax(
                w=module.weight,
                w_bits=w_bits,
            )
        elif weight_quant == "pre_tensor":
            quant_module.w_bits = w_bits
            quant_module.weight = quantize_weight_per_tensor_absmax(
                w=module.weight,
                w_bits=w_bits,
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        quant_module.weight_quant_name = weight_quant
        if module.bias is not None:
            quant_module.bias = module.bias
        return quant_module
    
    def __repr__(self):
        return f"QuantLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, weight_bits={self.w_bits}, act_quant={self.act_quant_name}, act_bits={self.a_bits}, output_quant={self.output_quant_name})"
    
    
def quantize_llama_like(
    model, weight_quant="per_channel", w_bits=8, act_quant="per_token", a_bits=8, quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP
    )
    
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )
    
    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, )):
            m.gate_proj = QuantLinear.from_float(
                m.gate_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits
            )
            m.up_proj = QuantLinear.from_float(
                m.up_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits
            )
            m.down_proj = QuantLinear.from_float(
                m.down_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = QuantLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                act_quant=act_quant,
                a_bits=a_bits,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = QuantLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                act_quant=act_quant,
                a_bits=a_bits,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = QuantLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                w_bits=w_bits,
                act_quant=act_quant,
                a_bits=a_bits,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = QuantLinear.from_float(
                m.o_proj, 
                weight_quant=weight_quant,
                w_bits=w_bits,
                act_quant=act_quant,
                a_bits=a_bits,
            )
    return model
    
    
def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    
@torch.no_grad()
def quantize_layer(module, weight_quant='per_channel', w_bits=8, act_quant='per_token', a_bits=8, quantize_bmm_input=False):
    for name, m in module.named_modules():
        if isinstance(m, LlamaAttention):
            m.q_proj = QuantLinear.from_float(
                m.q_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits, quantize_output=quantize_bmm_input)
            m.k_proj = QuantLinear.from_float(
                m.k_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits, quantize_output=quantize_bmm_input)
            m.v_proj = QuantLinear.from_float(
                m.v_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits, quantize_output=quantize_bmm_input)
            m.o_proj = QuantLinear.from_float(
                m.o_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits,)
        elif isinstance(m, LlamaMLP):
            m.gate_proj = QuantLinear.from_float(
                m.gate_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits,)
            m.down_proj = QuantLinear.from_float(
                m.down_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits,)
            m.up_proj = QuantLinear.from_float(
                m.up_proj, weight_quant=weight_quant, w_bits=w_bits, act_quant=act_quant, a_bits=a_bits,)
    return module