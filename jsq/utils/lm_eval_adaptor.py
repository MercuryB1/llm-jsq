import transformers
import torch
# from lm_eval.base import BaseLM
from jsq.models.models_utils import BaseLM
import fnmatch
import torch.nn.functional as F

class LMEvalAdaptor(BaseLM):
    def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model_name = model_name
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        # assert isinstance(self.tokenizer, (
        #     transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
        #     transformers.T5Tokenizer, transformers.T5TokenizerFast,
        # )), "this tokenizer has not been checked for compatibility yet!"
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token
    

    # @property
    # def max_length(self):
    #     if self._max_length != -1:
    #         return self._max_length
    #     if hasattr(self.model.config, "n_ctx"):
    #         return self.model.config.n_ctx
    #     elif hasattr(self.model.config, "max_position_embeddings"):
    #         return self.model.config.max_position_embeddings
    #     elif hasattr(self.model.config, "n_positions"):
    #         return self.model.config.n_positions
    #     elif "bloom" in self.model_name:
    #         return 2048
    #     elif "llama" in self.model_name:
    #         return 2048  # TODO: did not check this
    #     elif "mpt" in self.model_name:
    #         return 2048
    #     elif "falcon" in self.model_name:
    #         return 2048
    #     else:
    #         print(self.model.config)
    #         raise NotImplementedError
    
    
    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    
    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
    
    
    # def tok_decode(self, tokens):
    #     return self.tokenizer.decode(tokens)
    
    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps)["logits"]
        # """
        # inps: a torch tensor of shape [batch, sequence]
        # the size of sequence may vary from call to call

        # returns: a torch tensor of shape [batch, sequence, vocab] with the
        # logits returned from the model
        # """
        # with torch.no_grad():
        #     if isinstance(
        #         self.model,
        #         transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
        #     ):
        #         dec_inps = torch.cat(
        #             [
        #                 torch.tensor(
        #                     self.model.generation_config.decoder_start_token_id,
        #                 )
        #                 .tile(len(inps), 1)
        #                 .to(inps),
        #                 inps,
        #             ],
        #             dim=1,
        #         )

        #         kwargs = {
        #             "decoder_input_ids": dec_inps,
        #         }
        #     else:
        #         kwargs = {}
        #     out = self.model(inps, **kwargs)[0]
        #     if (
        #         "opt" in self.model_name
        #     ):  # there are a few extra tokens in opt, which we should omit
        #         return out[:, :, :50257]
        #     else:
        #         return out  # [:, :, :self.tokenizer.vocab_size]


    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    
    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )