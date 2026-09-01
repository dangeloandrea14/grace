# rmu_trainer_embed.py
import re, copy, torch
from torch import nn
from transformers import Trainer

try:
    from accelerate.utils import extract_model_from_parallel
    def _unwrap(m): return extract_model_from_parallel(m)
except Exception:
    def _unwrap(m):
        while hasattr(m, "module"): m = m.module
        return m

class RMUTrainer(Trainer):
    def __init__(
        self,
        *args,
        module_regex=r".*layers\.28$",
        trainable_params_regex=r".*layers\.(22|23|24|25|26|27|28)\.self_attn\.(q|k|v|o)_proj\.lora_[AB]\.default\.weight$",
        steering_coeff=20.0,
        alpha=1.0,   # retain (EMBED_DIFF) weight
        gamma=1.0,   # forget (RMU) weight
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.module_regex = module_regex
        self.trainable_patterns = trainable_params_regex if isinstance(trainable_params_regex, (list, tuple)) else (trainable_params_regex,)
        self.steering_coeff = steering_coeff
        self.alpha, self.gamma = alpha, gamma

        self._model_module = None
        self._ref_model = None
        self._ref_module = None
        self._control_vec = None

    # Freeze all → unfreeze only targeted params
    def create_optimizer(self):
        for p in self.model.parameters(): p.requires_grad = False
        for n, p in self.model.named_parameters():
            if any(re.fullmatch(pat, n) or re.search(pat, n) for pat in self.trainable_patterns):
                p.requires_grad = True
        super().create_optimizer()

    def _ensure_modules_and_ref(self):
        # resolve target module on train model
        if self._model_module is None:
            m = _unwrap(self.model)
            soft = {n: mod for n, mod in m.named_modules() if re.search(self.module_regex, n)}
            if len(soft) != 1:
                raise ValueError(f"Module regex '{self.module_regex}' matched {len(soft)}: {list(soft)[:8]}")
            self._model_module = next(iter(soft.values()))

        # create frozen ref copy once (on same device, eval)
        if self._ref_model is None:
            base = _unwrap(self.model)
            ref = copy.deepcopy(base).to(next(base.parameters()).device)
            ref.eval()
            self._ref_model = ref  # do NOT pass to optimizer/wrap with DDP
            # find ref's corresponding module
            soft = {n: mod for n, mod in ref.named_modules() if re.search(self.module_regex, n)}
            if len(soft) != 1:
                raise ValueError(f"[ref] Module regex '{self.module_regex}' matched {len(soft)}: {list(soft)[:8]}")
            self._ref_module = next(iter(soft.values()))

    def _forward_with_cache(self, model, inputs, module, no_grad=True):
        cache = []
        def hook(_m, _in, out): cache.append(out[0] if isinstance(out, tuple) else out)
        h = module.register_forward_hook(hook)
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**inputs)
        h.remove()
        if not cache: raise RuntimeError("Hook captured no activation.")
        return cache[0], outputs

    def _set_trainable(self, model, patterns, requires=True):
        for n, p in model.named_parameters():
            if any(re.fullmatch(pat, n) or re.search(pat, n) for pat in patterns):
                p.requires_grad = requires

    @staticmethod
    def _masked_mse_over_tokens(a, b, mask):
        valid = mask.sum(dim=-1)
        sq = torch.nn.functional.mse_loss(a, b, reduction="none")
        sq = (sq * mask.unsqueeze(-1)).mean(dim=2).sum(dim=1)
        return (sq / valid.clamp_min(1)).mean()

    def _get_control_vec(self, dim, dtype, device):
        if self._control_vec is None:
            v = torch.rand(1, 1, dim, dtype=dtype, device=device)
            v = v / torch.norm(v) * self.steering_coeff
            self._control_vec = v.detach()
        return self._control_vec.to(dtype=dtype, device=device)

    def compute_loss(self, model: nn.Module, inputs, return_outputs=False, num_items_in_batch=None):
        self._ensure_modules_and_ref()

        # --- forget (RMU) ---
        f = {k: inputs["forget"][k] for k in ("input_ids","attention_mask","labels")}
        f_act, f_out = self._forward_with_cache(model, f, self._model_module, no_grad=False)
        ctrl = f.get("control_vec", self._get_control_vec(f_act.shape[-1], f_act.dtype, f_act.device)).expand_as(f_act)
        f_mask = f["labels"] != -100
        forget_loss = self._masked_mse_over_tokens(f_act, ctrl, f_mask)

        # --- retain (EMBED_DIFF) ---
        r = {k: inputs["retain"][k] for k in ("input_ids","attention_mask","labels")}
        r_act_m, _ = self._forward_with_cache(model, r, self._model_module, no_grad=False)
        with torch.no_grad():
            r_act_ref, _ = self._forward_with_cache(self._ref_model, r, self._ref_module, no_grad=True)
        r_mask = r["labels"] != -100
        retain_loss = self._masked_mse_over_tokens(r_act_m, r_act_ref.to(r_act_m), r_mask)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, f_out) if return_outputs else loss
