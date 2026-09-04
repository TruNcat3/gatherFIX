"""End-to-end model test: real transformers BERT forward+backward on NPU with
flag_gems' gather op enabled, instrumented to prove our kernel is exercised.

wt-2026-09-04-fix: end-to-end model validation (issue #5746)

Why BERT: when token_type_ids is NOT passed (common in inference engines and
torch.export tracing), BertEmbeddings regenerates it via:
    buffered = self.token_type_ids.expand(bsz, -1)      # non-contiguous input
    buffered = torch.gather(buffered, 1, position_ids)  # <- real model gather
This is a genuine in-model torch.gather with an expanded (non-contiguous)
input tensor — exactly the class of layout the #5746 fix covers.

Oracle: identical model/seed with flag_gems gather disabled (native aclnnGather).
"""
import importlib
import torch
import torch_npu  # noqa: F401
import flag_gems
from transformers import BertModel, BertTokenizerFast

device = "npu:0"
model_path = "/root/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-BertModel/snapshots/fc08ad9cc33be9aef4f55cc80e16ef5ae3d5981c"

tok = BertTokenizerFast.from_pretrained(model_path)
enc = tok(["hello world", "gather op e2e validation test"],
          return_tensors="pt", padding=True).to(device)
# drop token_type_ids -> BertEmbeddings takes the gather path
model_inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

# ---- instrument the op via the REGISTRATION TABLE (not module attr) ----
# flag_gems' FULL_CONFIG_BY_FUNC captured the gather function object at import
# time; patching the module attribute later is invisible to the registrar.
# The reliable instrumentation point is the table entry itself.
import sys
import flag_gems as fg
g = sys.modules[fg.FULL_CONFIG_BY_FUNC["gather"][0][1].__module__]
assert hasattr(g, "gather_strided"), "this flag_gems copy does NOT contain the fix!"

calls = {"total": 0, "noncontig": 0}
orig_gather = g.gather

def spy(inp, dim, index, out=None, sparse_grad=False):
    calls["total"] += 1
    if not (inp.is_contiguous() and index.is_contiguous()):
        calls["noncontig"] += 1
    return orig_gather(inp, dim, index, out, sparse_grad)

# swap the entry the registrar will read at use_gems().__enter__
_orig_entry = fg.FULL_CONFIG_BY_FUNC["gather"][0]
fg.FULL_CONFIG_BY_FUNC["gather"][0] = (_orig_entry[0], spy)

def run():
    torch.manual_seed(42)
    model = BertModel.from_pretrained(model_path).to(device)
    model.train()
    out = model(**model_inputs).last_hidden_state
    loss = out.sum()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    del model
    return out.detach().clone(), grads, loss.item()

# 1) native oracle
hidden_ref, grads_ref, loss_ref = run()

# 2) flag_gems: ONLY the gather op (isolate from unrelated flag_gems 5.3.5 issues)
with flag_gems.use_gems(include=["gather"]):
    torch.manual_seed(42)
    hidden_g, grads_g, loss_g = run()

# restore the registration table entry
fg.FULL_CONFIG_BY_FUNC["gather"][0] = _orig_entry

# ---- compare ----
hidden_equal = torch.equal(hidden_ref.cpu(), hidden_g.cpu())
maxdiff = (hidden_ref.cpu().float() - hidden_g.cpu().float()).abs().max().item()
grad_ok, grad_worst, worst_name = True, 0.0, ""
for n in grads_ref:
    d = (grads_ref[n].cpu().float() - grads_g[n].cpu().float()).abs().max().item()
    if d > grad_worst:
        grad_worst, worst_name = d, n
    if not torch.allclose(grads_ref[n].cpu(), grads_g[n].cpu(), atol=1e-5, rtol=1e-4):
        grad_ok = False

print(f"gather op invoked           : {calls['total']} times")
print(f"  with non-contiguous layout: {calls['noncontig']}   <- strided-kernel territory")
print(f"hidden states bitwise equal : {hidden_equal} (max diff {maxdiff:.3e})")
print(f"loss native={loss_ref:.6f} gems={loss_g:.6f}")
print(f"grads allclose              : {grad_ok} (worst {grad_worst:.3e} @ {worst_name})")

ok = calls["total"] > 0 and hidden_equal and grad_ok
print("\nVERDICT:", "PASS — real model forward+backward exercised our gather op, results match native"
      if ok else "FAIL / INCONCLUSIVE")
raise SystemExit(0 if ok else 1)
