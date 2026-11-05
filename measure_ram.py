import os, resource, warnings
warnings.filterwarnings("ignore")

def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(f"start:                  {peak():.0f} MB")

import torch
print(f"after import torch:     {peak():.0f} MB")

from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM
print(f"after imports:          {peak():.0f} MB")

tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/work/Qwen3-0.6B"))
model = ExecuTorchModelForCausalLM.from_pretrained(os.path.expanduser("~/work/qwen3_pte"))
print(f"after loading model:    {peak():.0f} MB")

model.text_generation(tokenizer=tok, prompt="Photosynthesis is", echo=False, max_seq_len=40)
print(f"after one generation:   {peak():.0f} MB")