# Offline Smart Learning Assistant

**Namma AI Hackathon: AI for Inclusive Education (Edge AI PoC)**

Repository: https://github.com/JayaHaris2007/Offline-Smart-Learning-Assistant--ExecuTorch-PoC

An offline study aid that turns a short textbook passage into a summary, multiple-choice questions and an answer key. It runs a small language model locally through **ExecuTorch**, with no internet and no cloud AI.

---

## 1. The problem

| | |
|---|---|
| **Learning barrier** | Students in low-connectivity regions cannot get study support when they need it. Without internet, online tutors, search and cloud AI are out of reach. |
| **Learner group** | School students (for example Classes 8 to 10) in rural and underserved areas, studying from a textbook. |
| **How this PoC helps** | A low-cost device running a small model through ExecuTorch converts any passage into a short summary and a self-test quiz with answers, entirely offline. |

## 2. What it does

Input: one educational passage of about 150 to 300 words (`passage.txt`).

Output, printed in the terminal:

1. **Summary**: 3 short, simple sentences.
2. **Quiz**: 3 multiple-choice questions (options A to D), each about one single fact.
3. **Answer key**: the correct letter for each question, plus a one-sentence explanation taken word for word from the passage.
4. **Metrics**: model, model size, load time, generation time, tokens per second and peak memory.

## 3. How it works

```
passage.txt
   |
   v
prompt 1: summary ------------------------------+
prompt 2: 3 multiple-choice questions ----------+--> Qwen3-0.6B (.pte) on ExecuTorch (XNNPACK, CPU)
prompt 3 (x3): pick the correct option ---------+
   (the model is shown the question, the options and the
    passage sentence that best matches the question)
   |
   v
terminal output + metrics
```

Design notes:

- **Separate prompts per task.** A 0.6B model follows short, single-purpose prompts much better than one combined prompt.
- **Thinking mode is off** (`enable_thinking=False`) to save tokens and time.
- **Answer key is grounded in the text.** The model picks a letter after being shown the passage sentence that matches the question, and the explanation printed is that sentence itself. This fixed an early bug where the key marked a wrong option.
- **Warnings are silenced inside the script.** Set `QUIET = False` at the top of `run_qa.py` to see every log line.

## 4. Model and runtime

| Item | Value |
|---|---|
| Model | Qwen3-0.6B (Qwen/Qwen3-0.6B) |
| Runtime | ExecuTorch 1.5.1 (XNNPACK backend, CPU) |
| Export tool | optimum-executorch 1.1.0 |
| Quantization | 8da4w for linear layers (4-bit weights, 8-bit dynamic activations), 8w for embeddings |
| Exported file | `model.pte`, about 492 MB (original model about 1.5 GB) |
| Environment | Ubuntu on WSL2, Python 3.12, torch 2.15.0 (CPU) |

## 5. Setup (from scratch)

Run inside Ubuntu (WSL2 on Windows is fine).

```bash
# get the code
git clone https://github.com/JayaHaris2007/Offline-Smart-Learning-Assistant--ExecuTorch-PoC.git
cd Offline-Smart-Learning-Assistant--ExecuTorch-PoC

# tools
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git build-essential

# python environment with ExecuTorch
python3 -m venv ~/et-venv && source ~/et-venv/bin/activate
pip install --upgrade pip && pip install executorch

# download the model
mkdir -p ~/work && cd ~/work
pip install transformers huggingface_hub accelerate
hf download Qwen/Qwen3-0.6B --local-dir ~/work/Qwen3-0.6B

# export to a .pte file (takes about 10 to 20 minutes)
pip install optimum-executorch
optimum-cli export executorch \
  --model ~/work/Qwen3-0.6B --task text-generation --recipe xnnpack \
  --use_custom_sdpa --use_custom_kv_cache \
  --qlinear 8da4w --qembedding 8w \
  --output_dir ~/work/qwen3_pte
```

## 6. Run

1. Copy `application/run_qa.py` and `application/passage.txt` from the cloned repo into `~/work`.
2. Put the passage you want to study in `~/work/passage.txt`.
3. Run:

```bash
source ~/et-venv/bin/activate
cd ~/work
python run_qa.py
```

Files expected in `~/work`:

```
run_qa.py        the assistant script
passage.txt      the input passage
Qwen3-0.6B/      tokenizer files from the original model
qwen3_pte/       model.pte
```

## 7. Results

### Laptop (measured)

Windows laptop, AMD Ryzen 7, 16 GB RAM, run under WSL2.

| Metric | Value |
|---|---|
| Model file size | 492 MB |
| Load + warm-up (first call) | about 3.3 s |
| Total generation time (summary + quiz + answer key) | about 14 s |
| Generation speed (warm) | about 16 to 19 tokens/s |
| Peak RAM, whole Python process | about 2.8 GB |
| Internet used | None |

Memory breakdown from `measure_ram.py`:

| Stage | Peak RAM |
|---|---|
| After importing torch | 221 MB |
| After all imports | 499 MB |
| After loading the model | 654 MB |
| After the first generation | about 2.7 GB |

Most of the growth happens inside the first generation, not in Python or the model load.

### Raspberry Pi (to be measured at the event)

| Metric | Value |
|---|---|
| Board and RAM | |
| OS | |
| Total generation time | |
| Generation speed | |
| Peak RAM | |
| Internet used | |

## 8. Sample output

Summary produced for a passage about photosynthesis:

```
- Photosynthesis is when plants make their own food using sunlight.
- Plants use sunlight, water, and carbon dioxide to make glucose and oxygen.
- Photosynthesis helps grow plants and make oxygen for breathing.
```

> Paste the final quiz and answer key from your terminal here before submitting.

## 9. Impact example

The passage says photosynthesis is how green plants make their own food. The assistant turns this into the summary line "Photosynthesis is when plants make their own food using sunlight", then asks "What is the main purpose of photosynthesis?" with the answer "A) To produce food". A student studying alone reads the short summary, tests themselves, and sees right away whether they understood. This gives them the feedback loop a tutor would, with no connectivity.

## 10. Limitations

- A 0.6B model can make mistakes. It is a **study aid** and the passage sentence printed with each answer lets a student or teacher check it quickly.
- Quiz options can overlap. A rule in the prompt asks for clearly wrong distractors, but small models do not always follow it.
- English only. Passages longer than the model's context are not supported.
- The Python version of the script uses a lot of memory (about 2.8 GB peak on the laptop). On a small board, re-export with `--max_seq_len 1024` to reduce the context size, or use the ExecuTorch C++ runner. Both are untested here.

## 11. Event day checklist

- [ ] USB stick has: `model.pte`, `Qwen3-0.6B/` tokenizer files, `run_qa.py`, `passage.txt`, this README
- [ ] Confirm the board model, RAM, OS and whether it has internet
- [ ] Turn Wi-Fi off (or show a failed `ping`) to prove offline operation
- [ ] Replace `passage.txt` with the organisers' passage
- [ ] Run `python run_qa.py` and record the Raspberry Pi results in section 7
- [ ] Prepare the one-sentence impact story from section 9

## Credits

- Qwen3 by the Qwen Team, Alibaba Cloud (Apache-2.0)
- ExecuTorch by PyTorch / Meta
- optimum-executorch by Hugging Face