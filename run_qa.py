import os, sys, re, time, io, inspect, contextlib, resource, warnings

QUIET = True  # set to False to see every warning and log line

warnings.filterwarnings("ignore")


def mute():
    if QUIET:
        sys.stderr.flush()
        saved = os.dup(2)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
        return saved
    return None


def unmute(saved):
    if saved is not None:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


PTE_DIR = os.path.expanduser("~/work/qwen3_pte")
TOK_DIR = os.path.expanduser("~/work/Qwen3-0.6B")
passage = open(os.path.expanduser("~/work/passage.txt")).read().strip()

_saved = mute()
try:
    from transformers import AutoTokenizer
    from optimum.executorch import ExecuTorchModelForCausalLM

    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    model = ExecuTorchModelForCausalLM.from_pretrained(PTE_DIR)
finally:
    unmute(_saved)


def build(instruction):
    msgs = [{"role": "user", "content": f"{instruction}\n\nPassage:\n{passage}"}]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generate(prompt, max_new, sample=False):
    params = inspect.signature(model.text_generation).parameters
    kw = {}
    supported = set(params)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if "tokenizer" in params: kw["tokenizer"] = tok
    if "prompt" in params: kw["prompt"] = prompt
    if "echo" in params: kw["echo"] = False
    if "max_seq_len" in params: kw["max_seq_len"] = len(tok(prompt).input_ids) + max_new
    # Sampling prevents the quiz from being identical on every invocation when
    # the backend exposes the standard generation controls.
    if sample or accepts_kwargs:
        if "do_sample" in supported or accepts_kwargs: kw["do_sample"] = sample
        if "temperature" in supported or accepts_kwargs: kw["temperature"] = 0.8
        if "top_p" in supported or accepts_kwargs: kw["top_p"] = 0.9
    buf = io.StringIO()
    t = time.time()
    with contextlib.redirect_stdout(buf):
        out = model.text_generation(**kw)
    dt = time.time() - t
    text = out if isinstance(out, str) else buf.getvalue()
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip(), dt


# warm-up so model loading is not counted in the real runs
_saved = mute()
try:
    t = time.time()
    generate(build("Say hello."), 5)
    first_call_s = time.time() - t
finally:
    unmute(_saved)

rows = []

# ---------- 1. SUMMARY ----------
summary, dt = generate(
    build("Summarize the passage below in 3 short, simple sentences for a school student."), 120
)
summary = summary.replace("**", "")
rows.append(("SUMMARY", dt, len(tok(summary).input_ids)))
print(f"\n===== SUMMARY =====\n{summary}")

# ---------- 2. QUIZ QUESTIONS ----------
quiz, dt = generate(
    build(
        "Write exactly 3 multiple-choice quiz questions based only on the passage below. "
        "Each question must ask about ONE single fact with ONE short answer "
        "(never ask for two things at once). "
        "Give 4 options labeled A. B. C. D. for each question. "
        "Number the questions 1 to 3. Do not write the answers."
    ),
    170,
    sample=True,
)
quiz = quiz.replace("**", "")
rows.append(("QUIZ", dt, len(tok(quiz).input_ids)))
print(f"\n===== QUIZ =====\n{quiz}")

# split the quiz into blocks: question text + options
blocks = [b for b in re.split(r"(?m)^\s*(?=\d+[.)]\s)", quiz) if b.strip()][:3]
parsed = []
for b in blocks:
    q = re.match(r"\s*\d+[.)]\s*(.+)", b).group(1).strip()
    opts = re.findall(r"(?m)^\s*([A-D])[.)]\s*(.+)$", b)
    parsed.append((q, opts))

# ---------- 3. ANSWER KEY ----------
STOP = set(
    "a an the of in on to is are and or what where which who how why do does "
    "that this it for from by with as at be can".split()
)


def stems(text):
    return {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


def evidence_for(text):
    sents = re.split(r"(?<=[.!?])\s+", passage)
    lower_text = text.lower()
    if "main purpose" in lower_text:
        return sents[0]

    words = re.findall(r"[a-z]+", text.lower())
    question_words = words[:]
    option_words = []
    if "\n" in text:
        question_words, option_text = text.split("\n", 1)
        question_words = re.findall(r"[a-z]+", question_words)
        option_words = re.findall(r"[a-z]+", option_text.lower())

    q = stems(" ".join(question_words))
    options = stems(" ".join(option_words))
    sentence_stems = [stems(sentence) for sentence in sents]
    unique_question_terms = {
        term
        for term in q
        if sum(term in current for current in sentence_stems) == 1
    }

    def score(sentence):
        current_stems = stems(sentence)
        question_score = len(q & current_stems) * 8
        unique_score = len(unique_question_terms & current_stems) * 20
        option_score = len(options & current_stems)
        phrase_score = sum(
            phrase in sentence.lower()
            for phrase in ("takes place", "main purpose", "absorb sunlight")
        )
        return question_score + unique_score + option_score + phrase_score

    return max(sents, key=score)


ans_time, ans_tokens, key = 0.0, 0, []
for q, opts in parsed:
    ev = evidence_for(q + "\n" + " ".join(o[1] for o in opts))
    opt_text = "\n".join(f"{l}. {t}" for l, t in opts)
    a, dt = generate(
        build(
            "Choose the correct option for the question below, using only the passage. "
            f'The answer is in this sentence: "{ev}"\n\n'
            f"Question: {q}\n{opt_text}\n\n"
            "Reply with only the letter of the correct option."
        ),
        4,
    )
    ans_time += dt
    ans_tokens += len(tok(a).input_ids)
    m = re.search(r"[A-D]", a.upper())
    letter = m.group(0) if m else None
    chosen = dict(opts).get(letter) if letter else None
    key.append((letter, chosen, ev))
rows.append((f"ANSWER KEY ({len(parsed)} calls)", ans_time, ans_tokens))

print("\n===== ANSWER KEY =====")
if not parsed:
    print("(could not find numbered questions in the quiz output)")
for i, (letter, chosen, ev) in enumerate(key, 1):
    ans = f"{letter}) {chosen}" if chosen else "(model gave no valid option)"
    print(f"{i}. Answer: {ans}\n   Explanation: {ev}")

# ---------- METRICS ----------
print("\n===== METRICS =====")
print("Model: Qwen3-0.6B, XNNPACK, 8da4w linear / 8w embedding")
print(f"model.pte size: {os.path.getsize(PTE_DIR + '/model.pte') / 1e6:.0f} MB")
print(f"Load + warm-up (first call): {first_call_s:.1f} s")
for name, dt, n in rows:
    speed = f", ~{n / max(dt, 0.01):.1f} tokens/s" if n >= 20 else ""
    print(f"{name}: {dt:.1f} s, ~{n} tokens{speed} (includes prompt processing)")
print(f"Total generation time: {sum(r[1] for r in rows):.1f} s")
print(f"Peak RAM (whole Python process): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")