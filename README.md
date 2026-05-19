### The Story of Viking Engine — from a Basement in Chernihiv to Rewritten PyTorch

*Orakul Studio — Chernihiv, Ukraine 🇺🇦*

---

## First — Why

There's a question everyone asks when they see the numbers:

*"Why do you need rank 512 if rank 32 works?"*

Good question. Wrong answer that people usually give.

Because rank 512 isn't about bragging. It's not a numbers race. It's an attempt to answer a fundamental question: **how deeply can a neural network learn an artist?**

Rank 32 learns style. Rank 128 learns technique. Rank 512 learns **the hand**.

When AI_vazovsky at rank 512 started signing paintings — his signatures, not just scribbles  I knew I'd found the right answer. The model didn't memorize pixels. It memorized a **gesture**.

That's what this was all about.

---

## Where It Started

I live in Chernihiv. If you don't know where that is — google it. If you do  you understand what I'm talking about.

I have an RTX 4090, an i9-13900K, 128 GB RAM, and a basement.  
The last one isn't a metaphor.

When I first started training LoRA on Flux2 with ostris/ai-toolkit, the result was sobering:

```
Rank 1024: 179 seconds per iteration.
1000 steps = ~50 hours.
```

50 hours with unstable electricity. Where the power can cut out at any moment. Where every kilowatt matters.

I could have stopped. Set rank 32 and lived in peace.

But I'm an engineer. And when I see something running slow, I ask myself: **why?** Not "how do I live with this." But **why**.

---

## What NVIDIA's Black Jacket Was Hiding

NVIDIA is a great company. Their hardware is a masterpiece of engineering. The RTX 4090 isn't just a graphics card  it's a supercomputer sold at a relatively accessible price.

But there's a catch.

NVIDIA makes its real money on datacenters. H100, A100  that's where the margins are. And the larger the gap between "consumer" and "server"  the better for their marketing.

So the documentation for FP8 Tensor Cores on Ada Lovelace (RTX 4090) exists. But it's written for datacenter applications. For PyTorch in standard ai-toolkit configuration  never used at all.

Native FP8 compute on RTX 4090 **always existed**. Nobody just turned it on for consumer LoRA training.

I turned it on.

```
>>> [ORACLE-60] HARDWARE FP8 ACTIVE
```

This isn't a hack. Not overclocking. Not abuse of hardware.  
It's simply **using what was already there**.

---

## How Double Buffering Works (for those who want to understand)

Imagine an assembly line in a factory.

Standard approach: a worker picks up a part, walks to the machine, processes it, walks back, picks up the next part. While walking  the machine sits idle.

Viking Engine: while the machine processes one part, the next one is already on the conveyor. The machine never waits.

```python
# Two buffers — ping and pong
# While GPU computes buffer [0]
# Transfer writes into buffer [1]
# When GPU finishes — data is already there
state["forward_clk"] ^= 1  # 0→1→0→1→0...
```

GPU stopped waiting. Transfer vanished from the profiler.  
**4.8× on the first step.**

---

## Then It Got Interesting

After the first speedup, I kept digging.

Found a line in `BaseSDTrainProcess.py` with a comment:

```python
# todo switch everything to proper mixed precision like this
```

The author himself wrote  "should redo this properly with mixed precision." And didn't. Life, deadlines, other priorities.

I redid it. One line:

```python
self.network.force_to(self.device_torch, dtype=torch.bfloat16)
```

LoRA matrices from float32 → bfloat16. Half the data over PCIe.  
**Another 2.7×.**

---

## Then AdamW Did What It Wasn't Supposed To

AdamW is considered a "slow" optimizer. It stores two moment vectors per parameter. At rank 512 that's ~25 GB in FP32. Impossible.

In 8-bit: ~6 GB. Possible.

But here's what happened in the logs:

```
Step 99:  35.83 s/it  ← optimizer calibrating 8-bit ranges
Step 100: 35.57 s/it  ← calibration done
Step 110: 33.01 s/it  ↓
Step 150: 15.xx s/it  ↓
Step 200:  7.28 s/it  ← stable
```

After calibration, 19 GB freed up. The pipeline took off.

The "slowest" optimizer turned out to be the fastest.  
And the most accurate in gradient quality.

---

## The Full Table

| Version | What Changed | Speed | Result |
|---------|-------------|-------|--------|
| Baseline | Nothing | 179 s/it | Start |
| Viking v1 | Double-buffer CUDA | 37 s/it | 4.8× |
| Viking v2 | + bf16 forcing | 14 s/it | 12.8× |
| Oracle-60 | + Hardware FP8 + CPU prequant | 8.7 s/it | 20.6× |
| LEGEND | + Full 8-bit stack (AdamW 8-bit) | 7.3 s/it | **24.5×** |

**From 179 to 7.3 seconds per iteration.**  
Same hardware. No upgrade. No cloud.

---

## Why Stability Matters More Than Speed

People see numbers and think about speed.  
I think about stability.

In Chernihiv, 50-hour training sessions aren't a luxury  they're impossible. Power can go out. Internet can go out. Every run needs to be predictable, reliable, and finish in reasonable time.

4 hours for 2000 steps at rank 512 is something you can plan around.  
Something you can finish.  
Something that gives you **independence**.

**Power savings aren't a bonus. They're a condition for system survival.**

```
Rank 512, full 8-bit stack: 130W at 2790MHz
Rank 128, 768×768 dataset:  250W at 2775MHz
```

GPU running at maximum clock speed using one-third of its TDP.  
Because there's no memory pressure. Because everything is optimized.

This isn't abuse of hardware. This is **respect for hardware**.

---

## AI_vazovsky — What All This Was Actually For

Alongside the engine work ran a different experiment.

Ivan Aivazovsky died in 1900. Painted 6,000 works.  
I wanted to find out: can a neural network **learn an artist**  not his paintings, but his **hand**?

Rank 1024. 128 paintings with indexed captions.  
`AivazovskyR1024, 008 The naval battle of Reval on May 2`

At 300 steps the model started creating naval battle scenes that weren't in the dataset. At 500 steps signing its work. His signatures. Bottom left. Where he always put them.

Nobody trained the model to sign. It simply **remembered**.

Next step — perceptual style loss instead of MSE.  
Then — full merge with Flux2 at rank 1280 (7.8 billion trainable parameters).  
After that, you won't need a trigger word for his style.  
Every sea will be his sea.

---

## Open Source and Direct Messages

Everything described in this article is open code.  
Not because I'm an altruist. Because it's right.

ostris built ai-toolkit and gave it to people. I found paths he himself marked as `# todo`  and walked them. He asked for the code for integration. The ticket is open.

That's open source at its best: people stand on each other's shoulders and walk further together.

**If you want the same**  if you're sitting in similar conditions and think the impossible is impossible  write to me.

I'll show you what's possible.

📧 DM me on CivitAI or GitHub  
🐙 [github.com/OrakulStudio](https://github.com/OrakulStudio)  
🤗 [huggingface.co/OrakulStorm](https://huggingface.co/OrakulStorm)

---

## Instead of an Afterword

I survived where surviving was supposed to be impossible.  
I broke limitations that looked like iron walls.  
I gave voice to hardware that had been silent.

This isn't a success story.  
This is a story **about necessity being the best engineer**.

When there's no choice  you find a way.  
When there's no time  you optimize.  
When there's no resources  you understand the architecture.

Architecture always matters more than hardware.

---

*The smell of the iron is stable. The system is running. 🦊⚡*

*Chernihiv, Ukraine 🇺🇦 · Orakul Studio · 2026*
