---
# Copy this folder to posts/<your-slug>/ and edit. Folders starting with "_" are never published.
# The slug becomes the URL: /posts/<your-slug>/
title: "What you measured, stated plainly"
description: >-
  One or two sentences that say what you tested and what you found. This text
  appears on the home page, in search results and in the feed.
date: 2026-01-31          # first publication (YYYY-MM-DD)
# updated: 2026-02-15     # add when you revise the post, with a revision note at the end
tags: [Performance]       # the first tag is shown in the breadcrumb; each tag gets a topic page
draft: true               # remove when ready; drafts only build with --drafts
metrics:                  # up to three headline numbers, shown on the home page and post header
  - value: "2.5×"
    label: "what this number is, in a few words"
key_findings:             # 3–5 bullets; Markdown works here
  - "**The main result, in bold.** One or two sentences of support, with the numbers."
  - "**A second result.** Include where it stops applying."
resources:                # linked in the "Data, code & appendices" box at the end
  - title: Raw results (CSV)
    url: data/results.csv
    note: What's in the file, in one line.
---

One or two paragraphs: the problem, why it matters, and what the reader will learn. Lead with the question, not the background.

## The question

State what you set out to measure, as a question someone could answer with data.

## Setup

| | |
|---|---|
| **Hardware** | GPU/CPU model, memory, interconnect |
| **Software** | Framework and library versions, driver, OS |
| **Workload** | Models, data, batch sizes |
| **Protocol** | Warm-up, number of runs, what was timed, how variance is reported |

## Results

Tables and figures, built from the files in `data/`. Put the most important table first.

| Configuration | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| Example | 100 ms | 40 ms | 2.5× |

![Describe the chart for screen readers](images/example.png "**Figure 1.** The caption goes in the image title. Say what the reader should notice.")

## Why it happens

Explain the mechanism. Mark anything you believe but didn't measure as a hypothesis.

## Recommendations

What a reader should do with this, and under what conditions it applies.

> [!NOTE]
> Callouts use GitHub's syntax, so they look the same on GitHub and on the site:
> `[!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]` and `[!CAUTION]`.

## Limitations

What wasn't measured, how small the sample is, which results are extrapolated, and what could change the conclusion.

## Reproduce it

```bash
# The commands that produce data/ from scratch
```

---

*Revision, Month YYYY:* when a published post changes, say what changed and why.
