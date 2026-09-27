---
title: About
description: Who writes these notes, what they cover, and the standards every post follows.
---

I'm A.H. Javid. I work where machine learning meets system performance: making ML training and inference faster and cheaper through careful measurement rather than rules of thumb. This site is where I publish those measurements, along with the data behind them.

## What I write about

- **Performance engineering.** Finding the real bottleneck in training and inference, and benchmarking it properly. See [multi-GPU training](../posts/multi-gpu-training-analysis/) and [TensorFlow retracing](../posts/tensorflow-retracing-optimization/).
- **Model optimization.** Quantization and other ways to make inference cheaper, and what they cost you. See [vision transformer quantization](../posts/vision-model-quantization-study/).
- **Evaluating LLM systems.** Multi-agent collaboration, and how far LLM-as-a-judge scores can be trusted. See [APEE](../posts/apee-evaluation-ecosystem/).

## How the posts are written

Every post follows the same rules:

1. **Numbers come from committed data.** The raw results sit in the post's folder, and the tables are built from them.
2. **The setup comes first.** Hardware, software versions, models and the measurement protocol are stated before any result.
3. **Limitations are part of the post.** Each one says what wasn't measured and where the conclusions stop applying.
4. **Results and advice stay separate.** Recommendations are labelled as such, and anything extrapolated or untested is marked.
5. **Corrections are visible.** When a published post changes, a revision note at the end says what changed and why.

## About this site

The pages are generated from Markdown by a small Python script and hosted on GitHub Pages, without analytics or tracking. The [repository](https://github.com/ahjavid/technical-notes-blog) holds the source of every page along with each post's data and code.

## Get in touch

Questions, corrections and ideas for follow-up experiments are all welcome. [Open an issue](https://github.com/ahjavid/technical-notes-blog/issues) on GitHub.
