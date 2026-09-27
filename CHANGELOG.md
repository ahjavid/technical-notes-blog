# Changelog

Notable changes to the site and its posts. Each post also carries its own revision note when its content changes.

## 2026-09-27: Corrected model sizes, software versions and timing details

Several details in the original study notes were wrong. These values are now taken from the recorded results, the model code and package compatibility:

- **Multi-GPU training.** The benchmarked models have **54,403 and 2,021,379 parameters**, not 258K and 6.9M. This follows from the layer definitions (50 inputs, 3 outputs), and the TensorFlow study's results record the same counts for the same architectures. The environment is **Python 3.12.4, TensorFlow 2.19.0, NumPy 2.1.3, CUDA 12.5.1 and cuDNN 9**. Removed: TensorFlow 2.13 and pandas 2.0.3 (neither supports Python 3.12), NCCL 2.18.5 (not used by `HierarchicalCopyAllReduce`, and not the version TensorFlow 2.19 ships with), and two conflicting driver versions. The appendix no longer includes the communication breakdown, the memory-bandwidth figures, the confidence percentages or the "120+ hours" claim, none of which the measurements support.
- **TensorFlow retracing.** Removed the link to `ahjavid/aistock-analysis`, which doesn't exist. The benchmark scripts are in a private codebase.
- **Vision quantization.** Removed library versions that weren't recorded with the results (PyTorch 2.1 has no Python 3.12 build) and the claim of 1,000 timing iterations. The recorded run times show each configuration was timed over roughly two dozen passes, so differences of a few percent are treated as noise.

## 2026-09-27: Reorganization and accuracy pass

### Site

- Replaced the hand-written HTML pages with a Markdown-first site generator (`site/build.py`). Each post is now written once in `posts/<slug>/index.md`, and the home page, topic pages, Atom feed and sitemap are generated from it.
- New theme: light and dark modes, a sticky table of contents, syntax highlighting with copy buttons, captioned figures, callouts, and responsive tables.
- Added topic pages, an About page, an Atom feed (`/feed.xml`), a sitemap and a 404 page.
- The build now fails on broken internal links and anchors. CI checks pull requests and deploys only from `main` (previously pull requests also triggered a deploy).
- Removed unused files: `assets/js/main.js` and `assets/js/config.js` (never loaded by any page) and `posts/multi-gpu-training-analysis/index-original.html`.

### Posts

Every post was checked against its committed data and rewritten around it, with a Limitations section added:

- **APEE.** The four "evaluation modes" were four reruns scored by the same basic ensemble. The protocols' own scores (9.04, 5.05, 3.64) are now reported, and the per-scenario tables match the JSON results. The long-form study was merged into the post, and the package README now covers only the package.
- **Vision quantization.** Removed models that weren't part of the study, unmeasured accuracy figures, an incorrect GPU (RTX 4090), and ROI estimates. Now reports the INT8 slowdown (0.13–0.46×) and the INT4 fallback. The two supplementary write-ups were merged into the post.
- **TensorFlow retracing.** Figures now come from the committed results, which match the charts (3.85×, 2.54×, 1.88× and a 0.23× regression). The 72.6% and −45% memory claims were removed. The charts are shown in the post, and the data moved to `data/`.
- **Multi-GPU training.** Removed a model-size table, a cost table and a communication breakdown that the measurements don't support. Confidence intervals were added, and the benchmark code listings were fixed (TensorFlow API errors and missing imports).

### APEE package

- Fixed five coordinator tests that were stale after the December 11 coordination changes (new default context limits, feedback and meta-review phases). Added tests for the phases-off paths. All tests pass (4 are skipped without a running Ollama server).
- Added a CI workflow that runs the package's tests.

## 2025-12-11

- APEE: reviewer agent switched from `granite4:3b` to `phi4-mini:3.8b`, and new evaluation runs recorded for all four modes.
- Site redesign with a shared stylesheet.

## 2025-12-09

- New post: APEE, the Adaptive Poly-Agentic Evaluation Ecosystem, including its Python package, tests and results.

## 2025-06-20

- New post: vision model quantization study (16 models, 4 precisions).

## 2025-06-17

- Blog launched on GitHub Pages.
- New posts: multi-GPU training performance analysis, and TensorFlow retracing optimization.
