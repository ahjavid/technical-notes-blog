# Changelog

Notable changes to the site and its posts. Each post also carries its own revision note when its content changes.

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
