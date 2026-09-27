# Technical Notes

[![Deploy](https://github.com/ahjavid/technical-notes-blog/actions/workflows/deploy.yml/badge.svg)](https://github.com/ahjavid/technical-notes-blog/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Applied research notes on ML systems performance by [A.H. Javid](https://github.com/ahjavid): hands-on studies of GPU training, inference optimization and LLM evaluation, each published with the data and code behind every number.

**Read the site: [ahjavid.github.io/technical-notes-blog](https://ahjavid.github.io/technical-notes-blog/)**

## Posts

| Post | Published | What it found | Source |
|---|---|---|---|
| [APEE: Evaluating Multi-Agent LLM Systems with LLM-as-a-Judge](https://ahjavid.github.io/technical-notes-blog/posts/apee-evaluation-ecosystem/) | Dec 2025 | The judging protocol moved scores for the same agent outputs from 3.6 to 9.0 out of 10, more than anything the agents did. | [folder](posts/apee-evaluation-ecosystem/) |
| [Vision Transformer Quantization: What FP16 and INT8 Actually Buy You](https://ahjavid.github.io/technical-notes-blog/posts/vision-model-quantization-study/) | Jun 2025 | FP16 halves memory but only speeds up heavy models (up to 2.5×). bitsandbytes INT8 is 2–8× slower. | [folder](posts/vision-model-quantization-study/) |
| [Eliminating tf.function Retracing in TensorFlow](https://ahjavid.github.io/technical-notes-blog/posts/tensorflow-retracing-optimization/) | Jun 2025 | Three retracing fixes gave 1.9–3.9× speedups; one graph-mode rewrite made things 4× slower. | [folder](posts/tensorflow-retracing-optimization/) |
| [Multi-GPU Training: When Hardware Topology Matters](https://ahjavid.github.io/technical-notes-blog/posts/multi-gpu-training-analysis/) | Jun 2025 | Two GPUs behind a PCIe host bridge trained small models 10–27% *slower* than one. | [folder](posts/multi-gpu-training-analysis/) |

## How this repository is organized

```text
posts/<slug>/            One folder per post: everything behind it in one place
  index.md               The article (Markdown with YAML front matter)
  images/                Figures used in the article
  data/                  Raw results that every number in the article comes from
  *.md                   Appendices, published as pages next to the post
  ...                    Code (e.g. the APEE Python package), linked to GitHub
posts/_template/         Starting point for a new post (not published)
pages/                   Standalone pages such as /about/
site/                    The site generator
  build.py               Markdown → HTML, feed, sitemap, link checking
  config.yml             Title, author, URL and repository settings
  templates/             Page layouts (Jinja2)
  static/                CSS, JavaScript and favicon
.github/workflows/       Build and deploy to GitHub Pages; APEE unit tests
```

The site is plain HTML generated from the Markdown in `posts/` and `pages/`. Each post is written once, in `index.md`, and everything else is derived from it, including the home page cards, topic pages, the Atom feed and the sitemap.

## Preview the site locally

```bash
pip install -r site/requirements.txt
python site/build.py --serve            # http://localhost:8000/technical-notes-blog/
```

The build fails if any internal link or `#anchor` is broken, so run `python site/build.py` before pushing. CI runs the same check on every pull request and deploys `main` to GitHub Pages.

## Writing a post

Copy `posts/_template/` to `posts/<your-slug>/`, fill in the front matter and write. [CONTRIBUTING.md](CONTRIBUTING.md) covers the front matter fields, the Markdown features (callouts, figures, tables), and the checklist every post goes through.

## License

Content and code are released under the [MIT License](LICENSE).
