# Writing and Contributing

This guide covers how posts are structured, how to preview the site, and the standards every post has to meet. It applies to my own posts and to guest contributions alike.

## Quick start

```bash
pip install -r site/requirements.txt
cp -r posts/_template posts/my-new-study       # the folder name becomes the URL
python site/build.py --serve --drafts          # http://localhost:8000/technical-notes-blog/
```

Edit `posts/my-new-study/index.md`. The server rebuilds on every save, so reload the browser to see changes. Delete `draft: true` from the front matter when the post is ready.

## What goes in a post folder

```text
posts/my-new-study/
├── index.md       the article
├── images/        figures referenced from the article
├── data/          raw results (CSV/JSON): every number in the article must come from here
├── appendix.md    optional; any other .md file is published as a page next to the post
└── code/ ...      optional; Python files and packages stay on GitHub and aren't copied to the site
```

## Front matter

Every `index.md` starts with YAML front matter:

| Field | Required | What it's for |
|---|---|---|
| `title` | yes | Page title and heading. State the finding or the question plainly |
| `description` | yes | One or two sentences, shown on the home page, in search results and in the feed |
| `date` | yes | First publication date, `YYYY-MM-DD` |
| `tags` | yes | A list of topics. The first is shown in the breadcrumb, and each tag gets a topic page |
| `updated` | no | Date of the last substantive revision. Pair it with a revision note at the end of the post |
| `metrics` | no | Up to three `{value, label}` headline numbers for the home page and post header |
| `key_findings` | no | 3–5 Markdown bullets, shown in a box at the top of the post |
| `resources` | no | `{title, url, note}` links for the "Data, code & appendices" box. Appendix pages in the folder are listed automatically |
| `reading_time` | no | Minutes. Calculated automatically if left out |
| `draft` | no | `true` keeps the post out of the build unless you pass `--drafts` |

## Markdown features

- **Headings.** Start sections at `##`. Every `##` and `###` heading appears in the table of contents, and heading IDs match GitHub's, so `other.md#some-section` links work in both places.
- **Links.** Write them relative to the file, as you would on GitHub. Links to `.md` files become links to the published pages. Links to code, or to anything else that isn't published, are rewritten to point at the file on GitHub.
- **Figures.** An image on its own line becomes a figure, and its title becomes the caption:
  `![Alt text for screen readers](images/chart.png "**Figure 1.** What the reader should notice.")`
- **Tables.** Standard Markdown tables. Right-align numeric columns with `---:`. Wide tables scroll on small screens.
- **Callouts.** Use GitHub's alert syntax: `> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]` or `> [!CAUTION]`.
- **Code.** Fenced blocks with a language (`python`, `bash`, `yaml`, `json`, `text`, …) are syntax-highlighted and get a copy button.
- **Footnotes and task lists** work as they do on GitHub.

## Standards for every post

1. **Every number comes from committed data.** If a table can't be regenerated from `data/`, it doesn't belong in the post. Never estimate a number and present it as a measurement.
2. **The setup comes first:** hardware, software versions, workload and measurement protocol, including how many runs and how variance is reported.
3. **Report what happened, including regressions and failures.** A slower "optimization" is a result.
4. **Keep measurements, extrapolations and opinions apart.** Label extrapolated thresholds and untested suggestions as such.
5. **Include a Limitations section**: sample size, what wasn't measured, and where the results stop applying.
6. **Make corrections visible.** When a published post changes in substance, set `updated` and add a revision note at the end.

## Before you push

```bash
python site/build.py
```

The build fails on any broken internal link or missing `#anchor`, and CI runs the same check on pull requests. Pushes to `main` deploy to GitHub Pages automatically.

If you change the APEE package in `posts/apee-evaluation-ecosystem/`, also run its tests:

```bash
cd posts/apee-evaluation-ecosystem && pip install -e ".[dev]" && pytest tests/
```

## Guest posts and corrections

- **Found a mistake?** [Open an issue](https://github.com/ahjavid/technical-notes-blog/issues) with the post, the claim, and the data that contradicts it. Corrections are always welcome.
- **Want to contribute a study?** Open an issue first with a short abstract, the setup, and what data you'll publish. Then send the post as a pull request that includes its `data/` folder and code.
