# leeway00.github.io

Personal homepage for Younghun Lee, served by GitHub Pages from `main` at the repo root.

Design forked from [Daniel R. Jiang's webpage template](https://danielrjiang.github.io/) — the
attribution in the page footer stays.

## Editing

`index.html` is **generated** — do not edit it directly. The page is assembled from:

```
template.html        page shell (head, meta, footer) + @include lines
sections/            one file per section of the page
  bio.html           name, photo, links, short bio
  news.html          dated one-line updates
  publications.html  papers, with the tag filter
  teaching.html      courses
  projects.html      personal projects
build.py             expands the @include lines -> index.html
```

An `@include` works like `\input` in LaTeX. A line in `template.html` of the form

```html
<!-- @include sections/bio.html -->
```

is replaced by that file's contents, indented to match. Includes may nest.

After editing anything under `sections/` or `template.html`:

```sh
python3 build.py
```

Then commit both the source and the regenerated `index.html` — GitHub Pages serves the
generated file, so a build you forgot to run means a change that never goes live.

To remove a section entirely, delete its file and its `@include` line. To preview locally,
open `index.html` in a browser, or run `python3 -m http.server` and visit `localhost:8000`.

Every section except `bio.html` currently holds a commented-out skeleton; uncomment and fill
in the ones you need.

## Assets to replace

- `img/profile.svg` — placeholder gradient, swap for a real photo
- `files/cv.pdf` — placeholder (empty) PDF, currently linked from the bio
- `img/favicon.svg`, `img/publications/*.svg` — placeholder icons
