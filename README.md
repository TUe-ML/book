# Data Mining and Machine Learning
Introductory Course to Data Mining and Machine Learning

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.tue.nl%2F20214358%2Fdmml/HEAD)

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](<https://20214358.pages.tue.nl/dmml/intro.html>)




## getting started

- create virtual environment
`python3 -m venv .venv`

- activate virtual environmnet
`source .venv/bin/activate`

- install dependencies
`pip install -r requirements.txt`


-- build everything from scratch

`jupyter-book build --all ./`

you should find the website in `_build/html`


- build only the files that have some changes

`jupyter-book build ./`


- copy changes to github pages 

`ghp-import -n -p -f _build/html`
