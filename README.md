# Weather PCA — Built From Scratch

A little desktop app I put together to actually *understand* PCA instead of just calling `sklearn.decomposition.PCA()` and hoping for the best. No numpy linear algebra shortcuts either — the covariance matrix, eigenvalues, and eigenvectors are all computed by hand using power iteration.

You load a weather dataset (temperature, humidity, wind speed, pressure), pick a year range, and it shows you the raw features, a scree plot, and the data projected onto its first two principal components — plus a plain-English readout of what each component actually represents.

## Why I built this

I kept using PCA as a black box and it bugged me. The best way I know to actually learn something is to implement it myself and see where it breaks. So this does PCA the "long way":

1. Standardize the data (mean 0, std 1 per feature)
2. Build the covariance matrix
3. Find eigenvalues/eigenvectors one at a time using **power iteration**, deflating the matrix after each one
4. Project the original data onto the top components

No `numpy.linalg.eig`, no `sklearn`. Just loops and basic arithmetic. It's slower and less numerically robust than the real thing, but that's kind of the point — you can read every line and know exactly what's happening.

## What it looks like

It's a Tkinter GUI with three tabs:

- **Original Features** — the raw time series (temp, humidity, wind, pressure) over your chosen years
- **Scree Plot** — how much variance each principal component explains
- **2D PCA Projection** — the data compressed down to PC1 and PC2 over time

Underneath, a text panel spells out the loadings for each component and what they suggest about the data.

## Getting it running

You'll need Python 3 and a couple of packages:

```bash
pip install pandas matplotlib
```

`tkinter` ships with most Python installs. If it's missing on Linux:

```bash
sudo apt install python3-tk
```

Then just:

```bash
python main.py
```

Click **Load CSV**, point it at `sample_weather_data.csv` (included in this repo), set a start/end year, and hit **Run PCA & Visualise**.

## About the sample data

`sample_weather_data.csv` has daily weather readings from **2009 to 2013** — mean temperature, humidity, wind speed, and mean pressure. It'll work with any CSV that has the same five columns (`date`, `meantemp`, `humidity`, `wind_speed`, `meanpressure`), so feel free to swap in your own.

## Heads up / known limitations

Being upfront about the rough edges since this was a learning project, not a production tool:

- **It's a GUI app**, so it needs a desktop environment — it won't run on a headless server or in a notebook as-is.
- **Power iteration uses `random.random()` with no fixed seed**, so results can shift slightly between runs (they should converge to the same answer, but convergence speed will vary). Worth fixing if you want reproducible output.
- **Only tested against this specific weather dataset.** The column names are hardcoded, so other CSVs need the same structure.
- I haven't rigorously benchmarked the manual PCA against `sklearn`'s implementation — on my to-do list, since it'd be a good sanity check that the eigen-decomposition is actually correct.

## What's next

Things I'd like to add if I come back to this:

- [ ] Validate manual PCA output against `sklearn.decomposition.PCA` on the same data
- [ ] Seed the random vector in power iteration for reproducibility
- [ ] Let the user pick which columns to run PCA on, instead of hardcoding them
- [ ] Export plots/interpretation as a PDF report

---

Built as a hands-on way to learn the linear algebra behind PCA. If you spot a bug or have ideas, issues and PRs are welcome.
