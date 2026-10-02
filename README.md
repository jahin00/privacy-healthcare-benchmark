# Privacy-Preserving Healthcare AI Benchmark

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A benchmark and interactive app that **measures what privacy costs, and what protection it actually buys**, when
training healthcare models across hospitals. It compares centralized learning, federated averaging (FedAvg),
differential privacy (DP), secure aggregation (SecAgg), and their combination, then attacks every trained
configuration with simulated privacy attacks.

Built on top of a Federated Ensemble Learning (FEL) framework developed for an MS thesis on privacy-preserving
heart-disease prediction.

> **Scope.** This is a research and education tool. The attacks are simulations, SecAgg is simulated (not a
> cryptographic implementation), and communication cost is modeled. It does not certify that any system is private.
> See [Limitations](#limitations).

---

## What it does

| Configuration | Raw data shared? | Individual updates visible to server? | Formal ε |
|---|---|---|---|
| Centralized | Yes | n/a | none |
| FedAvg | No | Yes | none |
| FedAvg + SecAgg | No | No (aggregate only) | none |
| FedAvg + DP (ε sweep) | No | Yes (central DP trusts the server) | yes |
| FedAvg + DP + SecAgg (ε sweep) | No | No | yes |

**Metrics per configuration** (mean ± std over seeds): accuracy, balanced accuracy, F1, ROC-AUC, privacy budget ε,
training time, and communication overhead (MB). Results are combined into **privacy-utility frontiers** with
Pareto-optimal configurations flagged.

**Simulated attacks** (run against every trained model):

- **Membership inference**: loss-threshold attack distinguishing training rows from held-out rows.
- **Model inversion**: attribute inference that recovers a hidden binary feature from the model's likelihood and the
  true label, measured against a majority-class baseline.
- **Gradient leakage**: an honest-but-curious server reconstructs a patient's feature vector from the update it
  receives. For a logistic model a single-sample update reveals the features exactly, which makes it a clear worst case.

## Why it matters

"Privacy-preserving" is easy to claim and hard to quantify. This project puts utility, privacy budget, cost, and
attack resistance in one comparison so the trade-offs are visible instead of assumed. Two things it makes concrete:

1. Privacy mechanisms protect against *different threats*. DP limits what the released model reveals; SecAgg limits
   what the server sees during training. Neither replaces the other.
2. Utility loss from DP depends heavily on the budget and clipping choices, so it should be measured on your data.

## Quick start

```bash
git clone https://github.com/<your-username>/privacy-healthcare-benchmark.git
cd privacy-healthcare-benchmark
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option A: the app (upload your own CSVs)

```bash
streamlit run streamlit_app.py
```

1. Upload one CSV per hospital/site (or a single CSV and split it into K simulated sites).
2. Choose the outcome column.
3. Run. Results appear in the Frontier, Attacks, Cost, and Budget explorer tabs.

Sites must describe the same kind of patients with overlapping column names, and the outcome should be binary
(low-cardinality targets are binarised: lowest level = 0, others = 1).

### Option B: command line

```bash
# fast smoke test on the datasets configured in fel_v19.py
python fel_privacy_benchmark.py --quick

# your own files, one per site
python fel_privacy_benchmark.py --data siteA.csv siteB.csv siteC.csv --generic --target outcome

# one file split into 4 simulated sites, subsampled for speed
python fel_privacy_benchmark.py --data big.csv --simulate_clients 4 --max_rows 5000 --generic

# full sweep
python fel_privacy_benchmark.py --seeds 3 --rounds 30 --eps 1 5 10 50 200 1000
```

Key options: `--dp_clip` (update clipping norm; try 1-2 if DP utility collapses), `--sensitive_feature`
(hidden attribute for model inversion, `auto` by default), `--module` (name of the FEL module, default `fel_v19`).

### Outputs

Written to `--out` (default `benchmark_outputs/`):

- `results_summary.csv`: mean/std per configuration, with Pareto flags
- `results_raw.csv`: every (configuration, seed) run
- `benchmark_report.json`, `dataset_info.json`
- `privacy_utility_frontier.png`

## Results

> Add your own results here after running the benchmark, for example:
>
> ![Privacy-utility frontier](docs/privacy_utility_frontier.png)
>
> | Configuration | ε | F1 | Membership adv. | Gradient leakage (cos) | Comm. (MB) |
> |---|---|---|---|---|---|
> | _fill in from `results_summary.csv`_ | | | | | |
>
> State the datasets, number of sites, rounds, seeds, and DP clip norm used.

## Repository layout

```
.
├── fel_v19.py                  # FEL framework (FedLinear + FedDNN + federated stacking, DP accountant, SecAgg sim)
├── fel_privacy_benchmark.py    # benchmark harness, attacks, frontier computation, plots
├── streamlit_app.py            # upload-and-run dashboard
├── requirements.txt
└── docs/                       # figures for the README
```

## How it works

- **Training.** Each site trains a linear model, a small neural network, and a meta-learner locally; the server
  aggregates weighted updates (FedAvg). All federated runs use a fixed number of rounds so the DP accountant's ε is exact.
  The best round is selected on pooled validation data.
- **Differential privacy.** Client updates are clipped, aggregated, and perturbed with Gaussian noise calibrated by an
  RDP accountant for client-level (replace-one-client) DP.
- **Secure aggregation.** Simulated additive secret sharing. Masks cancel exactly, so utility matches the non-SecAgg
  run; the difference shows up in communication cost and in what a curious server can observe.
- **Communication model.** float64 parameters uploaded and downloaded each round; SecAgg uploads one share per peer;
  centralized counts the one-time raw-data upload.
- **Frontier.** A configuration is Pareto-optimal if no other has lower-or-equal ε and higher F1.

## Limitations

- SecAgg is a simulation, not a cryptographic protocol; communication numbers are modeled, not measured on a network.
- The DP accountant is a simple RDP implementation, not a vetted library. Do not treat reported ε as an audited guarantee.
- Central DP assumes a trusted server, so DP-only configurations do not stop a curious server from seeing individual
  updates. The gradient-leakage results reflect this by design.
- The attacks are basic. Membership inference on small tabular models often shows little leakage; that is a real
  result, not proof of safety. Stronger attacks (shadow models, iterative gradient inversion) are not included.
- Selecting the best checkpoint on validation data is a small leak against the formal DP guarantee.
- Text columns are encoded by sorted label, which can be inconsistent across sites with different label sets.
- Binary outcomes only. Not a medical device and not for clinical use.

## Data and ethics

Do not commit patient data. Use only datasets you are licensed to use, check each source's terms, and de-identify
anything you upload. The `.gitignore` excludes common data and output folders.

## Roadmap

- FedProx and SCAFFOLD configurations
- Local-DP variant (noise added on the client)
- Stronger attacks (shadow-model membership inference, iterative gradient inversion for the neural network)
- Non-IID site splits and per-site fairness reporting

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg), 2017
- Abadi et al., *Deep Learning with Differential Privacy*, 2016
- Mironov, *Rényi Differential Privacy*, 2017
- Bonawitz et al., *Practical Secure Aggregation for Privacy-Preserving Machine Learning*, 2017
- Shokri et al., *Membership Inference Attacks Against Machine Learning Models*, 2017
- Yeom et al., *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting*, 2018
- Fredrikson et al., *Model Inversion Attacks that Exploit Confidence Information*, 2015
- Zhu et al., *Deep Leakage from Gradients*, 2019

## License

MIT. See [LICENSE](LICENSE).
