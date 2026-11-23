# MosaCD

MosaCD combines conditional-independence (CI) skeleton discovery with LLM edge
votes, cross-order consistency and CI/cycle seed filtering, then interleaved
propagation and least-conflict orientation. Vote-based completion orients any
remaining undirected edges by default.

## Install

Use **Python 3.11** (tested) and Git:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The pgmpy dependency is pinned to the source revision used by this code; its
transitive dependencies include PyTorch. No GPU is needed.

## Run the example

From the repository root:

```bash
export OPENAI_API_KEY="your-api-key"
mosacd --data examples/asia.csv --metadata examples/asia.json --output result.json
```

`python -m mosacd` accepts the same options. The example contains 1,000 rows
from the existing Asia Bayesian-network sample file, with descriptive column
names and binary coding (`0=no`, `1=yes`); it is not clinical data.

To inspect the skeleton and expected request count **without an API key or charges**:

```bash
mosacd --data examples/asia.csv --metadata examples/asia.json --dry-run --output result_preview.json
```

## Run your own data

Provide a CSV with samples in rows and unique variable names in columns
(no index column, missing values, infinite values, or constant columns).
Prepare a metadata JSON whose variable keys exactly match the CSV columns:

```json
{
  "description": "Describe the domain and what each observation represents.",
  "variables": {
    "variable_1": "Meaning, coding, and units.",
    "variable_2": "Meaning, coding, and units."
  }
}
```

```bash
mosacd --data observations.csv --metadata metadata.json --ci-test pearsonr --output result.json
```

Use `chi_square` for categorical data (default), or `pearsonr` for continuous
numeric data with a linear partial-correlation CI test. Select and preprocess
your data explicitly; the tool does not normalize, impute, or encode it.

Main options (`mosacd --help` lists all):

| Option | Default |
|---|---|
| `--alpha` | `0.01` |
| `--max-cond-vars` | `3` |
| `--model` | `gpt-4o-mini` |
| `--temperature` | `0.6` |
| `--n-votes` | `5` per edge **per prompt order** |
| `--seed` | `0` |

There are **2 × n_votes × skeleton-edge-count** logical API requests (plus any
SDK retries), sent sequentially. Large graphs can be slow. Your provider bills
these requests; the output records token usage, not a dollar-cost estimate.
Seeds do not guarantee identical LLM responses or graph tie-breaking across runs.

## Output and privacy

`result.json` contains the configuration, timings, skeleton edges, token usage,
and `graph` with `nodes`, `directed_edges` (`[source, target]`), and
`undirected_edges`. Add `--no-completion` to retain unresolved orientations.
`directed_part_is_acyclic` reports whether the final directed part is acyclic;
the legacy orientation/completion procedure does **not** guarantee a DAG.

Only domain/variable descriptions and local CI summaries are sent to the LLM,
not the raw observation table. Do not put confidential information in metadata.
Keys are read from environment variables and are never written to the output.
For another OpenAI-compatible service, set `--base-url` and `--api-key-env`.
Models containing `/` use OpenRouter and `OPENROUTER_API_KEY` unless overridden.

Inferred edges are hypotheses, not experimentally validated causal effects.
