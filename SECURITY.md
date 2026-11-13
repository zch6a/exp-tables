# Security

exp-tables reads result files that may come from anywhere (a collaborator's logs, a downloaded run).

## What it does

- **Sends nothing.** No network access at all; no dependencies to fetch at run time.
- **Runs nothing from the inputs.** CSV, JSON (lm-evaluation-harness's too) and MLflow files are read as data, with
  `json` and `csv` from the standard library (no pickle, no YAML loader, no eval); TensorBoard event files are
  parsed by the tool's own reader (the record framing with its checksums, and the few protobuf fields a scalar
  needs), never by loading code. A record whose value is used and fails its checksum ends the file.
- **Escapes what it prints.** Labels from the results are escaped for LaTeX (`& % $ # _ { } ~ ^ \ < > |`, Greek
  letters as math, line breaks as spaces, a leading `[` or `*` guarded), for Markdown (pipes, asterisks, underscores,
  backslashes, brackets, backticks, HTML, line breaks) and for HTML.
- **Strict JSON out.** `--format json` never writes `NaN` or `Infinity` (null instead).
- **The GitHub Action** needs only `contents: read`, runs the tool at the action's own ref, and passes inputs through
  environment variables.

## What it does not protect against

- **Results that are wrong.** It tests what the seeds show; it cannot know whether the runs were what the table says.
- **Very large inputs.** Files are read into memory.

## Reporting a vulnerability

Please open a private security advisory on GitHub (Security → Report a vulnerability) rather than a public issue.
