# Format Drift Chain

Start at `r/01.md`. Follow each `next_file` pointer until it is null. Count files
where `risk` is true, then return the release decision as strict JSON only.
