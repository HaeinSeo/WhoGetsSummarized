# AEP selection labels - guideline v1.0

For each automatic candidate of a product (budget shown in `budget`, ~100-120 words total):
- `v_evidence_valid` = 1 if the candidate is a real experience reported in the reviews with its
  condition (check the quote / reviews), else 0.
- `y_include` = 1 if, given ALL candidates of this product and the budget, it belongs in a
  summary that helps a purchase decision. Prefer concrete, decision-relevant experiences over
  repeated generic praise; do not force a negative item when none is supported.
Label from the inputs only, never from a model summary.
