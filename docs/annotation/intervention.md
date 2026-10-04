# Controlled pair validation - guideline v1.0

`valid_pair` = 1 only if:
- target_preserved: removed and added reviews contain NO evidence about the target experience,
  its aspect, or its condition, and the swap does not change product variant/time context.
- target_removed: every review supporting the target (including paraphrases) was removed;
  opposing reviews may remain. aspect_removed: nothing about the aspect remains.
Otherwise 0, with a note.

`intervention_target` sheets: `target_in_original` / `target_in_modified` = 1 when the summary
expresses the target experience faithfully (same rule as summary.md), else 0.
