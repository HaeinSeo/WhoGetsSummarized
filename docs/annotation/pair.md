# Experience-review relation - guideline v1.0

Read the EXPERIENCE (+ CONDITION) and the REVIEW. Judge only this single experience.

| relation | meaning |
|---|---|
| SUPPORT | the review explicitly reports this experience under the same conditions |
| OPPOSE | the review explicitly reports the opposite experience under comparable conditions |
| IRRELEVANT | the review has no evidence about this experience |
| UNCERTAIN | mixed, ambiguous, condition mismatch, or the experience is badly decomposed |

- "Some users had it break" vs "mine never broke" -> OPPOSE (opposite experience), even though it
  does not logically refute the existence claim.
- Do not judge quantifiers ("most", "all"); that is evaluated at bundle level.
- Same aspect but different condition (e.g. different OS, after 1 day vs after 1 year) -> UNCERTAIN,
  and `condition_match = N`. `condition_match`: Y / N / NA (NA when IRRELEVANT).
- Relevance and ambiguity are different: never use IRRELEVANT for "I can't tell".
