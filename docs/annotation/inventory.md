# Input experience inventory - guideline v1.0

Done **before and without** looking at any summary. One file per product (`<bundle>_reviews.csv`).
Fill `<bundle>_inventory_<annotator>.csv`, one row per distinct experience:

- `experience_id`: e01, e02, ... ; `aspect`: snake_case attribute; `proposition`: standalone sentence;
  `condition`: usage condition/time/variant or empty; `polarity`: positive/negative/mixed/neutral.
- `support_anon_ids` / `oppose_anon_ids`: comma-separated R-ids (e.g. `R03,R17`).
- `importance`: 0 vague exclamation / duplicate; 1 concrete but low purchase relevance;
  2 concrete and relevant to use, compatibility, performance, durability, safety (positive or negative).
- `risk_flag`: 1 for possible safety issues (reported, not verified as product facts).
- Keep different conditions and early vs revised versions as separate experiences.
Adjudicator merges both into `<bundle>_inventory_gold.csv`, then run `wgs annotate inventory-gold`.
Frequency is computed from the ids; do not let frequency decide importance.
