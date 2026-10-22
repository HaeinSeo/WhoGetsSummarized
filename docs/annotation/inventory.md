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

---
## 한국어 상세 규칙 (열별)

한 줄 = 경험 하나. 리뷰를 읽다가 구체적인 경험이 나오면 줄을 만들고, 같은 경험이 또 나오면 번호만 추가한다.

| 열 | 쓰는 값 | 규칙 |
|---|---|---|
| experience_id | `e01`, `e02`, … | 제품마다 e01부터 순서대로. 중복 금지 |
| aspect | 아래 목록 중 하나 (영어 소문자) | 두 주석자가 같은 이름을 써야 비교가 쉬우므로 목록에서 고른다. 맞는 게 없을 때만 새로 만든다 |
| proposition | 영어 한 문장 | 제품에 대한 현재형 문장. "customers said", "some", "most" 같은 말은 쓰지 않는다. 경험 하나만 담는다 |
| condition | 영어 짧은 구, 없으면 빈칸 | 사용 기간·기기·사이즈·상황 등. 조건이 다르면 줄을 나눈다 |
| polarity | `positive` / `negative` / `mixed` / `neutral` | 아래 표 참고 |
| support_anon_ids | `R03,R17` (쉼표, 띄어쓰기 없음) | 이 경험을 직접 말한 리뷰 번호. 최소 1개 |
| oppose_anon_ids | `R22` 또는 빈칸 | 비슷한 조건에서 반대 경험을 말한 리뷰만. 언급이 없는 리뷰는 넣지 않는다 |
| importance | `0` / `1` / `2` | 아래 표 참고 |
| risk_flag | `0` / `1` | 아래 표 참고 |
| note | 자유 메모 (한국어 가능) | 애매한 점. 없으면 빈칸 |

### aspect 목록
`size_fit` 크기·핏, `color_appearance` 색·외관, `material_quality` 소재·마감, `durability` 내구성·고장,
`performance` 성능·작동, `battery_life` 배터리·충전, `compatibility` 호환성, `ease_of_use` 사용 편의,
`assembly_setup` 조립·설치, `comfort` 착용감·편안함, `effectiveness` 효과(화장품·건강용품),
`skin_reaction` 피부 반응, `smell_taste` 냄새·맛, `noise` 소음, `safety` 안전,
`accuracy_description` 설명·사진과 일치 여부, `packaging_shipping` 포장·배송, `customer_service` 판매자·고객 서비스,
`value_price` 가격 대비 가치, `overall` 전반적 만족(구체적 내용 없음)

### polarity
| 값 | 언제 | 예 |
|---|---|---|
| positive | 좋은 경험 | The battery lasts all day. |
| negative | 나쁜 경험 | The zipper broke within a week. |
| mixed | 한 경험 안에 좋고 나쁨이 같이 있음 | It cleans well but takes twice as long. |
| neutral | 평가 없는 사실 | It runs on two AA batteries. |

### importance (빈도와 무관하게 내용만 보고 판단)
판단 질문: **"이걸 알면 사려는 사람의 결정이나 기대가 달라지나?"** 달라지면 2, 구체적이지만 별 영향이 없으면 1, 정보가 없으면 0.

| 값 | 기준 | 예 |
|---|---|---|
| 0 | 감상만 있고 구체적 정보가 없음 | "Love it!", "Great product", "Would buy again" |
| 1 | 구체적이지만 구매 결정에 덜 중요함 | 색이 사진보다 약간 진함, 포장 상자가 찌그러짐, 배송이 하루 늦음 |
| 2 | 구매 전에 알아야 할 중요한 내용 (긍정·부정 모두) | 특정 기기와 호환 안 됨, 몇 주 만에 고장, 사이즈가 한 치수 작게 나옴, 배터리가 하루 종일 감, 민감한 피부에도 자극 없음 |

- **좋은 내용도 중요하면 2입니다.**
- 같은 항목도 제품에 따라 다릅니다. 옷·화장품의 색은 2일 수 있고, 충전기의 색은 보통 1입니다. 처음 연습할 때 두 사람이 기준을 맞춥니다.
- "Love it", "Great" 같은 감탄은 리뷰마다 줄을 만들지 말고 `overall` 한 줄(importance 0)에 번호를 모두 모읍니다.

### risk_flag
| 값 | 기준 |
|---|---|
| 1 | 리뷰가 **신체적 위험이나 피해**를 직접 보고함: 화상, 과열·연기·불, 감전, 베임·부상, 피부 발진·알레르기, 어린이 질식 위험, 냄새로 인한 두통 등 증상 |
| 0 | 그 외 전부: 고장, 사이즈 불만, 색 차이, 단순히 냄새가 남(증상 없음) |

리뷰에 없는 위험을 추측해서 1로 표시하지 않습니다.

### note 예
- "작다고 했지만 제품 전체 크기인지 특정 부분인지 불분명함."
- "R12는 다른 색상 옵션 이야기일 수 있음."

### 자주 하는 실수
- 경험 두 개를 한 줄에 쓰기 ("Small and breaks easily" → size_fit 줄과 durability 줄로 나눔)
- 언급하지 않은 리뷰를 oppose에 넣기
- 조건이 다른 경험을 한 줄로 합치기 ("처음부터 고장" vs "6개월 뒤 고장"은 두 줄)
- 리뷰 문장을 그대로 복사하기 (proposition은 제품에 대한 일반 문장으로 다시 씀)
