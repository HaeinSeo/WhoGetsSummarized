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

---
## 한국어 상세 규칙 (열별)

`experience`(요약에 나온 주장 하나)와 `review_text`(리뷰 하나)를 보고, **이 리뷰가 그 주장에 대해 무엇을 말하는지** 판단합니다. 주장이 사실인지, 다른 리뷰들은 어떤지는 보지 않습니다.

| 열 | 쓰는 값 | 규칙 |
|---|---|---|
| relation | `SUPPORT` / `OPPOSE` / `IRRELEVANT` / `UNCERTAIN` | 아래 순서도 |
| condition_match | `Y` / `N` / `NA` | 아래 표 |
| note | 자유 메모 | 애매한 이유 |

### relation 순서도
1. 리뷰가 이 주장과 **같은 속성**(배터리, 사이즈 등)을 이야기하나? 아니면 → `IRRELEVANT`
2. 같은 경험을 **명시적으로** 말하나? 그렇다면 → 3번으로. 반대 경험이면 → 4번으로
3. 조건이 맞거나 주장에 조건이 없으면 → `SUPPORT`. 조건이 다르거나 확인할 수 없으면 → `UNCERTAIN`
4. 비슷한 조건에서의 반대 경험이면 → `OPPOSE`. 조건이 달라 비교가 안 되면 → `UNCERTAIN`
5. 같은 리뷰 안에 찬반이 섞여 있거나, 무슨 뜻인지 애매하면 → `UNCERTAIN`

| 주장 (experience) | 리뷰 | relation | condition_match |
|---|---|---|---|
| The battery lasts a full day. | "Lasted all day at work." | SUPPORT | Y |
| The battery lasts a full day. | "Died after 3 hours." | OPPOSE | Y |
| The battery lasts a full day. | "Love the color." | IRRELEVANT | NA |
| It overheats (during long gaming sessions). | "Gets really hot after 2 hours of gaming." | SUPPORT | Y |
| It overheats (during long gaming sessions). | "Gets warm while charging." | UNCERTAIN | N |
| The case does not fit (iPhone 15 Pro). | "Didn't fit my Galaxy S23." | UNCERTAIN | N |
| Some users had it break. | "Mine never broke in a year." | OPPOSE | Y |
| The battery lasts a full day. | "Battery is fine some days, dead by noon on others." | UNCERTAIN | Y |

### condition_match
| 값 | 언제 |
|---|---|
| Y | 주장에 조건이 없거나, 리뷰의 조건이 주장의 조건과 같음 |
| N | 주장에 조건이 있는데 리뷰의 조건이 다르거나 확인할 수 없음 |
| NA | relation이 IRRELEVANT일 때 |

### 주의
- "잘 모르겠다"는 IRRELEVANT가 아니라 **UNCERTAIN**입니다.
- "some", "most", "many" 같은 빈도 표현은 무시하고 경험 자체만 봅니다.
- 반어법·비꼼은 실제 의미로 판단합니다 ("Great, it broke on day one" → 부정).
- 주장 문장이 이상하게 쪼개져 있어 판단이 불가능하면 UNCERTAIN, note에 "주장 분해 오류"라고 씁니다.
