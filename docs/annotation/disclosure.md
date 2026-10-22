# Disclosure labels (D / S / U / A) - guideline v1.0

Label what the **text discloses**, not whether the reviewer was "really" paid.

| label | use when |
|---|---|
| D | explicit free product, discount/rebate/refund for review, "in exchange for", Vine disclosure |
| S | explicit self-paid statement ("bought it myself", "paid full price", "not sponsored") |
| U | read the whole text and found no D/S disclosure (U is NOT "organic" or "unpaid") |
| A | ambiguous or conflicting: gift, sale price, free shipping, negations, D and S together |

`vine`: 1 only if the Vine program is disclosed. `evidence_span`: copy the exact words.
Hard negatives ("free shipping", "bought as a gift", "on sale") are usually U or A, not D.
Work independently; do not discuss items before both sheets are complete.

---
## 한국어 상세 규칙 (열별)

리뷰가 **"보상을 받고 썼다"고 스스로 밝혔는지**만 봅니다. 실제로 보상을 받았는지 추측하지 않습니다.

| 열 | 쓰는 값 | 규칙 |
|---|---|---|
| label | `D` / `S` / `U` / `A` | 아래 표 |
| vine | `0` / `1` | Amazon Vine 프로그램을 언급하면 1, 아니면 0 |
| evidence_span | 리뷰 문구 그대로 복사 | D·S·A의 근거가 된 부분. U면 빈칸 |
| note | 자유 메모 | 애매한 이유 |

### label
| 값 | 기준 | 예 |
|---|---|---|
| D | 무료 제품, 할인, 환불, 리뷰 대가를 받았다고 밝힘. Vine 포함 | "I received this product for free in exchange for my honest review", "Vine Customer Review of Free Product", "got a discount for my review" |
| S | 자기 돈으로 샀다고 밝힘 | "I bought this with my own money", "paid full price", "not sponsored" |
| U | 리뷰 전체를 읽었는데 D나 S에 해당하는 말이 없음 | 일반 리뷰 대부분 |
| A | 판단이 애매하거나 서로 충돌 | D와 S가 같이 나옴, "free"가 있지만 리뷰 대가인지 불분명, "got it as a gift" |

헷갈리는 경우:
| 문구 | label | 이유 |
|---|---|---|
| "free shipping" | U | 배송비 무료는 리뷰 대가가 아님 |
| "on sale", "great price" | U | 세일은 리뷰 대가가 아님 |
| "my friend gave it to me as a gift" | A (또는 U) | 판매자가 준 건지 불분명하면 A |
| "the seller sent me a replacement for free" | U | 불량 교환은 리뷰 대가가 아님 |
| "I was not paid for this review" | S | 보상이 없다고 밝힘 |
| "received a sample" (대가 언급 없음) | A | 리뷰 대가인지 불분명 |
