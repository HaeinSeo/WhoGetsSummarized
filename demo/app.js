// Static demo: renders precomputed evidence highlights (no review text, no live crawling).
// Data: data/products.json, produced by scripts/build_demo_data.py.

const I18N = {
  en: {
    title: "Who Gets Summarized?",
    intro: "Paste an Amazon product link. The demo shows which claims in a review summary are backed by reviews, and how many reviewers back each one.",
    go: "Show",
    hint: "Examples: amazon.com/dp/B000OFL8QG or a bare ASIN. Or pick a sample below.",
    placeholder: "Amazon product link or ASIN",
    sample: "Sample products",
    viewAgent: "Evidence agent v2",
    viewB0: "Direct summary (B0)",
    multi: "2+ reviews support it",
    single: "1 reviewer only",
    none: "no review supports it",
    meta: (p) => `${p.category.replace(/_/g, " ")} · ${p.n_reviews} reviews`,
    stats: (s) => `${s.claims} claims · ${s.multi} multi-reviewer · ${s.single} single-reviewer · ${s.none} unsupported · ${s.words} words`,
    support: (k, n) => `Supported by ${k} of ${n} reviews`,
    none_support: "No supporting review found",
    note: "Support is decided by an NLI judge over all reviews of the product, not by humans. Human labels for part of the data are reported in the paper, and they show that single-reviewer experiences are mentioned and stated faithfully less often than multi-reviewer ones. Summaries are in English; the interface is translated only.",
    notInDemo: "This product is not in the public demo set. The public demo shows precomputed results only. Live analysis needs the local pipeline (see the repository README).",
    badInput: "Could not find an ASIN in that input.",
    footer: "Research demo for the WWW short paper (anonymised for review). Review text is not included. Reviewer IDs are anonymised per product.",
  },
  ko: {
    title: "누가 요약되는가?",
    intro: "아마존 상품 링크를 붙여넣으면, 리뷰 요약의 각 주장이 리뷰로 뒷받침되는지, 그리고 몇 명의 리뷰어가 뒷받침하는지 보여줍니다.",
    go: "보기",
    hint: "예: amazon.com/dp/B000OFL8QG 또는 ASIN만 입력. 아래 샘플을 선택할 수도 있습니다.",
    placeholder: "아마존 상품 링크 또는 ASIN",
    sample: "샘플 상품",
    viewAgent: "근거 에이전트 v2",
    viewB0: "직접 요약 (B0)",
    multi: "리뷰 2명 이상이 뒷받침",
    single: "한 명의 리뷰어만",
    none: "뒷받침하는 리뷰 없음",
    meta: (p) => `${p.category.replace(/_/g, " ")} · 리뷰 ${p.n_reviews}개`,
    stats: (s) => `주장 ${s.claims}개 · 다수 리뷰어 ${s.multi} · 단일 리뷰어 ${s.single} · 근거 없음 ${s.none} · ${s.words}단어`,
    support: (k, n) => `전체 ${n}개 리뷰 중 ${k}개가 뒷받침`,
    none_support: "뒷받침하는 리뷰를 찾지 못함",
    note: "근거 판정은 상품의 모든 리뷰에 대한 NLI 판정기가 내린 것이며, 사람이 내린 것이 아닙니다. 논문은 일부 데이터의 사람 라벨을 보고하며, 단일 리뷰어 경험이 다수 리뷰어 경험보다 덜 언급되고 덜 충실하게 서술됨을 보여줍니다. 요약문은 영어이고, 인터페이스만 번역되어 있습니다.",
    notInDemo: "이 상품은 공개 데모 데이터에 없습니다. 공개 데모는 미리 계산된 결과만 보여줍니다. 실시간 분석은 로컬 파이프라인이 필요합니다(저장소 README 참고).",
    badInput: "입력에서 ASIN을 찾지 못했습니다.",
    footer: "WWW 단편 논문의 연구 데모(심사용 익명 처리). 리뷰 원문은 포함하지 않으며, 리뷰어 ID는 상품별로 익명 처리되었습니다.",
  },
  ja: {
    title: "誰の意見が要約されるのか？",
    intro: "Amazon商品のリンクを貼り付けると、レビュー要約の各主張がレビューに裏付けられているか、何人のレビュアーが裏付けているかを表示します。",
    go: "表示",
    hint: "例: amazon.co.jp/dp/B000OFL8QG やASINのみ。下のサンプルを選ぶこともできます。",
    placeholder: "Amazon商品リンクまたはASIN",
    sample: "サンプル商品",
    viewAgent: "根拠エージェント v2",
    viewB0: "直接要約 (B0)",
    multi: "2名以上のレビューが裏付け",
    single: "レビュアー1名のみ",
    none: "裏付けるレビューなし",
    meta: (p) => `${p.category.replace(/_/g, " ")} · レビュー ${p.n_reviews}件`,
    stats: (s) => `主張 ${s.claims}件 · 複数レビュアー ${s.multi} · 単一レビュアー ${s.single} · 根拠なし ${s.none} · ${s.words}語`,
    support: (k, n) => `全 ${n} 件のレビューのうち ${k} 件が裏付け`,
    none_support: "裏付けるレビューが見つからない",
    note: "根拠の判定は、商品のすべてのレビューに対するNLI判定器によるもので、人手によるものではありません。論文では一部データの人手ラベルを報告しており、単一レビュアーの経験は複数レビュアーの経験より言及・忠実な記述が少ないことを示しています。要約文は英語のままで、画面表示のみ翻訳しています。",
    notInDemo: "この商品は公開デモのデータに含まれていません。公開デモは事前に計算された結果のみ表示します。リアルタイム分析にはローカルのパイプラインが必要です（リポジトリのREADMEを参照）。",
    badInput: "入力からASINを見つけられませんでした。",
    footer: "WWW短編論文の研究デモ（査読用に匿名化）。レビュー本文は含まれず、レビュアーIDは商品ごとに匿名化されています。",
  },
  zh: {
    title: "谁的观点会被总结？",
    intro: "粘贴亚马逊商品链接，即可查看评论摘要中的每条论断是否有评论支持，以及有多少位评论者支持它。",
    go: "查看",
    hint: "例如：amazon.com/dp/B000OFL8QG 或直接输入 ASIN。也可以选择下方的示例商品。",
    placeholder: "亚马逊商品链接或 ASIN",
    sample: "示例商品",
    viewAgent: "证据代理 v2",
    viewB0: "直接摘要 (B0)",
    multi: "2 位及以上评论者支持",
    single: "仅 1 位评论者",
    none: "没有评论支持",
    meta: (p) => `${p.category.replace(/_/g, " ")} · ${p.n_reviews} 条评论`,
    stats: (s) => `论断 ${s.claims} 条 · 多人支持 ${s.multi} · 单人支持 ${s.single} · 无支持 ${s.none} · ${s.words} 词`,
    support: (k, n) => `${n} 条评论中有 ${k} 条支持`,
    none_support: "未找到支持的评论",
    note: "支持与否由 NLI 判定器基于该商品的全部评论做出，并非人工判断。论文报告了部分数据的人工标注，显示单人经验被提及和忠实陈述的频率低于多人经验。摘要为英文，界面已翻译。",
    notInDemo: "该商品不在公开演示数据中。公开演示仅显示预先计算的结果。实时分析需要本地流程（参见仓库 README）。",
    badInput: "无法从输入中找到 ASIN。",
    footer: "WWW 短文论文的研究演示（为匿名评审而处理）。不包含评论原文，评论者 ID 已按商品匿名化。",
  },
};

let lang = "en";
let data = null;
let current = null;
let view = "agent";

const $ = (id) => document.getElementById(id);
const t = () => I18N[lang];

function applyLang() {
  const s = t();
  document.documentElement.lang = lang;
  $("t-title").textContent = s.title;
  $("t-intro").textContent = s.intro;
  $("t-go").textContent = s.go;
  $("t-hint").textContent = s.hint;
  $("url").placeholder = s.placeholder;
  $("t-view-agent").textContent = s.viewAgent;
  $("t-view-b0").textContent = s.viewB0;
  $("t-leg-multi").textContent = s.multi;
  $("t-leg-single").textContent = s.single;
  $("t-leg-none").textContent = s.none;
  $("t-note").textContent = s.note;
  $("t-footer").textContent = s.footer;
  if (current) render();
}

// ASIN from a product link (any Amazon domain) or a bare 10-character ASIN.
function parseAsin(input) {
  const v = input.trim();
  const m = v.match(/\/(?:dp|gp\/product|product)\/([A-Z0-9]{10})(?:[\/?#]|$)/i);
  if (m) return m[1].toUpperCase();
  if (/^[A-Z0-9]{10}$/i.test(v)) return v.toUpperCase();
  return null;
}

// Split the summary at all claim boundaries; each piece takes the strongest support of the claims covering it.
function segments(text, claims) {
  const cuts = new Set([0, text.length]);
  claims.forEach((c) => { cuts.add(c.start); cuts.add(c.end); });
  const pts = [...cuts].filter((x) => x >= 0 && x <= text.length).sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    if (b <= a) continue;
    const covering = claims.filter((c) => c.start <= a && c.end >= b);
    let best = null;
    covering.forEach((c) => { if (!best || c.n_support > best.n_support) best = c; });
    out.push({ text: text.slice(a, b), claim: best });
  }
  return out;
}

function classOf(c) {
  if (!c) return null;
  if (c.n_support === 0) return "none";
  if (c.n_support === 1) return "single";
  return "multi";
}

function render() {
  const s = t();
  const p = current;
  const r = p[view];
  const box = $("summary");
  box.textContent = "";
  segments(r.text, r.claims).forEach((seg) => {
    const cls = classOf(seg.claim);
    if (!cls) { box.appendChild(document.createTextNode(seg.text)); return; }
    const mk = document.createElement("mark");
    mk.className = cls;
    mk.textContent = seg.text;
    const sup = seg.claim.n_support === 0
      ? s.none_support
      : `${s.support(seg.claim.n_support, p.n_reviews)}: ${seg.claim.reviews.join(", ")}`;
    mk.title = sup;
    box.appendChild(mk);
  });

  const n = r.claims.length;
  const count = (k) => r.claims.filter((c) => classOf(c) === k).length;
  $("stats").textContent = s.stats({ claims: n, multi: count("multi"), single: count("single"),
    none: count("none"), words: r.word_count });
  $("meta").textContent = s.meta(p);
  document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
}

function show(asin) {
  const s = t();
  const p = data.products.find((x) => x.product_id === asin);
  if (!p) {
    $("status").textContent = s.notInDemo;
    $("result").hidden = true;
    current = null;
    return;
  }
  $("status").textContent = "";
  current = p;
  $("result").hidden = false;
  render();
}

function init(payload) {
  data = payload;
  const sel = document.createElement("select");
  sel.id = "sample";
  sel.setAttribute("aria-label", t().sample);
  const first = document.createElement("option");
  first.textContent = t().sample;
  first.value = "";
  sel.appendChild(first);
  data.products.forEach((p) => {
    const o = document.createElement("option");
    o.value = p.product_id;
    o.textContent = `${p.product_id} · ${p.category.replace(/_/g, " ")}`;
    sel.appendChild(o);
  });
  sel.addEventListener("change", () => { if (sel.value) { $("url").value = sel.value; show(sel.value); } });
  $("form").after(sel);
  sel.style.margin = "10px 0 0";
}

document.addEventListener("DOMContentLoaded", () => {
  const saved = (navigator.language || "en").slice(0, 2);
  lang = I18N[saved] ? saved : "en";
  $("lang").value = lang;
  applyLang();

  $("lang").addEventListener("change", (e) => { lang = e.target.value; applyLang(); });
  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    const asin = parseAsin($("url").value);
    if (!asin) { $("status").textContent = t().badInput; $("result").hidden = true; return; }
    show(asin);
  });
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    view = b.dataset.view;
    if (current) render();
  }));

  fetch("data/products.json")
    .then((res) => res.json())
    .then(init)
    .catch(() => { $("status").textContent = "Could not load data/products.json"; });
});
