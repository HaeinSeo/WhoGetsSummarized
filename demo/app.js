// Static demo: renders precomputed evidence highlights (no review text, no live crawling).
// Data: data/products.json, produced by scripts/build_demo_data.py.
// Product photos are never fetched from Amazon: the visitor supplies their own image (file, paste, drop, or image URL).

const I18N = {
  en: {
    title: "Who Gets Summarized?",
    intro: "Paste an Amazon link. I'll check every claim in the summary against the reviews and show how many reviewers back it up.",
    go: "Analyze",
    hint: "e.g. amazon.com/dp/B000OFL8QG, or just the ASIN. Or try a sample:",
    placeholder: "Amazon product link or ASIN",
    viewAgent: "Evidence agent v2",
    viewB0: "Direct summary (B0)",
    multi: "2+ reviewers",
    single: "1 reviewer only",
    none: "no support",
    photoEmpty: "Add your product photo: choose a file, drop it here, or paste an image (Ctrl+V).",
    photoUrl: "or paste an image link",
    choose: "Choose photo",
    clear: "Remove",
    bars: { multi: "Backed by 2+ reviewers", single: "Single reviewer", none: "No supporting review" },
    claims: "claims",
    meta: (p) => `${p.category.replace(/_/g, " ")} · ${p.n_reviews} reviews`,
    stats: (s) => `${s.claims} claims · ${s.words} words · hover a highlight to see which reviewers back it`,
    support: (k, n) => `Supported by ${k} of ${n} reviews`,
    none_support: "No supporting review found",
    note: "Support is decided by an NLI judge over all reviews of the product, not by humans. The paper reports human labels for part of the data: single-reviewer experiences are mentioned and stated faithfully less often than multi-reviewer ones. Summaries are in English; only the interface is translated.",
    notInDemo: "This product isn't in the public demo set, which shows precomputed results only. Live analysis needs the local pipeline (see the repository README).",
    badInput: "Couldn't find an ASIN in that input.",
    footer: "Research demo for a WWW short paper. No review text is included; reviewer IDs are anonymised per product.",
  },
  ko: {
    title: "누가 요약되는가?",
    intro: "아마존 링크를 붙여넣으면, 요약의 각 주장이 리뷰로 뒷받침되는지, 몇 명의 리뷰어가 뒷받침하는지 보여드릴게요.",
    go: "분석",
    hint: "예: amazon.com/dp/B000OFL8QG 또는 ASIN만. 샘플도 써볼 수 있어요:",
    placeholder: "아마존 상품 링크 또는 ASIN",
    viewAgent: "근거 에이전트 v2",
    viewB0: "직접 요약 (B0)",
    multi: "리뷰어 2명 이상",
    single: "한 명의 리뷰어만",
    none: "뒷받침 없음",
    photoEmpty: "상품 사진을 넣어 주세요. 파일 선택, 여기에 끌어다 놓기, 또는 이미지 붙여넣기(Ctrl+V).",
    photoUrl: "또는 이미지 링크 붙여넣기",
    choose: "사진 선택",
    clear: "지우기",
    bars: { multi: "리뷰어 2명 이상이 뒷받침", single: "한 명의 리뷰어만", none: "뒷받침하는 리뷰 없음" },
    claims: "주장",
    meta: (p) => `${p.category.replace(/_/g, " ")} · 리뷰 ${p.n_reviews}개`,
    stats: (s) => `주장 ${s.claims}개 · ${s.words}단어 · 형광펜 위에 마우스를 올리면 뒷받침 리뷰어를 볼 수 있어요`,
    support: (k, n) => `전체 ${n}개 리뷰 중 ${k}개가 뒷받침`,
    none_support: "뒷받침하는 리뷰를 찾지 못함",
    note: "근거 판정은 상품의 모든 리뷰에 대한 NLI 판정기가 내린 것이며, 사람이 내린 것이 아닙니다. 논문은 일부 데이터의 사람 라벨을 보고하며, 단일 리뷰어 경험이 다수 리뷰어 경험보다 덜 언급되고 덜 충실하게 서술됨을 보여줍니다. 요약문은 영어이고, 인터페이스만 번역되어 있습니다.",
    notInDemo: "이 상품은 공개 데모 데이터에 없습니다. 공개 데모는 미리 계산된 결과만 보여줍니다. 실시간 분석은 로컬 파이프라인이 필요합니다(저장소 README 참고).",
    badInput: "입력에서 ASIN을 찾지 못했습니다.",
    footer: "WWW 단편 논문의 연구 데모. 리뷰 원문은 포함하지 않으며, 리뷰어 ID는 상품별로 익명 처리되었습니다.",
  },
  ja: {
    title: "誰の意見が要約されるのか？",
    intro: "Amazonのリンクを貼り付けてください。要約の各主張がレビューに裏付けられているか、何人のレビュアーが支持しているかを表示します。",
    go: "分析",
    hint: "例: amazon.co.jp/dp/B000OFL8QG またはASINのみ。サンプルも試せます:",
    placeholder: "Amazon商品リンクまたはASIN",
    viewAgent: "根拠エージェント v2",
    viewB0: "直接要約 (B0)",
    multi: "レビュアー2名以上",
    single: "レビュアー1名のみ",
    none: "裏付けなし",
    photoEmpty: "商品写真を追加してください。ファイル選択、ここにドロップ、または画像を貼り付け(Ctrl+V)。",
    photoUrl: "または画像リンクを貼り付け",
    choose: "写真を選択",
    clear: "削除",
    bars: { multi: "2名以上のレビューが裏付け", single: "レビュアー1名のみ", none: "裏付けるレビューなし" },
    claims: "件",
    meta: (p) => `${p.category.replace(/_/g, " ")} · レビュー ${p.n_reviews}件`,
    stats: (s) => `主張 ${s.claims}件 · ${s.words}語 · ハイライトにマウスを乗せると裏付けレビュアーを表示`,
    support: (k, n) => `全 ${n} 件のレビューのうち ${k} 件が裏付け`,
    none_support: "裏付けるレビューが見つからない",
    note: "根拠の判定は、商品のすべてのレビューに対するNLI判定器によるもので、人手によるものではありません。論文では一部データの人手ラベルを報告しており、単一レビュアーの経験は複数レビュアーの経験より言及・忠実な記述が少ないことを示しています。要約文は英語のままで、画面表示のみ翻訳しています。",
    notInDemo: "この商品は公開デモのデータに含まれていません。公開デモは事前に計算された結果のみ表示します。リアルタイム分析にはローカルのパイプラインが必要です（リポジトリのREADMEを参照）。",
    badInput: "入力からASINを見つけられませんでした。",
    footer: "WWW短編論文の研究デモ。レビュー本文は含まれず、レビュアーIDは商品ごとに匿名化されています。",
  },
  zh: {
    title: "谁的观点会被总结？",
    intro: "粘贴亚马逊链接，我会逐条核对摘要中的论断，并显示有多少位评论者支持它。",
    go: "分析",
    hint: "例如：amazon.com/dp/B000OFL8QG 或直接输入 ASIN。也可以试试示例：",
    placeholder: "亚马逊商品链接或 ASIN",
    viewAgent: "证据代理 v2",
    viewB0: "直接摘要 (B0)",
    multi: "2 位及以上评论者",
    single: "仅 1 位评论者",
    none: "无支持",
    photoEmpty: "请添加商品照片：选择文件、拖到这里，或粘贴图片（Ctrl+V）。",
    photoUrl: "或粘贴图片链接",
    choose: "选择照片",
    clear: "移除",
    bars: { multi: "2 位及以上评论者支持", single: "仅 1 位评论者", none: "没有评论支持" },
    claims: "条",
    meta: (p) => `${p.category.replace(/_/g, " ")} · ${p.n_reviews} 条评论`,
    stats: (s) => `论断 ${s.claims} 条 · ${s.words} 词 · 将鼠标移到高亮上可查看支持的评论者`,
    support: (k, n) => `${n} 条评论中有 ${k} 条支持`,
    none_support: "未找到支持的评论",
    note: "支持与否由 NLI 判定器基于该商品的全部评论做出，并非人工判断。论文报告了部分数据的人工标注，显示单人经验被提及和忠实陈述的频率低于多人经验。摘要为英文，界面已翻译。",
    notInDemo: "该商品不在公开演示数据中。公开演示仅显示预先计算的结果。实时分析需要本地流程（参见仓库 README）。",
    badInput: "无法从输入中找到 ASIN。",
    footer: "WWW 短文论文的研究演示。不包含评论原文，评论者 ID 已按商品匿名化。",
  },
};

// Local analysis of pasted reviews (needs the backend: python -m uvicorn wgs.serve:app --port 8000).
const BACKEND = "http://127.0.0.1:8000";
const LOCAL = {
  en: { title: "Analyze your own reviews", hint: "Paste 3+ reviews, one per line. Runs on your computer; nothing is uploaded.", go: "Analyze pasted reviews", ph: "Great product, works every day.\n\nBroke after two weeks...", on: "Local backend connected", off: "Local backend not running", running: "Analyzing on your computer (about a minute)...", few: "Paste at least 3 reviews (20+ characters each).", fail: "Analysis failed", heading: "Your reviews", sub: (m, j) => `Local run · ${m} · judge: ${j}` },
  ko: { title: "내 리뷰 직접 분석", hint: "리뷰를 한 줄에 하나씩 3개 이상 붙여넣으세요. 내 컴퓨터에서 실행되며 아무것도 업로드되지 않습니다.", go: "붙여넣은 리뷰 분석", ph: "정말 좋아요, 매일 잘 써요.\n\n2주 만에 망가졌어요...", on: "로컬 백엔드 연결됨", off: "로컬 백엔드가 꺼져 있음", running: "내 컴퓨터에서 분석 중 (약 1분)...", few: "20자 이상 리뷰를 3개 이상 붙여넣으세요.", fail: "분석 실패", heading: "내 리뷰", sub: (m, j) => `로컬 실행 · ${m} · 판정기: ${j}` },
  ja: { title: "自分のレビューを分析", hint: "レビューを1行に1件、3件以上貼り付けてください。お使いのPCで実行され、何もアップロードされません。", go: "貼り付けたレビューを分析", ph: "とても良い商品です。毎日使っています。\n\n2週間で壊れました...", on: "ローカルバックエンド接続中", off: "ローカルバックエンドが停止中", running: "お使いのPCで分析中（約1分）...", few: "20文字以上のレビューを3件以上貼り付けてください。", fail: "分析に失敗しました", heading: "あなたのレビュー", sub: (m, j) => `ローカル実行 · ${m} · 判定器: ${j}` },
  zh: { title: "分析你自己的评论", hint: "每行粘贴一条评论，至少 3 条。在你的电脑上运行，不会上传任何内容。", go: "分析粘贴的评论", ph: "非常好用，每天都在用。\n\n两周后就坏了……", on: "本地后端已连接", off: "本地后端未运行", running: "正在你的电脑上分析（约一分钟）...", few: "请至少粘贴 3 条（每条 20 字以上）。", fail: "分析失败", heading: "你的评论", sub: (m, j) => `本地运行 · ${m} · 判定器：${j}` },
};

Object.assign(LOCAL.en, {
  fetching: "Fetching public App Store reviews on your computer (about a minute)...",
  fromStore: (c, id, n, d) => `Fetched ${n} reviews from the App Store (${c.toUpperCase()}); ${d} non-English dropped · judge: NLI (English)`,
});
Object.assign(LOCAL.ko, {
  fetching: "내 컴퓨터에서 공개 App Store 리뷰를 가져오는 중 (약 1분)...",
  fromStore: (c, id, n, d) => `App Store(${c.toUpperCase()})에서 리뷰 ${n}개 수집; 영어가 아닌 리뷰 ${d}개 제외 · 판정기: NLI (영어)`,
});
Object.assign(LOCAL.ja, {
  fetching: "お使いのPCで公開App Storeのレビューを取得中（約1分）...",
  fromStore: (c, id, n, d) => `App Store(${c.toUpperCase()})からレビュー ${n}件取得; 英語以外 ${d}件を除外 · 判定器: NLI（英語）`,
});
Object.assign(LOCAL.zh, {
  fetching: "正在你的电脑上获取公开的 App Store 评论（约一分钟）...",
  fromStore: (c, id, n, d) => `从 App Store（${c.toUpperCase()}）获取 ${n} 条评论；排除 ${d} 条非英文评论 · 判定器：NLI（英文）`,
});

Object.assign(LOCAL.en, {
  judgeNote: (l) => `Judged with a multilingual NLI model for ${l}. This is not validated yet, so treat the counts as indicative.`,
});
Object.assign(LOCAL.ko, {
  judgeNote: (l) => `${l} 리뷰는 다국어 NLI 판정기로 판단했습니다. 아직 검증되지 않아 개수는 참고용입니다.`,
});
Object.assign(LOCAL.ja, {
  judgeNote: (l) => `${l}のレビューは多言語NLI判定器で判定しました。まだ検証されていないため、件数は参考値です。`,
});
Object.assign(LOCAL.zh, {
  judgeNote: (l) => `${l}评论使用多语言 NLI 判定器判断。尚未验证，数量仅供参考。`,
});

const MODE = document.body.dataset.mode || "user";
let lang = "en";
let data = null;
let current = null;
let view = "agent";
let photoUrl = null; // object URL of the visitor's own photo, kept in memory only

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
  $("photo-url").placeholder = s.photoUrl;
  $("t-pick").textContent = s.choose;
  const L = LOCAL[lang];
  $("t-local-title").textContent = L.title;
  $("t-local-hint").textContent = L.hint;
  $("t-local-go").textContent = L.go;
  $("reviews-in").placeholder = L.ph;
  checkBackend();
  $("photo-empty").textContent = s.photoEmpty;
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

// Non-English results use a multilingual judge that is not validated yet; say so on the user page too.
function renderJudgeNote(p) {
  const note = $("judge-note");
  const unvalidated = p.b0 && p.b0.judge_validated === false;
  note.hidden = !unvalidated;
  note.textContent = unvalidated ? LOCAL[lang].judgeNote(p.b0.language_name || p.b0.language) : "";
}

// Test page only: claim table with per-claim support, and the raw response.
function renderDebug(p, r) {
  const box = $("debug");
  if (MODE !== "test") return;
  box.hidden = false;
  const tbl = $("claims-table");
  tbl.textContent = "";
  const head = document.createElement("tr");
  ["#", "support", "reviews", "proposition"].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    head.appendChild(th);
  });
  tbl.appendChild(head);
  r.claims.forEach((c, i) => {
    const tr = document.createElement("tr");
    [String(i + 1), String(c.n_support), c.reviews.join(", ") || "-", c.proposition || ""].forEach((v) => {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    });
    tbl.appendChild(tr);
  });
  $("raw").textContent = JSON.stringify({ ...r, claims: r.claims }, null, 2);
}

function renderBars(r) {
  const s = t();
  const n = r.claims.length || 1;
  const kinds = ["multi", "single", "none"];
  const box = $("bars");
  box.textContent = "";
  kinds.forEach((k) => {
    const cnt = r.claims.filter((c) => classOf(c) === k).length;
    const pct = Math.round((100 * cnt) / n);
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = s.bars[k];
    const num = document.createElement("span");
    num.className = "bar-num";
    num.textContent = `${cnt} · ${pct}%`;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = `bar-fill ${k}`;
    track.appendChild(fill);
    row.append(label, num, track);
    box.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${pct}%`; }));
  });
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
    mk.title = seg.claim.n_support === 0
      ? s.none_support
      : `${s.support(seg.claim.n_support, p.n_reviews)}: ${seg.claim.reviews.join(", ")}`;
    box.appendChild(mk);
  });

  $("stats").textContent = s.stats({ claims: r.claims.length, words: r.word_count });
  if (p.local) {
    const L = LOCAL[lang];
    $("meta").textContent = p.source
      ? L.fromStore(p.source.country, p.source.app_id, p.source.fetched, p.source.dropped_non_english)
      : L.sub(p.b0.model, p.b0.judge);
    $("product-title").textContent = p.source ? `App Store · ${p.source.app_id}` : L.heading;
    document.querySelector(".seg").hidden = true;
  } else {
    $("meta").textContent = s.meta(p);
    $("product-title").textContent = `ASIN ${p.product_id}`;
    document.querySelector(".seg").hidden = false;
  }
  renderBars(r);
  renderJudgeNote(p);
  renderDebug(p, r);
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
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

// Visitor's own product photo: file, paste, drop, or an image link. Never uploaded or fetched from Amazon.
function setPhoto(src) {
  if (photoUrl && photoUrl.startsWith("blob:")) URL.revokeObjectURL(photoUrl);
  photoUrl = src;
  const img = $("photo-img");
  img.src = src;
  img.hidden = false;
  $("photo-empty").hidden = true;
}

function clearPhoto() {
  if (photoUrl && photoUrl.startsWith("blob:")) URL.revokeObjectURL(photoUrl);
  photoUrl = null;
  $("photo-img").removeAttribute("src");
  $("photo-img").hidden = true;
  $("photo-empty").hidden = false;
  $("photo-file").value = "";
  $("photo-url").value = "";
}

function fromFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  setPhoto(URL.createObjectURL(file));
}

async function checkBackend() {
  const el = $("backend");
  if (!el) return;
  try {
    const r = await fetch(`${BACKEND}/health`, { cache: "no-store" });
    const ok = r.ok;
    el.textContent = ok ? LOCAL[lang].on : LOCAL[lang].off;
    el.className = `backend ${ok ? "ok" : "off"}`;
  } catch {
    el.textContent = LOCAL[lang].off;
    el.className = "backend off";
  }
}

async function analyzeLocal() {
  const L = LOCAL[lang];
  const text = $("reviews-in").value;
  const n = text.split(/\n\s*\n|\n/).filter((x) => x.trim().length >= 20).length;
  if (n < 3) { $("status").textContent = L.few; return; }
  $("status").textContent = L.running;
  $("local-card").classList.add("busy");
  try {
    const r = await fetch(`${BACKEND}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reviews_text: text,
        category: "product",
        language: MODE === "test" ? $("lang-override").value : "auto",
      }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || r.statusText);
    current = { product_id: "LOCAL", category: "product", n_reviews: body.n_reviews, local: true, b0: body, agent: body };
    view = "b0";
    $("status").textContent = "";
    $("result").hidden = false;
    render();
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    $("status").textContent = `${L.fail}: ${e.message}`;
  } finally {
    $("local-card").classList.remove("busy");
  }
}

// App Store link: the local backend fetches public reviews itself, then analyzes them.
async function analyzeAppStore(url) {
  const L = LOCAL[lang];
  $("status").textContent = L.fetching;
  try {
    const r = await fetch(`${BACKEND}/analyze-url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, max_reviews: 100 }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || r.statusText);
    const src = body.source;
    current = { product_id: "APPSTORE", category: "app", n_reviews: body.n_reviews, local: true,
                source: src, b0: body, agent: body };
    view = "b0";
    $("status").textContent = "";
    $("result").hidden = false;
    render();
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    $("status").textContent = `${L.fail}: ${e.message}`;
  }
}

function initChips() {
  const box = $("chips");
  if (MODE !== "test" || !box) return;
  box.textContent = "";
  data.products.forEach((p) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip-btn";
    b.textContent = `${p.product_id} · ${p.category.replace(/_/g, " ")}`;
    b.addEventListener("click", () => { $("url").value = p.product_id; show(p.product_id); });
    box.appendChild(b);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const saved = (navigator.language || "en").slice(0, 2);
  lang = I18N[saved] ? saved : "en";
  $("lang").value = lang;
  applyLang();

  $("lang").addEventListener("change", (e) => { lang = e.target.value; applyLang(); });
  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (/apps\.apple\.com\//i.test($("url").value)) { analyzeAppStore($("url").value); return; }
    const asin = parseAsin($("url").value);
    if (!asin) { $("status").textContent = t().badInput; $("result").hidden = true; return; }
    show(asin);
  });
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    view = b.dataset.view;
    if (current) render();
  }));

  $("t-local-go").addEventListener("click", analyzeLocal);

  // photo slot
  $("photo-file").addEventListener("change", (e) => fromFile(e.target.files[0]));
  $("photo-url").addEventListener("change", (e) => {
    const v = e.target.value.trim();
    if (/^https?:\/\//i.test(v)) setPhoto(v);
  });
  const photo = $("photo");
  photo.addEventListener("dragover", (e) => { e.preventDefault(); photo.classList.add("drag"); });
  photo.addEventListener("dragleave", () => photo.classList.remove("drag"));
  photo.addEventListener("drop", (e) => {
    e.preventDefault();
    photo.classList.remove("drag");
    fromFile(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item && current) fromFile(item.getAsFile());
  });

  fetch("data/products.json")
    .then((res) => res.json())
    .then((payload) => { data = payload; initChips(); })
    .catch(() => { $("status").textContent = "Could not load data/products.json"; });
});
