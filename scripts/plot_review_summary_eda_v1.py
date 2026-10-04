"""Create presentation-ready figures from completed review-summary EDA tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "review_summary_eda_v1" / "full"
COLORS = {
    "ink": "#27313B",
    "muted": "#667481",
    "grid": "#E5E9ED",
    "teal": "#237F78",
    "teal_light": "#A9D6CF",
    "coral": "#D56557",
    "gold": "#D8A342",
    "blue": "#4A78A5",
    "violet": "#786C9B",
    "pale": "#F3F6F7",
}


def setup_style() -> None:
    fonts = set(font_manager.get_font_names())
    if "Malgun Gothic" not in fonts:
        raise RuntimeError("Korean charts require the installed Malgun Gothic font")
    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelcolor": COLORS["muted"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8)


def title(fig: plt.Figure, heading: str, subtitle: str, note: str) -> None:
    fig.text(0.055, 0.96, heading, fontsize=17, weight="bold", va="top")
    fig.text(0.055, 0.915, subtitle, fontsize=10, color=COLORS["muted"], va="top")
    fig.text(0.055, 0.025, note, fontsize=9, color=COLORS["muted"], va="bottom")


def save(fig: plt.Figure, target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        plt.close(fig)
        raise FileExistsError(f"Existing figure preserved: {target}; pass --overwrite to regenerate")
    fig.savefig(target, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(target)


def rating_mix(cat: pd.DataFrame, target: Path, overwrite: bool) -> None:
    top = cat.nlargest(10, "review_count").copy()
    top["low"] = top["rating_1_count"] + top["rating_2_count"]
    top["mid"] = top["rating_3_count"]
    top["high"] = top["rating_4_count"] + top["rating_5_count"]
    total = top[["low", "mid", "high"]].sum(axis=1)
    for col in ("low", "mid", "high"):
        top[col] = 100 * top[col] / total
    top = top.sort_values("low", ascending=True)
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.79, bottom=0.14)
    y = np.arange(len(top))
    left = np.zeros(len(top))
    for col, label, color in (
        ("low", "1~2점", COLORS["coral"]),
        ("mid", "3점", COLORS["gold"]),
        ("high", "4~5점", COLORS["teal"]),
    ):
        widths = top[col].to_numpy()
        ax.barh(y, widths, left=left, height=0.68, color=color, label=label)
        if col in ("low", "high"):
            for i, width in enumerate(widths):
                if width >= 7:
                    ax.text(left[i] + width / 2, i, f"{width:.1f}%", color="white", ha="center", va="center", fontsize=8.5, weight="bold")
        left += widths
    ax.set_yticks(y, top["category"])
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    clean_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    title(fig, "리뷰가 많은 카테고리의 별점 구성", "리뷰 수 상위 10개 카테고리 · 카테고리별 1~5점 리뷰 내부 비율", "별점은 문장 감성의 대리 지표다. 0점 151건은 이 구성 비율에서 제외했다.")
    save(fig, target, overwrite)


def annual_trend(monthly: pd.DataFrame, target: Path, overwrite: bool) -> None:
    annual = monthly.groupby("review_year", as_index=False)[["review_count", "low_rating_count", "mid_rating_count", "high_rating_count"]].sum()
    annual = annual[(annual["review_year"] >= 2010) & (annual["review_year"] <= 2023)].copy()
    years = annual["review_year"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.1]})
    fig.subplots_adjust(left=0.10, right=0.96, top=0.85, bottom=0.12, hspace=0.26)
    colors = [COLORS["teal"]] * (len(annual) - 1) + [COLORS["coral"]]
    ax1.bar(years, annual["review_count"] / 1e6, color=colors, width=0.76)
    ax1.set_ylabel("리뷰 수 (백만 건)")
    ax1.set_ylim(0, annual["review_count"].max() / 1e6 * 1.15)
    clean_axes(ax1, "y")
    mix = annual[["low_rating_count", "mid_rating_count", "high_rating_count"]].div(
        annual[["low_rating_count", "mid_rating_count", "high_rating_count"]].sum(axis=1), axis=0
    ) * 100
    ax2.stackplot(years, mix.iloc[:, 0], mix.iloc[:, 1], mix.iloc[:, 2], colors=[COLORS["coral"], COLORS["gold"], COLORS["teal"]], labels=["1~2점", "3점", "4~5점"], alpha=0.92)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax2.set_xticks([2010, 2012, 2014, 2016, 2018, 2020, 2022, 2023])
    ax2.set_xlabel("리뷰 작성 연도")
    clean_axes(ax2, "y")
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    title(fig, "리뷰 규모와 별점 구성의 시간 변화", "가독성을 위해 2010~2023년을 표시 · 원본 분석은 1996~2023년 전체", "2023년은 9월 중순까지만 수록됐다. 시기별 변화가 요약 편향의 증거는 아니다.")
    save(fig, target, overwrite)


def text_quantiles(cat: pd.DataFrame, target: Path, overwrite: bool) -> None:
    top = cat.nlargest(12, "review_count").sort_values("median_text_chars")
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12.2, 7.4))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.78, bottom=0.13)
    ax.hlines(y, top["p10_text_chars"], top["p90_text_chars"], color=COLORS["teal_light"], linewidth=9, label="10~90분위")
    ax.scatter(top["median_text_chars"], y, marker="D", s=56, color=COLORS["coral"], zorder=3, label="중앙값")
    ax.scatter(top["avg_text_chars"], y, marker="o", s=40, facecolor="white", edgecolor=COLORS["ink"], linewidth=1.6, zorder=3, label="평균")
    ax.set_yticks(y, top["category"])
    ax.set_xlabel("리뷰 본문 문자 수")
    ax.set_xlim(0, max(top["p90_text_chars"]) * 1.06)
    clean_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    title(fig, "카테고리별 리뷰 본문 길이", "리뷰 수 상위 12개 카테고리 · 10~90분위 범위, 중앙값, 평균", "본문 문자 수의 카테고리별 집계다. 평균과 중앙값의 차이는 긴 리뷰의 영향을 보여준다.")
    save(fig, target, overwrite)


def metadata_coverage(join: pd.DataFrame, target: Path, overwrite: bool) -> None:
    groups = [("전체", join), ("이름 있는 카테고리", join[join["category"] != "Unknown"]), ("Unknown", join[join["category"] == "Unknown"])]
    values = []
    for label, frame in groups:
        review_pct = 100 * frame["matched_review_rows"].sum() / frame["review_rows"].sum()
        product_pct = 100 * frame["matched_products"].sum() / frame["products_with_key"].sum()
        values.append((label, review_pct, product_pct))
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    fig.subplots_adjust(left=0.23, right=0.95, top=0.76, bottom=0.18)
    y = np.array([2, 1, 0], dtype=float)
    for i, (label, review_pct, product_pct) in enumerate(values):
        ax.hlines(y[i], min(review_pct, product_pct), max(review_pct, product_pct), color=COLORS["grid"], linewidth=5)
        ax.scatter(review_pct, y[i] + 0.10, s=115, color=COLORS["teal"], zorder=3, label="리뷰 기준" if i == 0 else None)
        ax.scatter(product_pct, y[i] - 0.10, s=115, color=COLORS["coral"], zorder=3, label="제품 기준" if i == 0 else None)
        review_x = review_pct - 1.2 if review_pct >= 95 else review_pct + 1.2
        product_x = product_pct - 1.2 if product_pct >= 95 else product_pct + 1.2
        ax.text(review_x, y[i] + 0.10, f"{review_pct:.1f}%", va="center", ha="right" if review_pct >= 95 else "left", fontsize=9, color=COLORS["teal"])
        ax.text(product_x, y[i] - 0.10, f"{product_pct:.1f}%", va="center", ha="right" if product_pct >= 95 else "left", fontsize=9, color=COLORS["coral"])
    ax.set_yticks(y, [row[0] for row in values])
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    clean_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    title(fig, "메타데이터 연결은 Unknown에서 급감", "parent_asin 조인 · 리뷰 기준과 제품 기준을 별도로 계산", "이름 있는 카테고리는 모두 연결됐다. 미연결 제품을 임의로 제외하지 않았다.")
    save(fig, target, overwrite)


def candidate_thresholds(cand: pd.DataFrame, target: Path, overwrite: bool) -> None:
    thresholds = [20, 50, 100]
    eligible = np.array([cand[f"products_valid_reviews_min{n}"].sum() for n in thresholds]) / 1e6
    both = np.array([cand[f"min{n}_products_with_low_and_high"].sum() for n in thresholds]) / 1e6
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.77, bottom=0.19)
    y = np.arange(3)
    ax.barh(y + 0.16, eligible, height=0.29, color=COLORS["blue"], label="유효 리뷰 조건 충족")
    ax.barh(y - 0.16, both, height=0.29, color=COLORS["coral"], label="그중 고·저평점 공존")
    for i, (e, b) in enumerate(zip(eligible, both)):
        ax.text(e + 0.05, y[i] + 0.16, f"{e:.2f}M", va="center", fontsize=9, color=COLORS["blue"])
        ax.text(b + 0.05, y[i] - 0.16, f"{b:.2f}M  ({b/e:.1%})", va="center", fontsize=9, color=COLORS["coral"])
    ax.set_yticks(y, [f"최소 {n}개" for n in thresholds])
    ax.invert_yaxis()
    ax.set_xlim(0, 4.6)
    ax.set_xlabel("제품 수 (백만 개)")
    clean_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    title(fig, "리뷰 수 기준별 실험 가능 제품", "유효 텍스트 리뷰 최소 20 / 50 / 100개 · 괄호는 해당 기준 내 고·저평점 공존 비율", "고·저평점은 1~2점과 4~5점이다. 공존은 장단점 문장이나 양극화의 라벨이 아니다.")
    save(fig, target, overwrite)


def eligibility_heatmap(cat: pd.DataFrame, cand: pd.DataFrame, target: Path, overwrite: bool) -> None:
    top_categories = cat.nlargest(15, "review_count")["category"].tolist()
    frame = cand.set_index("category").loc[top_categories]
    columns = [f"products_valid_reviews_min{n}" for n in (20, 50, 100)]
    matrix = frame[columns].to_numpy(dtype=float) / frame["products_total"].to_numpy(dtype=float)[:, None] * 100
    fig, ax = plt.subplots(figsize=(10.3, 9.1))
    fig.subplots_adjust(left=0.33, right=0.91, top=0.87, bottom=0.10)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(30, np.ceil(matrix.max() / 5) * 5))
    ax.set_yticks(range(len(top_categories)), top_categories)
    ax.set_xticks(range(3), ["20개 이상", "50개 이상", "100개 이상"])
    ax.tick_params(length=0, pad=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=9, color="white" if matrix[i, j] > im.norm.vmax * 0.58 else COLORS["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.03)
    cbar.set_label("카테고리 제품 중 조건 충족 비율")
    title(fig, "카테고리마다 실험 후보 밀도가 다르다", "리뷰 수 상위 15개 카테고리 · 분모는 각 카테고리의 전체 제품 수", "유효 텍스트 리뷰 기준의 제품 비율이다. 카테고리 크기와는 다른 지표다.")
    save(fig, target, overwrite)


def reviewer_activity(cat: pd.DataFrame, activity: pd.DataFrame, target: Path, overwrite: bool) -> None:
    top = cat.nlargest(10, "review_count")[["category", "review_count"]].merge(activity, on="category").sort_values("p99_reviews_per_user")
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.84, bottom=0.14)
    ax.hlines(y, top["median_reviews_per_user"], top["p99_reviews_per_user"], color=COLORS["grid"], linewidth=5)
    ax.scatter(top["median_reviews_per_user"], y, s=55, color=COLORS["blue"], label="중앙값", zorder=3)
    ax.scatter(top["p90_reviews_per_user"], y, s=60, color=COLORS["gold"], label="90분위", zorder=3)
    ax.scatter(top["p99_reviews_per_user"], y, s=65, color=COLORS["violet"], label="99분위", zorder=3)
    ax.set_yticks(y, top["category"])
    ax.set_xlabel("리뷰어 1명당 해당 카테고리 리뷰 수")
    ax.set_xlim(0, top["p99_reviews_per_user"].max() * 1.15)
    clean_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    title(fig, "리뷰어 활동의 긴 꼬리", "리뷰 수 상위 10개 카테고리 · 리뷰어별 작성 수의 중앙값, 90·99분위", "카테고리별 집계다. 같은 사용자가 여러 카테고리에 등장할 수 있다.")
    save(fig, target, overwrite)


def rule_candidates(rules: pd.DataFrame, compensation: pd.DataFrame, target: Path, overwrite: bool) -> None:
    rule_labels = {
        "high_rating_with_drawback_marker": "고평점 + 단점 단서",
        "low_rating_with_positive_marker": "저평점 + 장점 단서",
        "mixed_positive_drawback_markers": "장·단점 단서 동시",
        "attribute_complaint_marker": "속성 불만 단서",
    }
    comp_labels = {
        "free_product_or_exchange_candidate": "무료 제품·대가",
        "discount_rebate_candidate": "할인·환급",
        "vine_disclosure_candidate": "Vine",
        "self_paid_statement_candidate": "자기 돈 구매 자기보고",
    }
    r = rules.groupby("candidate_type")["sampled_candidate_rows"].sum().reindex(rule_labels)
    c = compensation.groupby("rule_id")["sampled_candidate_rows"].sum().reindex(comp_labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.18, right=0.96, top=0.77, bottom=0.17, wspace=0.42)
    for ax, series, labels, colors, scale, suffix in (
        (ax1, r, rule_labels, [COLORS["coral"], COLORS["blue"], COLORS["violet"], COLORS["gold"]], 1000, "천 건"),
        (ax2, c, comp_labels, [COLORS["teal"], COLORS["gold"], COLORS["violet"], COLORS["blue"]], 1, "건"),
    ):
        values = series.to_numpy(dtype=float) / scale
        y = np.arange(len(values))
        ax.barh(y, values, color=colors, height=0.58)
        ax.set_yticks(y, [labels[key] for key in series.index])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_xlabel(f"표본 내 규칙 후보 ({suffix})")
        for i, value in enumerate(values):
            ax.text(value + max(values) * 0.025, i, f"{int(series.iloc[i]):,}", va="center", fontsize=9)
        clean_axes(ax)
    ax1.set_title("장단점·속성 단서", loc="left", fontsize=12, weight="bold", pad=18)
    ax2.set_title("보상·자기보고 문구", loc="left", fontsize=12, weight="bold", pad=18)
    title(fig, "탐색 규칙은 검수할 후보를 찾는다", "결정적 1/1,000 리뷰 표본 · 왼쪽 유형은 서로 겹칠 수 있음", "규칙 후보 수는 실제 불만·보상 빈도가 아니며 전체 데이터로 외삽하지 않는다.")
    save(fig, target, overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Completed full EDA output directory")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate only these enhanced figures")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "figures" / "enhanced_v1"
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    cat = pd.read_csv(root / "category_summary.csv")
    monthly = pd.read_csv(root / "monthly_summary.csv")
    cand = pd.read_csv(root / "candidate_product_summary.csv")
    join = pd.read_csv(root / "metadata_join_by_category.csv")
    activity = pd.read_csv(root / "reviewer_activity_distribution.csv")
    rules = pd.read_csv(root / "exploratory_rule_summary.csv")
    compensation = pd.read_csv(root / "compensation_disclosure_summary.csv")
    rating_mix(cat, out / "rating_mix_top10.png", args.overwrite)
    annual_trend(monthly, out / "annual_volume_rating_mix.png", args.overwrite)
    text_quantiles(cat, out / "text_length_quantiles_top12.png", args.overwrite)
    metadata_coverage(join, out / "metadata_join_coverage.png", args.overwrite)
    candidate_thresholds(cand, out / "candidate_thresholds.png", args.overwrite)
    eligibility_heatmap(cat, cand, out / "eligibility_heatmap_top15.png", args.overwrite)
    reviewer_activity(cat, activity, out / "reviewer_activity_top10.png", args.overwrite)
    rule_candidates(rules, compensation, out / "exploratory_rule_candidates.png", args.overwrite)


if __name__ == "__main__":
    main()
