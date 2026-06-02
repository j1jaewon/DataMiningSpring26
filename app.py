import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
import plotly.express as px

from recommendation_logic import build_similarity, recommend, grade_label
from db_setup import (
    setup_db, get_conn, save_similarity, save_campaign,
    SQL_BRANDS, SQL_CREATORS, SQL_CAMPAIGNS, SQL_RATINGS, SQL_SIMILARITY,
    SQL_COLLAB_COUNT, SQL_COLLAB_SUCCESS, SQL_SIMILAR_CASES, P1_SQL3,
    DB_PATH,
)

st.set_page_config(
    page_title="Creator Match",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 디자인 시스템 ─────────────────────────────────────────────────────────────
GRADE_COLOR  = {"A": "#1a7a4a", "B": "#2d6a9f", "C": "#b07c00", "D": "#c0392b"}
GRADE_BG     = {"A": "#e8f7ef", "B": "#e8f0fb", "C": "#fdf6e3", "D": "#fdecea"}
GRADE_BORDER = {"A": "#a3d9b8", "B": "#a3c0e8", "C": "#e8d5a0", "D": "#f0a8a0"}
GRADE_LABEL  = {"A": "강력 추천", "B": "추천", "C": "보통", "D": "참고"}
RANK_MEDAL   = {1: "🥇", 2: "🥈", 3: "🥉"}
PLOTLY_COLORS = [
    "#2d6a9f", "#4a87bb", "#6aa3d0", "#8ec0e4", "#b0d4ee",
    "#1a7a4a", "#3a9a6a", "#6abf90", "#9fdcb8", "#c5edd8",
    "#8a6aaa",
]

# ── 전역 CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stDataFrame,
.stSelectbox, .stSlider, button, input, textarea {
    font-family: 'Noto Sans KR', sans-serif !important;
}

/* 탭 스타일 */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    font-family: 'Noto Sans KR', sans-serif !important;
}

/* 추천 받기 버튼 색상 통일 */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a5c, #2d6a9f) !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2d6a9f, #1a3a5c) !important;
    box-shadow: 0 4px 14px rgba(45,106,159,0.35) !important;
}

/* 카드 hover */
.creator-card {
    border: 1px solid #e0e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    background: white;
    transition: box-shadow 0.2s, transform 0.2s;
    height: 100%;
}
.creator-card:hover {
    box-shadow: 0 6px 24px rgba(45,106,159,0.13);
    transform: translateY(-2px);
}
/* KPI 카드 */
.kpi-card {
    background: white;
    border: 1px solid #e0e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.kpi-value { font-size: 2rem; font-weight: 800; color: #1a3a5c; }
.kpi-label { font-size: 0.82rem; color: #888; margin-top: 0.2rem; }
/* 섹션 헤더 */
.section-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #4a6080;
    margin: 1.4rem 0 0.5rem;
    padding: 0.3rem 0.7rem;
    background: #f0f4f8;
    border-radius: 6px;
    display: inline-block;
    letter-spacing: 0.2px;
}
/* 사유 태그 */
.reason-tag {
    display: inline-block;
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0.1rem 0.1rem 0 0;
}
</style>
""", unsafe_allow_html=True)

# ── creator.db 초기화 ─────────────────────────────────────────────────────────
if not os.path.exists(DB_PATH):
    with st.spinner("creator.db 초기화 중... (최초 1회)"):
        setup_db()

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = get_conn()
    creators   = pd.read_sql(SQL_CREATORS,   conn)
    brands     = pd.read_sql(SQL_BRANDS,     conn)
    collabs    = pd.read_sql(SQL_CAMPAIGNS,  conn)
    ratings    = pd.read_sql(SQL_RATINGS,    conn)
    cnt = conn.execute("SELECT COUNT(*) FROM CreatorSimilarity").fetchone()[0]
    similarity = pd.read_sql(SQL_SIMILARITY, conn) if cnt > 0 else None
    conn.close()
    return creators, brands, collabs, ratings, similarity

@st.cache_data
def load_collab_stats():
    conn = get_conn()
    cnt_df  = pd.read_sql(SQL_COLLAB_COUNT,   conn)
    succ_df = pd.read_sql(SQL_COLLAB_SUCCESS, conn)
    conn.close()
    return (dict(zip(cnt_df['Creator_ID'],  cnt_df['cnt'])),
            dict(zip(succ_df['Creator_ID'], succ_df['cnt'])))

build_similarity_cached = st.cache_data(build_similarity)

creators, brands, collabs, ratings, similarity_df = load_data()

if similarity_df is None:
    with st.spinner("추천 점수를 계산 중입니다... (최초 1회)"):
        similarity_df = build_similarity_cached(creators, brands, ratings)
        save_similarity(similarity_df)

collab_count, collab_success = load_collab_stats()
max_followers = creators['Followers'].max()
name_map_c  = dict(zip(creators['Creator_ID'], creators['Channel_Name']))
brand_name_map = dict(zip(brands['Brand_ID'], brands['Brand_Name']))

# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────
def fmt_followers(n):
    if n >= 100_000_000: return f"{n/100_000_000:.1f}억"
    if n >= 10_000:      return f"{n/10_000:.1f}만"
    return f"{n:,}"

def build_reasons(row, brand_row):
    pos, neg = [], []
    if row['category_score'] >= 1.0:   pos.append("카테고리 일치")
    elif row['category_score'] > 0:    pos.append("카테고리 유사")
    else:                              neg.append("카테고리 불일치")
    if row['context_score'] >= 0.5:    pos.append("오디언스 적합")
    elif row['context_score'] < 0.25:  neg.append("오디언스 미스매칭")
    if row['Engagement_Rate'] >= 5.0:  pos.append("높은 참여율")
    elif row['Engagement_Rate'] < 2.0: neg.append("낮은 참여율")
    if row['cf_score'] > 0:            pos.append("협업 이력 반영")
    return pos, neg

def reason_tags_html(pos, neg):
    tags = "".join(
        f"<span class='reason-tag' style='background:#e8f7ef;color:#1a7a4a;'>✔ {r}</span>"
        for r in pos
    )
    tags += "".join(
        f"<span class='reason-tag' style='background:#fdecea;color:#c0392b;'>✖ {r}</span>"
        for r in neg
    )
    return tags

def plotly_score_bar(row):
    labels = ['카테고리(CBF)', '조건매칭(CBF)', '협업필터링(CF)']
    values = [row['category_score'], row['context_score'], row['cf_score']]
    colors = ['#6a9ec8', '#4aaa7a', '#e8c97a']
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.2f}" for v in values],
        textposition='outside',
        width=0.45,
    ))
    fig.update_layout(
        height=160, margin=dict(l=0, r=50, t=10, b=10),
        xaxis=dict(range=[0, 1.15], showgrid=False, visible=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Noto Sans KR, sans-serif', size=11),
        bargap=0.5,
    )
    return fig

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
            padding: 2rem 2.5rem; border-radius: 14px; margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(26,58,92,0.18);
            display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;'>
    <div>
        <div style='color:#a8c8e8; font-size:0.75rem; font-weight:600; letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:0.35rem;'>
            KAIST BIZ &nbsp;·&nbsp; 비즈니스 애널리틱스 2026
        </div>
        <h1 style='color: white; margin: 0; font-size: 2.2rem; font-weight:800;
                   letter-spacing:-1px; line-height:1.15;'>
            Creator <span style='color:#7ec8f0;'>Match</span>
        </h1>
        <p style='color: #c8dff0; margin: 0.45rem 0 0; font-size: 0.92rem; font-weight:400;'>
            AI 기반 기업·크리에이터 매칭 플랫폼
        </p>
    </div>
    <div style='text-align:right;'>
        <div style='color:white; font-size:0.78rem; opacity:0.7; margin-bottom:0.3rem;'>
            데이터 현황
        </div>
        <div style='display:flex; gap:1.2rem;'>
            <div style='text-align:center;'>
                <div style='color:white; font-size:1.3rem; font-weight:800;'>490</div>
                <div style='color:#a8c8e8; font-size:0.7rem;'>크리에이터</div>
            </div>
            <div style='text-align:center;'>
                <div style='color:white; font-size:1.3rem; font-weight:800;'>100</div>
                <div style='color:#a8c8e8; font-size:0.7rem;'>브랜드</div>
            </div>
            <div style='text-align:center;'>
                <div style='color:white; font-size:1.3rem; font-weight:800;'>976</div>
                <div style='color:#a8c8e8; font-size:0.7rem;'>협업 이력</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 소개 문구 ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#f5f8fc;border-radius:12px;padding:1rem 1.5rem;
            margin:0.2rem 0 1.2rem;border-left:4px solid #2d6a9f;'>
    <div style='font-size:1.15rem;font-weight:800;color:#1a3a5c;letter-spacing:-0.3px;'>
        브랜드 ↔ 크리에이터, 양방향 매칭
    </div>
    <div style='font-size:0.85rem;color:#666;margin-top:0.3rem;line-height:1.6;'>
        976건의 실제 협업 데이터로 학습 &nbsp;·&nbsp; AI가 최적 파트너를 점수로 추천합니다
        &nbsp;&nbsp;
        <span style='color:#2d6a9f;font-weight:600;'>🎯 브랜드→크리에이터 추천</span>
        &nbsp;|&nbsp;
        <span style='color:#1a7a4a;font-weight:600;'>🔍 크리에이터→브랜드 탐색</span>
        &nbsp;|&nbsp;
        <span style='color:#b07c00;font-weight:600;'>📊 캠페인 성과 분석</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 메인 탭 ───────────────────────────────────────────────────────────────────
tab_match, tab_explore, tab_dashboard = st.tabs([
    "🎯 브랜드 매칭", "🔍 크리에이터 탐색", "📊 성과 대시보드"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: 브랜드 매칭
# ════════════════════════════════════════════════════════════════════════════
with tab_match:

    # ① 브랜드 조건 입력
    st.markdown("<div class='section-title'>① 브랜드 조건 입력</div>", unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            brand_options = brands[['Brand_ID', 'Brand_Name', 'Industry']].copy()
            brand_display = brand_options.apply(
                lambda r: f"{r['Brand_Name']} ({r['Industry']})", axis=1
            ).tolist()
            selected_idx = st.selectbox("브랜드 선택", range(len(brand_display)),
                                        format_func=lambda i: brand_display[i])
            brand_id  = brand_options.iloc[selected_idx]['Brand_ID']
            brand_row = brands[brands['Brand_ID'] == brand_id].iloc[0]

        with col2:
            rc1, rc2 = st.columns([8, 1])
            with rc1:
                risk_threshold = st.slider("최소 Risk Score", 1.0, 5.0, 2.5, 0.5)
            with rc2:
                st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
                with st.popover("❓"):
                    st.markdown("""
**Risk Score란?**

크리에이터의 콘텐츠 신뢰도·브랜드 안전성 지수입니다.

| 점수 | 의미 |
|------|------|
| 4.0 ~ 5.0 | 🟢 우수 — 브랜드 안전 |
| 3.0 ~ 4.0 | 🟡 보통 — 검토 권장 |
| 2.5 ~ 3.0 | 🟠 주의 — 선별 필요 |
| 2.5 미만   | 🔴 제외 — 자동 필터링 |

> 기본값 2.5 미만은 추천에서 자동 제외됩니다.
                    """)
        with col3:
            gc1, gc2 = st.columns([8, 1])
            with gc1:
                top_n = st.slider("추천 크리에이터 수", 1, 10, 3)
            with gc2:
                st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
                with st.popover("❓"):
                    st.markdown("""
**추천 등급 기준**

매칭 점수 = 카테고리×0.3 + 조건매칭×0.3 + 협업필터링×0.4

| 등급 | 점수 | 캠페인 성공률 |
|------|------|-------------|
| 🏆 A | 0.9 이상 | **75.7%** |
| 🥈 B | 0.8 ~ 0.9 | 58.8% |
| 🥉 C | 0.7 ~ 0.8 | 46.6% |
| ⬇️ D | 0.7 미만  | 20.0% |

> **등급 A 이상(0.8+) 크리에이터를 추천합니다.**
                    """)


        st.markdown(f"""
        <div style='background:linear-gradient(90deg,#f0f4f8,#e8f0fb);
                    border-radius:10px; padding:0.8rem 1.2rem; margin-top:0.4rem;
                    display:flex; gap:2rem; flex-wrap:wrap; font-size:0.87rem; color:#444;'>
            <span>🏢 <b>{brand_row['Brand_Name']}</b></span>
            <span>🏷️ {brand_row['Industry']}</span>
            <span>💰 월 예산 {brand_row['Monthly_Budget']:,}원</span>
            <span>🎯 타겟 {brand_row['Target_Age']} / {brand_row['Target_Gender']}</span>
            <span>📱 선호 플랫폼 {brand_row['Preferred_Platform']}</span>
        </div>
        """, unsafe_allow_html=True)

        run = st.button("🔍 추천 받기", type="primary", use_container_width=True)

    # ② 추천 결과
    if run or 'last_brand_id' in st.session_state:
        if run:
            st.session_state.update({
                'last_brand_id':       brand_id,
                'last_risk_threshold': risk_threshold,
                'last_top_n':          top_n,
            })

        brand_id       = st.session_state['last_brand_id']
        risk_threshold = st.session_state['last_risk_threshold']
        top_n          = st.session_state['last_top_n']
        brand_row      = brands[brands['Brand_ID'] == brand_id].iloc[0]

        top_df = recommend(brand_id, similarity_df, creators, risk_threshold, top_n)

        st.markdown("<div class='section-title'>② 추천 결과</div>", unsafe_allow_html=True)

        if top_df.empty:
            st.warning("조건을 만족하는 크리에이터가 없습니다. Risk Score 기준을 낮춰보세요.")
        else:
            all_cats = ["전체"] + sorted(top_df['Category'].unique().tolist())
            cat_tabs = st.tabs(all_cats)

            for cat_tab, cat_label in zip(cat_tabs, all_cats):
                with cat_tab:
                    filtered = top_df if cat_label == "전체" \
                               else top_df[top_df['Category'] == cat_label]
                    if filtered.empty:
                        st.info("해당 카테고리의 추천 결과가 없습니다.")
                        continue

                    rows_list = list(filtered.iterrows())
                    for row_start in range(0, len(rows_list), 3):
                        chunk = rows_list[row_start:row_start + 3]
                        cols  = st.columns(len(chunk))
                        for col, (_, row) in zip(cols, chunk):
                            grade  = row.get('recommendation_grade', grade_label(row['matching_score']))
                            color  = GRADE_COLOR[grade]
                            bg     = GRADE_BG[grade]
                            border = GRADE_BORDER[grade]
                            rank_n = int(row['Rank'])
                            medal  = RANK_MEDAL.get(rank_n,
                                         f"<span style='font-family:\"Noto Sans KR\",sans-serif;"
                                         f"font-size:1.1rem;font-weight:700;color:#888;'>"
                                         f"{rank_n}위</span>")
                            pos_reasons, neg_reasons = build_reasons(row, brand_row)
                            tags_html = reason_tags_html(pos_reasons, neg_reasons)
                            c_id       = row['Creator_ID']
                            n_collab   = collab_count.get(c_id, 0)
                            n_success  = collab_success.get(c_id, 0)
                            follow_pct = min(int(row['Followers'] / max_followers * 100), 100)
                            score_pct  = int(row['matching_score'] * 100)
                            with col:
                                card_html = (
                                    f"<div class='creator-card' style='border-color:{border};"
                                    f"border-top:3px solid {color};'>"
                                    f"<div style='display:flex;justify-content:space-between;"
                                    f"align-items:center;margin-bottom:0.5rem;'>"
                                    f"<span style='font-size:1.5rem;'>{medal}</span>"
                                    f"<span style='background:{bg};color:{color};"
                                    f"border-radius:20px;padding:0.15rem 0.6rem;"
                                    f"font-size:0.75rem;font-weight:700;'>{GRADE_LABEL[grade]}</span>"
                                    f"</div>"
                                    f"<div style='font-size:1.1rem;font-weight:800;color:#1a3a5c;"
                                    f"margin-bottom:0.3rem;letter-spacing:-0.3px;'>{row['Channel_Name']}</div>"
                                    f"<div style='font-size:0.82rem;color:#888;margin-bottom:0.8rem;'>"
                                    f"{row['Platform']} &nbsp;·&nbsp; {row['Category']}</div>"
                                    f"<div style='margin-bottom:0.8rem;'>"
                                    f"<div style='display:flex;justify-content:space-between;"
                                    f"font-size:0.78rem;color:#666;margin-bottom:0.3rem;'>"
                                    f"<span>매칭 점수</span>"
                                    f"<span style='font-weight:700;color:{color};'>"
                                    f"{row['matching_score']:.2f} &nbsp; 등급 {grade}</span></div>"
                                    f"<div style='background:#f0f0f0;border-radius:6px;height:8px;'>"
                                    f"<div style='background:linear-gradient(90deg,{color}88,{color});"
                                    f"height:8px;border-radius:6px;width:{score_pct}%;'></div></div></div>"
                                    f"<div style='margin-bottom:0.8rem;'>"
                                    f"<div style='display:flex;justify-content:space-between;"
                                    f"font-size:0.78rem;color:#666;margin-bottom:0.3rem;'>"
                                    f"<span>👥 구독자</span>"
                                    f"<span style='font-weight:600;'>{fmt_followers(row['Followers'])}</span></div>"
                                    f"<div style='background:#f0f0f0;border-radius:6px;height:6px;'>"
                                    f"<div style='background:#a3c0e8;height:6px;border-radius:6px;"
                                    f"width:{follow_pct}%;'></div></div></div>"
                                    f"<div style='display:flex;gap:0.6rem;margin-bottom:0.8rem;"
                                    f"font-size:0.78rem;'>"
                                    f"<div style='flex:1;background:#f8f9fa;border-radius:8px;"
                                    f"padding:0.4rem;text-align:center;'>"
                                    f"<div style='color:#888;font-size:0.7rem;'>참여율</div>"
                                    f"<div style='font-weight:700;color:#1a3a5c;'>{row['Engagement_Rate']}%</div></div>"
                                    f"<div style='flex:1;background:#f8f9fa;border-radius:8px;"
                                    f"padding:0.4rem;text-align:center;'>"
                                    f"<div style='color:#888;font-size:0.7rem;'>협업</div>"
                                    f"<div style='font-weight:700;color:#1a3a5c;'>{n_collab}회</div></div>"
                                    f"<div style='flex:1;background:#f8f9fa;border-radius:8px;"
                                    f"padding:0.4rem;text-align:center;'>"
                                    f"<div style='color:#888;font-size:0.7rem;'>Risk</div>"
                                    f"<div style='font-weight:700;color:#1a3a5c;'>{row['Risk_Score']}</div></div></div>"
                                    f"<div>{tags_html}</div>"
                                    f"</div>"
                                )
                                st.markdown(card_html, unsafe_allow_html=True)
                                with st.expander("📊 상세 분석"):
                                    st.plotly_chart(plotly_score_bar(row),
                                                    use_container_width=True,
                                                    config={'displayModeBar': False},
                                                    key=f"score_bar_{brand_id}_{cat_label}_{c_id}")
                                    past = collabs[collabs['Creator_ID'] == c_id][
                                        ['Brand_ID', 'CTR', 'CVR', 'is_success']
                                    ].copy()
                                    if not past.empty:
                                        past['브랜드'] = past['Brand_ID'].map(brand_name_map)
                                        past['성공']   = past['is_success'].map({'Y': '✅', 'N': '❌'})
                                        st.caption("과거 협업 성과")
                                        st.dataframe(
                                            past[['브랜드', 'CTR', 'CVR', '성공']].head(5),
                                            use_container_width=True, hide_index=True
                                        )

            # ③ 매칭 점수 분포
            st.markdown("<div class='section-title'>③ 매칭 점수 분포</div>",
                        unsafe_allow_html=True)
            with st.container(border=True):
                brand_scores = similarity_df[similarity_df['Brand_ID'] == brand_id].copy()
                risk_map_all = dict(zip(creators['Creator_ID'], creators['Risk_Score']))
                brand_scores['Risk_Score'] = brand_scores['Creator_ID'].map(risk_map_all)
                brand_scores = brand_scores[brand_scores['Risk_Score'] >= risk_threshold]

                bins   = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                labels = ['~0.4', '0.4~0.5', '0.5~0.6', '0.6~0.7',
                          '0.7~0.8', '0.8~0.9', '0.9~']
                bin_colors = ['#e4e4e4','#d4d4d4','#c4d4e4','#a0bcd4',
                              '#e8c97a','#6a9ec8','#4aaa7a']
                hist_data = pd.cut(brand_scores['matching_score'],
                                   bins=bins, labels=labels).value_counts().sort_index()

                y_max = int(hist_data.max() * 1.25) + 1
                fig_hist = go.Figure(go.Bar(
                    x=hist_data.index.tolist(),
                    y=hist_data.values,
                    marker_color=bin_colors,
                    marker_line_width=0,
                    text=hist_data.values,
                    textposition='outside',
                    width=0.45,
                ))
                fig_hist.update_layout(
                    height=220, margin=dict(l=0, r=0, t=24, b=0),
                    plot_bgcolor='white', paper_bgcolor='white',
                    yaxis=dict(range=[0, y_max], showgrid=True,
                               gridcolor='#f0f0f0', tickfont=dict(size=11)),
                    xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                    font=dict(family='Noto Sans KR, sans-serif', size=12),
                    bargap=0.3,
                )

                col_chart, col_info = st.columns([2, 1])
                with col_chart:
                    st.plotly_chart(fig_hist, use_container_width=True,
                                    config={'displayModeBar': False})
                with col_info:
                    st.markdown("**등급별 현황**")
                    total = len(brand_scores)
                    grade_ranges = [("A", 0.9, 1.1), ("B", 0.8, 0.9),
                                    ("C", 0.7, 0.8), ("D", 0.0, 0.7)]
                    for g, lo, hi in grade_ranges:
                        cnt = ((brand_scores['matching_score'] >= lo) &
                               (brand_scores['matching_score'] < hi)).sum()
                        pct = cnt / total * 100 if total > 0 else 0
                        bar_w = int(pct)
                        st.markdown(f"""
                        <div style='margin-bottom:0.5rem;'>
                            <div style='display:flex; justify-content:space-between;
                                        font-size:0.82rem; margin-bottom:0.2rem;'>
                                <span style='color:{GRADE_COLOR[g]};font-weight:700;'>
                                    등급 {g}
                                </span>
                                <span style='color:#555;'>{cnt}명 ({pct:.0f}%)</span>
                            </div>
                            <div style='background:#f0f0f0;border-radius:4px;height:5px;'>
                                <div style='background:{GRADE_COLOR[g]};height:5px;
                                            border-radius:4px;width:{bar_w}%;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # ④ 유사 협업 사례
            st.markdown("<div class='section-title'>④ 유사 협업 사례</div>",
                        unsafe_allow_html=True)
            top_creator_ids = top_df['Creator_ID'].tolist()
            placeholders    = ','.join('?' * len(top_creator_ids))
            conn = get_conn()
            _similar_sql = (
                "SELECT camp.Brand_ID, camp.Creator_ID, b.Brand_Name,"
                " c.Channel_Name AS Creator_Name, camp.Budget_Spent,"
                " camp.Impressions, camp.CTR, camp.CVR, camp.is_success"
                " FROM Campaign camp"
                " JOIN Brand b ON camp.Brand_ID = b.Brand_ID"
                " JOIN Creator c ON camp.Creator_ID = c.Creator_ID"
                f" WHERE b.Industry = ? AND camp.Creator_ID IN ({placeholders})"
                " AND camp.is_success = 'Y'"
                " LIMIT 5"
            )
            cases = pd.read_sql(
                _similar_sql,
                conn, params=[brand_row['Industry']] + top_creator_ids,
            )
            conn.close()

            if cases.empty:
                st.info("동일 업종의 성공 협업 사례가 없습니다.")
            else:
                with st.container(border=True):
                    st.markdown(
                        "<div style='display:grid;grid-template-columns:2fr 2fr 1fr 1fr 1fr;"
                        "gap:0.3rem;padding:0.35rem 0.5rem;background:#f5f7fa;"
                        "border-radius:8px;font-size:0.75rem;font-weight:700;color:#888;"
                        "margin-bottom:0.4rem;'>"
                        "<span>기업</span><span>크리에이터</span>"
                        "<span style='text-align:right;'>노출</span>"
                        "<span style='text-align:right;'>CTR</span>"
                        "<span style='text-align:right;'>CVR</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    for _, c in cases.iterrows():
                        st.markdown(
                            "<div style='display:grid;grid-template-columns:2fr 2fr 1fr 1fr 1fr;"
                            f"gap:0.3rem;padding:0.3rem 0.5rem;font-size:0.82rem;border-bottom:1px solid #f0f0f0;'>"
                            f"<span style='font-weight:600;color:#1a3a5c;'>✅ {c['Brand_Name']}</span>"
                            f"<span style='color:#444;'>{c['Creator_Name']}</span>"
                            f"<span style='text-align:right;color:#555;'>{c['Impressions']:,}</span>"
                            f"<span style='text-align:right;font-weight:600;color:#2d6a9f;'>{c['CTR']}%</span>"
                            f"<span style='text-align:right;font-weight:600;color:#1a7a4a;'>{c['CVR']}%</span>"
                            "</div>",
                            unsafe_allow_html=True,
                        )

            # ⑤ 같은 업종 브랜드 비교
            st.markdown("<div class='section-title'>⑤ 같은 업종 브랜드 비교</div>",
                        unsafe_allow_html=True)
            with st.container(border=True):
                compare_brands = brands[
                    (brands['Industry'] == brand_row['Industry']) &
                    (brands['Brand_ID'] != brand_id)
                ].head(4)

                comp_rows = []
                # 현재 브랜드
                cur_sc  = similarity_df[similarity_df['Brand_ID'] == brand_id]
                cur_avg = cur_sc['matching_score'].mean()
                cur_top = cur_sc.nlargest(1, 'matching_score')
                cur_top_name = name_map_c.get(
                    cur_top.iloc[0]['Creator_ID'], "") if not cur_top.empty else ""
                comp_rows.append({
                    '브랜드': f"⭐ {brand_row['Brand_Name']}",
                    '평균 매칭점수': round(cur_avg, 3),
                    'Top 크리에이터': cur_top_name,
                    '_current': True,
                })
                for _, br in compare_brands.iterrows():
                    br_sc = similarity_df[similarity_df['Brand_ID'] == br['Brand_ID']]
                    avg   = br_sc['matching_score'].mean()
                    top1  = br_sc.nlargest(1, 'matching_score')
                    top1_name = name_map_c.get(
                        top1.iloc[0]['Creator_ID'], "") if not top1.empty else ""
                    comp_rows.append({
                        '브랜드': br['Brand_Name'],
                        '평균 매칭점수': round(avg, 3),
                        'Top 크리에이터': top1_name,
                        '_current': False,
                    })

                comp_df = pd.DataFrame(comp_rows)
                bar_colors = [
                    "#2d6a9f" if r else "#b0c8d8"
                    for r in comp_df['_current']
                ]
                comp_h = max(180, len(comp_df) * 44 + 40)
                fig_comp = go.Figure(go.Bar(
                    y=comp_df['브랜드'],
                    x=comp_df['평균 매칭점수'],
                    orientation='h',
                    marker_color=bar_colors,
                    text=[f"{v:.3f}" for v in comp_df['평균 매칭점수']],
                    textposition='outside',
                ))
                fig_comp.update_layout(
                    height=comp_h, margin=dict(l=0, r=60, t=10, b=10),
                    plot_bgcolor='white', paper_bgcolor='white',
                    xaxis=dict(range=[0, 1.05], showgrid=False, visible=False),
                    yaxis=dict(showgrid=False, autorange='reversed'),
                    font=dict(family='Noto Sans KR, sans-serif', size=12),
                    bargap=0.5,
                )
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    st.plotly_chart(fig_comp, use_container_width=True,
                                    config={'displayModeBar': False})
                with col_c2:
                    st.dataframe(
                        comp_df[['브랜드', '평균 매칭점수', 'Top 크리에이터']],
                        use_container_width=True, hide_index=True
                    )

            # ⑥ 캠페인 성과 입력
            st.markdown("<div class='section-title'>⑥ 캠페인 성과 입력</div>",
                        unsafe_allow_html=True)
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    creator_options = top_df.apply(
                        lambda r: f"{RANK_MEDAL.get(int(r['Rank']),str(int(r['Rank']))+'위')} "
                                  f"{r['Channel_Name']}", axis=1
                    ).tolist()
                    sel_label = st.selectbox("크리에이터 선택", creator_options)
                    sel_idx   = creator_options.index(sel_label)
                    sel_cid   = top_df.iloc[sel_idx]['Creator_ID']
                with col2:
                    impressions_input = st.number_input("실제 노출수", min_value=0, step=1000)
                with col3:
                    ctr_input = st.number_input("CTR (%)", min_value=0.0,
                                                max_value=100.0, step=0.1, format="%.2f")
                with col4:
                    success_input = st.selectbox("성공 여부", ["Y", "N"])

                if st.button("💾 성과 저장", type="secondary", use_container_width=True):
                    new_row = {
                        'Collab_ID':      f"CB_NEW_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                        'Brand_ID':       brand_id,
                        'Creator_ID':     sel_cid,
                        'Campaign_Start': str(pd.Timestamp.now().date()),
                        'Campaign_End':   str(pd.Timestamp.now().date()),
                        'Budget_Spent':   0,
                        'Impressions':    impressions_input,
                        'Clicks':         int(impressions_input * ctr_input / 100),
                        'CTR':            ctr_input,
                        'Conversions':    0,
                        'CVR':            0,
                        'is_success':     success_input,
                    }
                    save_campaign(new_row)
                    st.success(f"성과가 저장되었습니다! ({new_row['Collab_ID']})")
                    st.cache_data.clear()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: 크리에이터 탐색
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("<div class='section-title'>크리에이터 → 맞는 브랜드 탐색</div>",
                unsafe_allow_html=True)
    st.caption("크리에이터 관점에서 협업 가능성이 높은 브랜드를 역방향으로 조회합니다.")

    with st.container(border=True):
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            cat_filter = st.selectbox(
                "카테고리", ["전체"] + sorted(creators['Category'].unique().tolist()),
                key="explore_cat_filter")
        with ecol2:
            plat_filter = st.selectbox(
                "플랫폼", ["전체"] + sorted(creators['Platform'].unique().tolist()),
                key="explore_plat_filter")
        with ecol3:
            min_risk = st.slider("최소 Risk Score", 1.0, 5.0, 2.5, 0.5,
                                 key="explore_min_risk")

    fc = creators.copy()
    if cat_filter  != "전체": fc = fc[fc['Category'] == cat_filter]
    if plat_filter != "전체": fc = fc[fc['Platform']  == plat_filter]
    fc = fc[fc['Risk_Score'] >= min_risk]

    if fc.empty:
        st.warning("조건에 맞는 크리에이터가 없습니다.")
    else:
        sel_creator = st.selectbox("크리에이터 선택", fc['Channel_Name'].tolist(),
                                   key="explore_creator_select")
        sel_cid_exp = fc[fc['Channel_Name'] == sel_creator].iloc[0]['Creator_ID']
        ci          = fc[fc['Creator_ID'] == sel_cid_exp].iloc[0]

        # 프로필 카드
        n_c = collab_count.get(sel_cid_exp, 0)
        n_s = collab_success.get(sel_cid_exp, 0)
        succ_rate = int(n_s / n_c * 100) if n_c > 0 else 0
        with st.container(border=True):
            metrics = [
                ("플랫폼",     ci['Platform']),
                ("카테고리",   ci['Category']),
                ("구독자",     fmt_followers(ci['Followers'])),
                ("참여율",     f"{ci['Engagement_Rate']}%"),
                ("Risk Score", f"{ci['Risk_Score']}"),
            ]
            cols_m = st.columns(len(metrics))
            for col_m, (label, val) in zip(cols_m, metrics):
                col_m.markdown(
                    f"<div style='font-family:\"Noto Sans KR\",sans-serif;'>"
                    f"<div style='font-size:0.78rem;color:#888;margin-bottom:0.25rem;'>{label}</div>"
                    f"<div style='font-size:1.45rem;font-weight:700;color:#1a3a5c;letter-spacing:-0.5px;'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='font-family:\"Noto Sans KR\",sans-serif;font-size:0.88rem;"
                f"color:#555;margin-top:0.6rem;'>"
                f"협업 이력 <b>{n_c}회</b> &nbsp;|&nbsp; 성공 <b>{n_s}회</b> ({succ_rate}%)"
                f"</div>",
                unsafe_allow_html=True,
            )

        # 맞는 브랜드 Top 10
        st.markdown("#### 이 크리에이터에게 맞는 브랜드 Top 10")
        cs = similarity_df[similarity_df['Creator_ID'] == sel_cid_exp].nlargest(
            10, 'matching_score').copy()
        cs['브랜드']  = cs['Brand_ID'].map(brand_name_map)
        cs['업종']    = cs['Brand_ID'].map(dict(zip(brands['Brand_ID'], brands['Industry'])))
        cs['월예산']  = cs['Brand_ID'].map(dict(zip(brands['Brand_ID'], brands['Monthly_Budget'])))
        cs['등급']    = cs.get('recommendation_grade', cs['matching_score'].apply(grade_label))
        cs['순위']    = range(1, len(cs) + 1)

        col_table, col_bar = st.columns([1, 1])
        with col_table:
            st.dataframe(
                cs[['순위', '브랜드', '업종', '월예산', 'matching_score', '등급']].rename(
                    columns={'matching_score': '매칭점수'}),
                use_container_width=True, hide_index=True
            )
        with col_bar:
            fig_exp = go.Figure(go.Bar(
                y=cs['브랜드'], x=cs['matching_score'],
                orientation='h',
                marker_color=[GRADE_COLOR.get(g, "#888") for g in cs['등급']],
                text=[f"{v:.2f}" for v in cs['matching_score']],
                textposition='outside',
            ))
            exp_h = max(200, len(cs) * 36 + 60)
            fig_exp.update_layout(
                height=exp_h, margin=dict(l=0, r=50, t=10, b=10),
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis=dict(range=[0, 1.05], showgrid=False, visible=False),
                yaxis=dict(showgrid=False, autorange='reversed'),
                font=dict(family='Noto Sans KR, sans-serif', size=11),
                bargap=0.5,
            )
            st.plotly_chart(fig_exp, use_container_width=True,
                            config={'displayModeBar': False})


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: 성과 대시보드
# ════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("<div class='section-title'>캠페인 성과 대시보드</div>",
                unsafe_allow_html=True)

    total_collabs = len(collabs)
    success_cnt   = (collabs['is_success'] == 'Y').sum()
    success_rate  = success_cnt / total_collabs * 100 if total_collabs > 0 else 0
    avg_ctr       = collabs['CTR'].mean()
    avg_cvr       = collabs['CVR'].mean()

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label, color in [
        (k1, f"{total_collabs:,}건", "총 협업 수",  "#1a3a5c"),
        (k2, f"{success_rate:.1f}%", "성공률",      "#1a7a4a"),
        (k3, f"{avg_ctr:.2f}%",      "평균 CTR",   "#2d6a9f"),
        (k4, f"{avg_cvr:.2f}%",      "평균 CVR",   "#b07c00"),
    ]:
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value' style='color:{color};'>{val}</div>
            <div class='kpi-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("**업종별 성공률**")
        brand_ind_map2 = dict(zip(brands['Brand_ID'], brands['Industry']))
        ci2 = collabs.copy()
        ci2['Industry'] = ci2['Brand_ID'].map(brand_ind_map2)
        ind_stats = ci2.groupby('Industry').apply(
            lambda x: round((x['is_success'] == 'Y').mean() * 100, 1)
        ).reset_index()
        ind_stats.columns = ['업종', '성공률(%)']
        ind_stats = ind_stats.sort_values('성공률(%)')

        n_ind = len(ind_stats)
        ind_opacities = [0.35 + 0.55 * i / max(n_ind - 1, 1) for i in range(n_ind)]
        fig_ind = go.Figure(go.Bar(
            y=ind_stats['업종'], x=ind_stats['성공률(%)'],
            orientation='h',
            marker=dict(
                color=["rgba(60,140,100," + f"{op:.2f})" for op in ind_opacities]
            ),
            text=[f"{v}%" for v in ind_stats['성공률(%)']],
            textposition='outside',
        ))
        ind_h = max(200, len(ind_stats) * 36 + 60)
        fig_ind.update_layout(
            height=ind_h, margin=dict(l=0, r=50, t=10, b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(range=[0, 100], showgrid=False, visible=False),
            yaxis=dict(showgrid=False),
            font=dict(family='Noto Sans KR, sans-serif', size=12),
            bargap=0.5,
        )
        st.plotly_chart(fig_ind, use_container_width=True,
                        config={'displayModeBar': False})

    with dcol2:
        st.markdown("**카테고리별 평균 CTR**")
        creator_cat_map = dict(zip(creators['Creator_ID'], creators['Category']))
        cc = collabs.copy()
        cc['Category'] = cc['Creator_ID'].map(creator_cat_map)
        cat_ctr = cc.groupby('Category')['CTR'].mean().round(2).reset_index()
        cat_ctr.columns = ['카테고리', '평균CTR(%)']
        cat_ctr = cat_ctr.sort_values('평균CTR(%)')

        n_ctr = len(cat_ctr)
        ctr_opacities = [0.35 + 0.55 * i / max(n_ctr - 1, 1) for i in range(n_ctr)]
        fig_ctr = go.Figure(go.Bar(
            y=cat_ctr['카테고리'], x=cat_ctr['평균CTR(%)'],
            orientation='h',
            marker=dict(
                color=["rgba(80,130,180," + f"{op:.2f})" for op in ctr_opacities]
            ),
            text=[f"{v}%" for v in cat_ctr['평균CTR(%)']],
            textposition='outside',
        ))
        ctr_h = max(200, len(cat_ctr) * 36 + 60)
        fig_ctr.update_layout(
            height=ctr_h, margin=dict(l=0, r=50, t=10, b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False),
            font=dict(family='Noto Sans KR, sans-serif', size=12),
            bargap=0.5,
        )
        st.plotly_chart(fig_ctr, use_container_width=True,
                        config={'displayModeBar': False})

    st.divider()
    dcol3, dcol4 = st.columns(2)

    with dcol3:
        st.markdown("**성공 협업 Top 10 크리에이터**")
        top_c = (collabs[collabs['is_success'] == 'Y']
                 .groupby('Creator_ID').size().nlargest(10).reset_index())
        top_c.columns = ['Creator_ID', '성공횟수']
        top_c['크리에이터'] = top_c['Creator_ID'].map(name_map_c)
        top_c = top_c.sort_values('성공횟수')

        fig_top = go.Figure(go.Bar(
            y=top_c['크리에이터'], x=top_c['성공횟수'],
            orientation='h',
            marker_color="#1a7a4a",
            text=top_c['성공횟수'], textposition='outside',
        ))
        top_h = max(200, len(top_c) * 36 + 60)
        fig_top.update_layout(
            height=top_h, margin=dict(l=0, r=40, t=10, b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False),
            font=dict(family='Noto Sans KR, sans-serif', size=12),
            bargap=0.5,
        )
        st.plotly_chart(fig_top, use_container_width=True,
                        config={'displayModeBar': False})

    with dcol4:
        st.markdown("**노출수 vs CTR (성공/실패)**")
        sample = collabs.sample(min(300, len(collabs)), random_state=42).copy()
        sample['결과'] = sample['is_success'].map({'Y': '성공', 'N': '실패'})

        fig_sc = px.scatter(
            sample, x='Impressions', y='CTR',
            color='결과',
            color_discrete_map={'성공': '#1a7a4a', '실패': '#c0392b'},
            opacity=0.65,
            labels={'Impressions': '노출수', 'CTR': 'CTR (%)'},
        )
        fig_sc.update_traces(marker_size=6)
        fig_sc.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='#fafafa', paper_bgcolor='white',
            legend=dict(title='', orientation='h', y=1.08),
            font=dict(family='Noto Sans KR, sans-serif', size=12),
        )
        st.plotly_chart(fig_sc, use_container_width=True,
                        config={'displayModeBar': False})

    st.divider()
    st.markdown("**전체 협업 데이터**")
    dc = collabs.copy()
    dc['크리에이터'] = dc['Creator_ID'].map(name_map_c)
    dc['브랜드']     = dc['Brand_ID'].map(brand_name_map)
    st.dataframe(
        dc[['브랜드', '크리에이터', 'CTR', 'CVR',
            'Impressions', 'Budget_Spent', 'is_success']].rename(
            columns={'is_success': '성공'}),
        use_container_width=True, hide_index=True
    )


# ── 푸터 ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#bbb; font-size:0.78rem;'>"
    "KAIST BIZ &nbsp;|&nbsp; 비즈니스 애널리틱스 2026 &nbsp;|&nbsp; "
    "CBF + CF Hybrid Recommendation System</p>",
    unsafe_allow_html=True
)
