import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
import plotly.express as px

from recommendation_logic import (
    build_similarity, recommend, grade_label,
    calc_category_score, calc_context_score,
)
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
GRADE_COLOR  = {"A": "#15803d", "B": "#2433ff", "C": "#9a6207", "D": "#c42626"}
GRADE_BG     = {"A": "#e4f4e7", "B": "rgba(36,51,255,.08)", "C": "#fbf0d9", "D": "#fbe6e4"}
GRADE_BORDER = {"A": "#bbe3c4", "B": "#c0c8ff", "C": "#efd9a8", "D": "#f1c2bd"}
GRADE_LABEL  = {"A": "강력 추천", "B": "추천", "C": "보통", "D": "참고"}
RANK_MEDAL   = {1: "🥇", 2: "🥈", 3: "🥉"}
PLOTLY_COLORS = [
    "#2433ff", "#4a5fff", "#7a87ff", "#a8b0ff", "#d0d4ff",
    "#15803d", "#3a9a6a", "#6abf90", "#9fdcb8", "#c5edd8",
    "#8a6aaa",
]

# ── 전역 CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');

:root {
  --bg: #fafaf9;
  --surface: #ffffff;
  --ink: #0f0f0e;
  --ink-s: #3a3a38;
  --muted: #76766f;
  --faint: #adadA6;
  --line: #e8e8e3;
  --line-s: #d6d6d0;
  --accent: #2433ff;
  --accent-s: rgba(36,51,255,.08);
  --safe: #15803d; --safe-bg: #e4f4e7; --safe-line: #bbe3c4;
  --warn: #9a6207; --warn-bg: #fbf0d9; --warn-line: #efd9a8;
  --risk: #c42626; --risk-bg: #fbe6e4; --risk-line: #f1c2bd;
  --sans: 'Pretendard', 'Pretendard Variable', -apple-system, sans-serif;
  --brand: 'Playfair Display', Georgia, serif;
  --r: 14px; --r-lg: 20px; --r-sm: 9px;
}

html, body, [class*="css"], .stMarkdown, .stDataFrame,
.stSelectbox, .stSlider, button, input, textarea, .stTabs {
    font-family: var(--sans) !important;
    color: var(--ink);
    letter-spacing: -0.01em;
}

/* 배경 */
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] { background: var(--surface) !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px; }

/* 탭 */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--line) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.55rem 1.3rem;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
    border-bottom: 2px solid var(--ink) !important;
    background: transparent !important;
}

/* 버튼 */
.stButton > button[kind="primary"] {
    background: var(--ink) !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em;
    padding: 0.65rem 1.6rem !important;
    transition: background .15s !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--ink-s) !important;
    box-shadow: none !important;
}

/* 크리에이터 카드 */
.creator-card {
    border: 1px solid var(--line);
    border-radius: var(--r-lg);
    padding: 1.5rem 1.4rem;
    background: var(--surface);
    transition: border-color .18s, box-shadow .18s;
    height: 100%;
}
.creator-card:hover {
    border-color: var(--line-s);
    box-shadow: 0 12px 32px -16px rgba(15,14,14,.12);
}

/* KPI 카드 */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--r);
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1;
}
.kpi-label { font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem; }

/* 섹션 헤더 */
.section-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ink);
    margin: 2rem 0 0.8rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid var(--line);
    display: block;
    letter-spacing: -0.2px;
}

/* 사유 태그 */
.reason-tag {
    display: inline-block;
    border-radius: 999px;
    padding: 0.18rem 0.65rem;
    font-size: 0.71rem;
    font-weight: 600;
    margin: 0.1rem 0.1rem 0 0;
    border: 1px solid transparent;
}

/* selectbox / text_area 테두리 */
.stTextArea textarea, .stSelectbox > div > div {
    border-radius: var(--r) !important;
    border-color: var(--line) !important;
    background: var(--surface) !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--line-s) !important;
    box-shadow: 0 0 0 3px var(--accent-s) !important;
}
/* 모바일 대응 */
@media (max-width: 768px) {
  .creator-card { padding: 1.1rem 1rem; }
  .kpi-value { font-size: 1.5rem; }
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

# ── 텍스트 파싱 & 텍스트 기반 추천 ──────────────────────────────────────────
import re as _re

def parse_brand_text(text):
    t = text.lower()

    industry = '뷰티'
    industry_map = {
        '뷰티':    ['뷰티', '스킨케어', '화장', '코스메틱', '메이크업', '향수'],
        '패션':    ['패션', '의류', '옷', '스타일', '코디'],
        '식품':    ['식품', '음식', '요리', '먹', '푸드', '식음료', '베이커리', '카페'],
        '테크':    ['테크', '기술', '전자', 'it', '소프트웨어', '앱', '스타트업'],
        '게임':    ['게임', '게이밍', 'e스포츠'],
        '생활용품':['생활', '가전', '인테리어', '청소', '주방'],
        '피트니스':['피트니스', '운동', '헬스', '다이어트', '스포츠', '요가'],
        '교육':    ['교육', '학습', '강의', '튜터', '어학'],
        '여행':    ['여행', '투어', '관광', '호텔'],
        '헬스케어':['헬스케어', '건강', '의료', '영양', '보건', '비건'],
    }
    for ind, kws in industry_map.items():
        if any(kw in text for kw in kws):
            industry = ind
            break

    target_age = '18-34'
    if any(k in text for k in ['10대', '청소년', '틴']):
        target_age = '13-17'
    elif any(k in text for k in ['5060', '50대', '60대', '중장년', '시니어']):
        target_age = '35-54'
    elif any(k in text for k in ['3040', '40대', '4050']):
        target_age = '25-44'
    elif any(k in text for k in ['2030', '20대', '30대', '젊']):
        target_age = '18-34'

    target_gender = 'Mixed'
    if any(k in text for k in ['여성', '여자', '여성분']):
        target_gender = 'Female'
    elif any(k in text for k in ['남성', '남자', '남성분']):
        target_gender = 'Male'

    platform = 'Mixed'
    if any(k in text for k in ['유튜브', 'youtube']):
        platform = 'YouTube'
    elif any(k in text for k in ['인스타', 'instagram']):
        platform = 'Instagram'
    elif any(k in text for k in ['틱톡', 'tiktok']):
        platform = 'TikTok'

    max_cpm = 5000.0
    m = _re.search(r'(\d[\d,]*)\s*만\s*원', text)
    if m:
        budget_won = int(m.group(1).replace(',', '')) * 10_000
        max_cpm = budget_won / 300.0

    return {
        'Industry':          industry,
        'Target_Age':        target_age,
        'Target_Gender':     target_gender,
        'Preferred_Platform': platform,
        'Max_CPM':           max_cpm,
        'Monthly_Budget':    int(max_cpm * 300),
        'Brand_Name':        '입력된 브랜드',
    }


def recommend_from_text(brand_attrs, creators_df, risk_threshold=2.5, top_n=3):
    rows = []
    for _, c in creators_df.iterrows():
        if c['Risk_Score'] < risk_threshold:
            continue
        cat_score = calc_category_score(brand_attrs['Industry'], c['Category'])
        if cat_score == 0:
            continue
        ctx_score = calc_context_score(brand_attrs, c)
        matching  = round(cat_score * 0.5 + ctx_score * 0.5, 4)
        rows.append({
            'Creator_ID':           c['Creator_ID'],
            'Channel_Name':         c['Channel_Name'],
            'Category':             c['Category'],
            'Platform':             c['Platform'],
            'Followers':            c['Followers'],
            'Engagement_Rate':      c['Engagement_Rate'],
            'Risk_Score':           c['Risk_Score'],
            'category_score':       round(cat_score, 4),
            'context_score':        round(ctx_score, 4),
            'cf_score':             0.0,
            'matching_score':       matching,
            'recommendation_grade': grade_label(matching),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).nlargest(top_n, 'matching_score').copy()
    df['Rank'] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


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
        f"<span class='reason-tag' style='background:var(--safe-bg);color:var(--safe);border-color:var(--safe-line);'>✔ {r}</span>"
        for r in pos
    )
    tags += "".join(
        f"<span class='reason-tag' style='background:var(--risk-bg);color:var(--risk);border-color:var(--risk-line);'>✖ {r}</span>"
        for r in neg
    )
    return tags

def plotly_score_bar(row):
    labels = ['카테고리(CBF)', '조건매칭(CBF)', '협업필터링(CF)']
    values = [row['category_score'], row['context_score'], row['cf_score']]
    colors = ['#2433ff', '#15803d', '#9a6207']
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.2f}" for v in values],
        textposition='outside',
        width=0.4,
    ))
    fig.update_layout(
        height=160, margin=dict(l=0, r=50, t=10, b=10),
        xaxis=dict(range=[0, 1.15], showgrid=False, visible=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color='#76766f')),
        plot_bgcolor='#fafaf9', paper_bgcolor='#fafaf9',
        font=dict(family='Pretendard, sans-serif', size=11),
        bargap=0.5,
    )
    return fig

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='display:flex;align-items:center;justify-content:space-between;"
    "padding:1.2rem 0 1rem;border-bottom:1px solid #e8e8e3;margin-bottom:0.5rem;'>"
    "<div style='display:flex;align-items:center;gap:10px;'>"
    "<span style='width:30px;height:30px;border-radius:7px;background:#0f0f0e;"
    "color:#fafaf9;display:inline-flex;align-items:center;justify-content:center;"
    "font-family:var(--brand);font-size:18px;font-weight:700;line-height:1;'>V</span>"
    "<span style='font-family:var(--brand);font-size:24px;font-weight:700;"
    "letter-spacing:0px;color:#0f0f0e;'>Vouch</span>"
    "</div>"
    "<span style='font-size:0.75rem;color:#adadA6;'>KAIST BIZ · 2026</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ── 메인 탭 ───────────────────────────────────────────────────────────────────
tab_about, tab_match, tab_explore, tab_dashboard = st.tabs([
    "About", "🎯 브랜드 매칭", "(베타) 크리에이터 탐색", "📊 성과 대시보드"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: 브랜드 매칭
# ════════════════════════════════════════════════════════════════════════════
with tab_match:

    st.markdown(
        "<div style='text-align:center;padding:2.5rem 0 0;'>"
        "<div style='font-size:0.75rem;font-weight:600;color:#76766f;letter-spacing:.08em;"
        "text-transform:uppercase;margin-bottom:1rem;font-family:Pretendard,sans-serif;'>"
        "리스크까지 측정하는 크리에이터 매칭</div>"
        "<div style='font-size:2.6rem;font-weight:700;line-height:1.1;"
        "letter-spacing:-1px;margin-bottom:0.8rem;font-family:Pretendard,sans-serif;color:#0f0f0e;'>"
        "어떤 크리에이터를<br>찾고 계신가요?</div>"
        "<div style='font-size:1rem;color:#76766f;line-height:1.75;margin-bottom:2rem;'>"
        "브랜드와 캠페인을 자유롭게 설명해 주세요.<br>"
        "Vouch가 성실함부터 팔로워 진정성까지 검증해 추천합니다.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if 'brief_input' not in st.session_state:
        st.session_state['brief_input'] = ''
    # 칩 클릭으로 주입된 텍스트가 있으면 위젯 렌더 전에 적용
    if '_chip_pending' in st.session_state:
        st.session_state['brief_input'] = st.session_state.pop('_chip_pending')

    brand_text = st.text_area(
        label="브랜드 소개",
        label_visibility="collapsed",
        placeholder=(
            "예) 저희는 2030 여성을 타깃으로 하는 비건 스킨케어 브랜드입니다. "
            "신제품 세럼 런칭을 위해 진정성 있고 꾸준히 활동하는 뷰티 크리에이터를 찾고 있어요. "
            "마감 약속을 잘 지키는 분이 특히 중요하고, 팔로워가 실제 구매로 이어질 수 있는 분이면 좋겠습니다."
        ),
        height=140,
        key='brief_input',
    )

    col_o1, col_o2, col_o3 = st.columns([3, 3, 4])
    with col_o1:
        risk_threshold = st.selectbox(
            "최소 Risk Score",
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            index=3,
            help=(
                "**콘텐츠 신뢰도 · 브랜드 안전성 지수**\n\n"
                "크리에이터의 과거 협업 이력, 논란 여부, 광고 콘텐츠 비율 등을 종합한 점수입니다.\n\n"
                "- **4.0 ~ 5.0** — 우수: 브랜드 안전, 적극 추천\n"
                "- **3.0 ~ 4.0** — 양호: 대부분 안전, 검토 권장\n"
                "- **2.5 ~ 3.0** — 주의: 선별 필요, 면밀한 검토 요망\n"
                "- **2.5 미만** — 자동 제외"
            ),
        )
    with col_o2:
        top_n = st.selectbox(
            "추천 인원",
            list(range(1, 11)),
            index=2,
            help="추천받을 크리에이터 수를 선택하세요 (1~10명).",
        )
    with col_o3:
        st.write("")
        run = st.button("크리에이터 추천받기 →", type="primary", use_container_width=True)

    # 실제 데이터 기반 예시 칩 (헬스케어·게임·패션 업종이 성공 협업 사례 가장 많음)
    CHIP_EXAMPLES = [
        ("헬스케어 브랜드 — 피트니스 크리에이터",
         "저희는 3554 남녀를 타깃으로 하는 헬스케어 브랜드입니다. 영양제 신제품 출시를 앞두고 피트니스·건강 콘텐츠를 꾸준히 올리는 크리에이터를 찾고 있어요. 팔로워 진정성이 높고 신뢰도 있는 분을 우선합니다."),
        ("게임 주변기기 브랜드 — e스포츠 채널",
         "게임 주변기기 브랜드로, 1324 남성 유튜브 시청자에게 신제품을 알리고 싶습니다. 게이밍 리뷰·e스포츠 관련 콘텐츠를 제작하는 크리에이터와 장기 파트너십을 원합니다."),
        ("패션 브랜드 — 2030 여성 타깃",
         "2030 여성을 위한 패션 브랜드입니다. 의류와 스타일 코디 콘텐츠를 인스타그램 또는 유튜브에서 활발히 운영하는 분을 찾습니다. 약속 이행과 커뮤니케이션을 중시합니다."),
    ]
    chip_c1, chip_c2, chip_c3 = st.columns(3)
    for chip_col, (chip_label, chip_text) in zip([chip_c1, chip_c2, chip_c3], CHIP_EXAMPLES):
        if chip_col.button(chip_label, key=f"chip_{chip_label}", use_container_width=True):
            st.session_state['_chip_pending']       = chip_text
            st.session_state['last_brand_text']     = chip_text
            st.session_state['last_risk_threshold'] = risk_threshold
            st.session_state['last_top_n']          = top_n
            st.rerun()

    # 추천 결과
    if run or 'last_brand_text' in st.session_state:
        if run:
            st.session_state['last_brand_text']      = brand_text
            st.session_state['last_risk_threshold']  = risk_threshold
            st.session_state['last_top_n']           = top_n

        _text          = st.session_state['last_brand_text']
        risk_threshold = st.session_state['last_risk_threshold']
        top_n          = st.session_state['last_top_n']

        if not _text.strip():
            st.warning("브랜드 소개를 입력해 주세요.")
            st.stop()

        with st.spinner("브리프를 분석하고 크리에이터를 추천하는 중..."):
            brand_attrs = parse_brand_text(_text)
            top_df      = recommend_from_text(brand_attrs, creators, risk_threshold, top_n)

        # 파싱 결과 요약 카드
        _brief_preview = _text[:80] + '...' if len(_text) > 80 else _text
        st.markdown(
            "<div style='background:#fafaf9;border:1px solid #e8e8e3;border-radius:12px;"
            "padding:1rem 1.2rem;margin:0.8rem 0 0.3rem;'>"
            "<div style='font-size:0.72rem;font-weight:600;color:#76766f;letter-spacing:.08em;"
            "text-transform:uppercase;margin-bottom:0.5rem;'>브리프 분석 결과</div>"
            f"<div style='font-size:0.88rem;color:#3a3a38;margin-bottom:0.7rem;line-height:1.6;'>\"{_brief_preview}\"</div>"
            "<div style='display:flex;gap:6px;flex-wrap:wrap;'>"
            f"<span style='font-size:0.75rem;font-weight:600;color:#15803d;background:#e4f4e7;"
            f"border:1px solid #bbe3c4;border-radius:999px;padding:3px 10px;'>업종 {brand_attrs['Industry']}</span>"
            f"<span style='font-size:0.75rem;font-weight:600;color:#3a3a38;background:#fafaf9;"
            f"border:1px solid #e8e8e3;border-radius:999px;padding:3px 10px;'>타깃 {brand_attrs['Target_Age']} / {brand_attrs['Target_Gender']}</span>"
            f"<span style='font-size:0.75rem;font-weight:600;color:#3a3a38;background:#fafaf9;"
            f"border:1px solid #e8e8e3;border-radius:999px;padding:3px 10px;'>플랫폼 {brand_attrs['Preferred_Platform']}</span>"
            f"<span style='font-size:0.75rem;font-weight:600;color:#3a3a38;background:#fafaf9;"
            f"border:1px solid #e8e8e3;border-radius:999px;padding:3px 10px;'>Max CPM {brand_attrs['Max_CPM']:,.0f}원</span>"
            "</div>"
            "<div style='margin-top:0.75rem;padding-top:0.6rem;border-top:1px solid #e8e8e3;"
            "font-size:0.75rem;color:#76766f;line-height:1.7;'>"
            "<span style='font-weight:600;color:#3a3a38;'>참고 — 매칭 등급 기준</span>&nbsp;&nbsp;"
            "매칭 점수 = 카테고리(CBF) × 0.5 + 조건 매칭(CBF) × 0.5&nbsp;&nbsp;|&nbsp;&nbsp;"
            "<span style='color:#15803d;font-weight:600;'>A 0.9+</span> 강력 추천&nbsp;"
            "<span style='color:#2433ff;font-weight:600;'>B 0.8~0.9</span> 추천&nbsp;"
            "<span style='color:#9a6207;font-weight:600;'>C 0.7~0.8</span> 보통&nbsp;"
            "<span style='color:#c42626;font-weight:600;'>D 0.7-</span> 참고용"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        brand_row = brand_attrs  # 하위 코드에서 brand_row['Industry'] 등 그대로 사용

        st.markdown("<div class='section-title'>추천 결과</div>", unsafe_allow_html=True)

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
                        n_cols = len(chunk)

                        # ── 1st pass: 카드 (같은 높이 유지)
                        card_cols = st.columns(n_cols)
                        for col, (_, row) in zip(card_cols, chunk):
                            grade  = row.get('recommendation_grade', grade_label(row['matching_score']))
                            color  = GRADE_COLOR[grade]
                            bg     = GRADE_BG[grade]
                            border = GRADE_BORDER[grade]
                            rank_n     = int(row['Rank'])
                            medal_icon = RANK_MEDAL.get(rank_n, "")
                            medal = (
                                f"<span style='font-size:0.85rem;font-weight:700;color:#76766f;'>"
                                f"{rank_n}위{'&nbsp;' + medal_icon if medal_icon else ''}</span>"
                            )
                            pos_reasons, neg_reasons = build_reasons(row, brand_row)
                            tags_html = reason_tags_html(pos_reasons, neg_reasons)
                            c_id       = row['Creator_ID']
                            n_collab   = collab_count.get(c_id, 0)
                            follow_pct = min(int(row['Followers'] / max_followers * 100), 100)
                            score_pct  = int(row['matching_score'] * 100)
                            card_html = (
                                f"<div class='creator-card' style='border-color:{border};'>"
                                f"<div style='display:flex;justify-content:space-between;"
                                f"align-items:center;margin-bottom:0.75rem;'>"
                                f"{medal}"
                                f"<span style='background:{bg};color:{color};border:1px solid {border};"
                                f"border-radius:999px;padding:0.2rem 0.7rem;"
                                f"font-size:0.72rem;font-weight:600;'>{GRADE_LABEL[grade]}</span>"
                                f"</div>"
                                f"<div style='font-size:1.05rem;font-weight:700;color:#0f0f0e;"
                                f"margin-bottom:0.2rem;letter-spacing:-0.3px;'>{row['Channel_Name']}</div>"
                                f"<div style='font-size:0.8rem;color:#76766f;margin-bottom:1rem;'>"
                                f"{row['Platform']} · {row['Category']}</div>"
                                f"<div style='margin-bottom:0.9rem;'>"
                                f"<div style='display:flex;justify-content:space-between;"
                                f"font-size:0.75rem;color:#76766f;margin-bottom:0.35rem;'>"
                                f"<span>매칭 점수</span>"
                                f"<span style='font-weight:700;color:{color};'>{row['matching_score']:.2f}</span></div>"
                                f"<div style='background:#e8e8e3;border-radius:999px;height:4px;'>"
                                f"<div style='background:{color};height:4px;border-radius:999px;width:{score_pct}%;'></div></div></div>"
                                f"<div style='margin-bottom:0.9rem;'>"
                                f"<div style='display:flex;justify-content:space-between;"
                                f"font-size:0.75rem;color:#76766f;margin-bottom:0.35rem;'>"
                                f"<span>구독자</span>"
                                f"<span style='font-weight:600;color:#0f0f0e;'>{fmt_followers(row['Followers'])}</span></div>"
                                f"<div style='background:#e8e8e3;border-radius:999px;height:3px;'>"
                                f"<div style='background:#adadA6;height:3px;border-radius:999px;"
                                f"width:{follow_pct}%;'></div></div></div>"
                                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                                f"gap:0.5rem;margin-bottom:0.9rem;'>"
                                f"<div style='background:#fafaf9;border:1px solid #e8e8e3;border-radius:10px;"
                                f"padding:0.45rem 0.5rem;text-align:center;'>"
                                f"<div style='color:#76766f;font-size:0.68rem;margin-bottom:2px;'>참여율</div>"
                                f"<div style='font-weight:700;font-size:0.9rem;color:#0f0f0e;'>{row['Engagement_Rate']}%</div></div>"
                                f"<div style='background:#fafaf9;border:1px solid #e8e8e3;border-radius:10px;"
                                f"padding:0.45rem 0.5rem;text-align:center;'>"
                                f"<div style='color:#76766f;font-size:0.68rem;margin-bottom:2px;'>협업</div>"
                                f"<div style='font-weight:700;font-size:0.9rem;color:#0f0f0e;'>{n_collab}회</div></div>"
                                f"<div style='background:#fafaf9;border:1px solid #e8e8e3;border-radius:10px;"
                                f"padding:0.45rem 0.5rem;text-align:center;'>"
                                f"<div style='color:#76766f;font-size:0.68rem;margin-bottom:2px;'>Risk</div>"
                                f"<div style='font-weight:700;font-size:0.9rem;color:#0f0f0e;'>{row['Risk_Score']}</div></div></div>"
                                f"<div>{tags_html}</div>"
                                f"</div>"
                            )
                            col.markdown(card_html, unsafe_allow_html=True)

                        # ── 2nd pass: 상세 분석 expander (카드와 분리해서 높이 영향 없음)
                        exp_cols = st.columns(n_cols)
                        for exp_col, (_, row) in zip(exp_cols, chunk):
                            c_id = row['Creator_ID']
                            with exp_col:
                                with st.expander("📊 상세 분석"):
                                    st.plotly_chart(plotly_score_bar(row),
                                                    use_container_width=True,
                                                    config={'displayModeBar': False},
                                                    key=f"score_bar_text_{cat_label}_{c_id}")
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
            st.markdown("<div class='section-title'>매칭 점수 분포</div>",
                        unsafe_allow_html=True)
            with st.container():
                all_scores_df = recommend_from_text(brand_attrs, creators, risk_threshold=0.0, top_n=len(creators))

                bins   = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                labels = ['~0.4', '0.4~0.5', '0.5~0.6', '0.6~0.7',
                          '0.7~0.8', '0.8~0.9', '0.9~']
                bin_colors = ['#e8e8e3','#d6d6d0','#c0c8ff','#a8b0ff',
                              '#efd9a8','#bbe3c4','#15803d']
                hist_data = pd.cut(all_scores_df['matching_score'],
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
                    plot_bgcolor='#fafaf9', paper_bgcolor='#fafaf9',
                    yaxis=dict(range=[0, y_max], showgrid=True,
                               gridcolor='#e8e8e3', tickfont=dict(size=11, color='#76766f')),
                    xaxis=dict(showgrid=False, tickfont=dict(size=11, color='#76766f')),
                    font=dict(family='Pretendard, sans-serif', size=12),
                    bargap=0.3,
                )

                col_chart, col_info = st.columns([2, 1])
                with col_chart:
                    st.plotly_chart(fig_hist, use_container_width=True,
                                    config={'displayModeBar': False})
                with col_info:
                    st.markdown("**등급별 현황**")
                    total = len(all_scores_df)
                    grade_ranges = [("A", 0.9, 1.1), ("B", 0.8, 0.9),
                                    ("C", 0.7, 0.8), ("D", 0.0, 0.7)]
                    for g, lo, hi in grade_ranges:
                        cnt = ((all_scores_df['matching_score'] >= lo) &
                               (all_scores_df['matching_score'] < hi)).sum()
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
            st.markdown("<div class='section-title'>유사 협업 사례</div>",
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
                conn, params=[brand_attrs['Industry']] + top_creator_ids,
            )
            conn.close()

            if cases.empty:
                st.markdown(
                    "<div style='text-align:center;padding:2rem 1rem;border:1px dashed #e8e8e3;"
                    "border-radius:14px;margin:0.5rem 0;'>"
                    "<div style='font-size:1.8rem;margin-bottom:0.5rem;'>🔍</div>"
                    "<div style='font-size:0.95rem;font-weight:600;color:#0f0f0e;margin-bottom:0.3rem;'>"
                    "유사 사례를 찾지 못했어요</div>"
                    "<div style='font-size:0.82rem;color:#76766f;line-height:1.6;'>"
                    f"{brand_attrs['Industry']} 업종의 성공 협업 데이터가 아직 충분하지 않습니다.<br>"
                    "브리프에 다른 키워드를 추가하거나 업종을 바꿔 검색해보세요.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                with st.container():
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



# ════════════════════════════════════════════════════════════════════════════
# TAB 2: 크리에이터 탐색
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown(
        "<div style='background:#fbf0d9;border:1px solid #efd9a8;border-radius:12px;"
        "padding:0.9rem 1.2rem;margin:1.2rem 0 1.5rem;display:flex;align-items:center;gap:0.8rem;'>"
        "<span style='font-size:1rem;'>🔬</span>"
        "<div>"
        "<span style='font-size:0.8rem;font-weight:700;color:#9a6207;'>베타 기능 · 데모 예시</span>"
        "<span style='font-size:0.8rem;color:#9a6207;margin-left:0.6rem;'>"
        "현재 개발 중인 기능으로, 아래 내용은 실제 데이터 기반 예시입니다.</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-title'>크리에이터 탐색</div>", unsafe_allow_html=True)
    st.caption("크리에이터 관점에서 협업 가능성이 높은 브랜드를 역방향으로 조회합니다.")

    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin:1.2rem 0 0.6rem;'>"
        "<span style='width:22px;height:22px;border-radius:50%;background:#0f0f0e;color:#fafaf9;"
        "display:inline-flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;flex-shrink:0;'>1</span>"
        "<span style='font-size:0.9rem;font-weight:600;'>필터 설정</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container():
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
            min_risk = st.slider(
                "최소 Risk Score", 1.0, 5.0, 2.5, 0.5,
                key="explore_min_risk",
                help=(
                    "**콘텐츠 신뢰도 · 브랜드 안전성 지수**  \n"
                    "4.0 ~ 5.0 — 우수 (브랜드 안전)  \n"
                    "3.0 ~ 4.0 — 보통 (검토 권장)  \n"
                    "2.5 ~ 3.0 — 주의 (선별 필요)  \n"
                    "2.5 미만 — 자동 제외"
                ),
            )

    fc = creators.copy()
    if cat_filter  != "전체": fc = fc[fc['Category'] == cat_filter]
    if plat_filter != "전체": fc = fc[fc['Platform']  == plat_filter]
    fc = fc[fc['Risk_Score'] >= min_risk]

    if fc.empty:
        st.warning("조건에 맞는 크리에이터가 없습니다.")
    else:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;margin:1.5rem 0 0.6rem;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#0f0f0e;color:#fafaf9;"
            "display:inline-flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;flex-shrink:0;'>2</span>"
            "<span style='font-size:0.9rem;font-weight:600;'>크리에이터 선택</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        sel_creator = st.selectbox("크리에이터 선택", fc['Channel_Name'].tolist(),
                                   key="explore_creator_select")
        sel_cid_exp = fc[fc['Channel_Name'] == sel_creator].iloc[0]['Creator_ID']
        ci          = fc[fc['Creator_ID'] == sel_cid_exp].iloc[0]

        # 프로필 카드
        n_c = collab_count.get(sel_cid_exp, 0)
        n_s = collab_success.get(sel_cid_exp, 0)
        succ_rate = int(n_s / n_c * 100) if n_c > 0 else 0
        metrics = [
            ("플랫폼",     ci['Platform']),
            ("카테고리",   ci['Category']),
            ("구독자",     fmt_followers(ci['Followers'])),
            ("참여율",     f"{ci['Engagement_Rate']}%"),
            ("Risk Score", f"{ci['Risk_Score']}"),
        ]
        metric_items = "".join(
            f"<div style='flex:1;text-align:center;'>"
            f"<div style='font-size:0.75rem;color:#888;margin-bottom:0.2rem;'>{label}</div>"
            f"<div style='font-size:1.3rem;font-weight:700;color:#1a3a5c;letter-spacing:-0.5px;'>{val}</div>"
            f"</div>"
            for label, val in metrics
        )
        st.markdown(
            f"<div style='background:#f8fafc;border-radius:10px;padding:1rem 1.2rem 0.8rem;"
            f"margin:0.5rem 0 1rem;'>"
            f"<div style='display:flex;gap:0.5rem;align-items:center;'>"
            f"{metric_items}"
            f"</div>"
            f"<div style='font-size:0.85rem;color:#555;margin-top:0.7rem;border-top:1px solid #e8edf2;"
            f"padding-top:0.5rem;'>"
            f"협업 이력 <b>{n_c}회</b> &nbsp;|&nbsp; 성공 <b>{n_s}회</b> ({succ_rate}%)"
            f"</div>"
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

        cs['매칭점수'] = cs['matching_score'].map(lambda v: f"{v:.3f}")
        col_table, col_bar = st.columns([1, 1])
        with col_table:
            st.dataframe(
                cs[['순위', '브랜드', '업종', '월예산', '매칭점수', '등급']],
                use_container_width=True, hide_index=True,
                column_config={
                    '순위':   st.column_config.NumberColumn(width="small"),
                    '브랜드': st.column_config.TextColumn(width="medium"),
                    '업종':   st.column_config.TextColumn(width="small"),
                    '월예산': st.column_config.NumberColumn(width="small", format="%d"),
                    '매칭점수': st.column_config.TextColumn(width="small"),
                    '등급':   st.column_config.TextColumn(width="small"),
                }
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
                font=dict(family='Pretendard, sans-serif', size=11),
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
        (k1, f"{total_collabs:,}건", "총 협업 수",  "#0f0f0e"),
        (k2, f"{success_rate:.1f}%", "성공률",      "#15803d"),
        (k3, f"{avg_ctr:.2f}%",      "평균 CTR",   "#2433ff"),
        (k4, f"{avg_cvr:.2f}%",      "평균 CVR",   "#9a6207"),
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
            font=dict(family='Pretendard, sans-serif', size=12),
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
            font=dict(family='Pretendard, sans-serif', size=12),
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
            font=dict(family='Pretendard, sans-serif', size=12),
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
            font=dict(family='Pretendard, sans-serif', size=12),
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


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: About
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(
        "<div style='max-width:680px;margin:3rem auto 0;padding:0 1rem;'>"

        # 로고 + 이름
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:2rem;'>"
        "<span style='width:36px;height:36px;border-radius:9px;background:#0f0f0e;"
        "color:#fafaf9;display:inline-flex;align-items:center;justify-content:center;"
        "font-size:22px;font-weight:700;font-family:Pretendard,sans-serif;'>V</span>"
        "<span style='font-size:1.8rem;font-weight:700;letter-spacing:-0.5px;'>Vouch</span>"
        "</div>"

        # Vouch의 의미
        "<div style='margin-bottom:2rem;'>"
        "<div style='font-size:0.72rem;font-weight:600;color:#76766f;letter-spacing:.1em;"
        "text-transform:uppercase;margin-bottom:0.6rem;'>이름의 의미</div>"
        "<div style='font-size:1.1rem;font-weight:700;color:#0f0f0e;margin-bottom:0.5rem;'>"
        "\"Vouch\" — 보증하다, 책임지고 추천하다</div>"
        "<div style='font-size:0.92rem;color:#3a3a38;line-height:1.8;'>"
        "Vouch는 <b>보증(vouch for)</b>에서 따온 이름입니다. "
        "단순히 팔로워 수가 많은 크리에이터가 아니라, "
        "성실함·커뮤니케이션·약속 이행·팔로워 진정성을 데이터로 검증한 뒤 "
        "브랜드에 <b>책임지고 추천</b>한다는 의미를 담았습니다."
        "</div>"
        "</div>"

        # 만들게 된 계기
        "<div style='margin-bottom:2rem;'>"
        "<div style='font-size:0.72rem;font-weight:600;color:#76766f;letter-spacing:.1em;"
        "text-transform:uppercase;margin-bottom:0.6rem;'>만들게 된 계기</div>"
        "<div style='font-size:0.92rem;color:#3a3a38;line-height:1.8;'>"
        "인플루언서 마케팅 시장이 빠르게 성장하고 있지만, "
        "브랜드 담당자들은 여전히 <b>크리에이터의 신뢰도를 검증할 마땅한 방법</b>이 없었습니다. "
        "허위 팔로워, 잦은 마감 지연, 소통 단절 — 실제 협업 현장에서 반복되는 문제들을 "
        "데이터로 해결하고자 이 프로젝트를 시작했습니다."
        "</div>"
        "</div>"

        # 어떻게 작동하는가
        "<div style='margin-bottom:2rem;'>"
        "<div style='font-size:0.72rem;font-weight:600;color:#76766f;letter-spacing:.1em;"
        "text-transform:uppercase;margin-bottom:0.6rem;'>어떻게 작동하나요</div>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;'>"
        "<div style='background:#ffffff;border:1px solid #e8e8e3;border-radius:12px;padding:1rem 1.1rem;'>"
        "<div style='font-size:0.85rem;font-weight:700;margin-bottom:0.3rem;'>브리프 입력</div>"
        "<div style='font-size:0.8rem;color:#76766f;line-height:1.6;'>브랜드와 캠페인을 자유롭게 설명하면 자동으로 조건을 분석합니다.</div>"
        "</div>"
        "<div style='background:#ffffff;border:1px solid #e8e8e3;border-radius:12px;padding:1rem 1.1rem;'>"
        "<div style='font-size:0.85rem;font-weight:700;margin-bottom:0.3rem;'>데이터 검증</div>"
        "<div style='font-size:0.8rem;color:#76766f;line-height:1.6;'>490명의 크리에이터를 CBF + CF 하이브리드 모델로 스코어링합니다.</div>"
        "</div>"
        "<div style='background:#ffffff;border:1px solid #e8e8e3;border-radius:12px;padding:1rem 1.1rem;'>"
        "<div style='font-size:0.85rem;font-weight:700;margin-bottom:0.3rem;'>리스크 분석</div>"
        "<div style='font-size:0.8rem;color:#76766f;line-height:1.6;'>성실함·커뮤니케이션·약속 이행·팔로워 진정성 4개 축으로 평가합니다.</div>"
        "</div>"
        "<div style='background:#ffffff;border:1px solid #e8e8e3;border-radius:12px;padding:1rem 1.1rem;'>"
        "<div style='font-size:0.85rem;font-weight:700;margin-bottom:0.3rem;'>최적 매칭</div>"
        "<div style='font-size:0.8rem;color:#76766f;line-height:1.6;'>976건의 실제 협업 데이터를 기반으로 최적의 파트너를 추천합니다.</div>"
        "</div>"
        "</div>"
        "</div>"

        # 팀 정보
        "<div style='border-top:1px solid #e8e8e3;padding-top:1.5rem;margin-bottom:3rem;'>"
        "<div style='font-size:0.72rem;font-weight:600;color:#76766f;letter-spacing:.1em;"
        "text-transform:uppercase;margin-bottom:0.6rem;'>팀 정보</div>"
        "<div style='font-size:0.88rem;color:#3a3a38;line-height:1.8;'>"
        "KAIST 경영대학 &nbsp;·&nbsp; Business Analytics 2026"
        "</div>"
        "</div>"

        "</div>",
        unsafe_allow_html=True,
    )

# ── 푸터 ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#bbb; font-size:0.78rem;'>"
    "KAIST BIZ &nbsp;|&nbsp; 비즈니스 애널리틱스 2026 &nbsp;|&nbsp; "
    "CBF + CF Hybrid Recommendation System</p>",
    unsafe_allow_html=True
)
