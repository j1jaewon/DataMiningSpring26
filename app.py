import streamlit as st
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="기업-크리에이터 매칭 추천 시스템", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    creators = pd.read_csv(os.path.join(BASE_DIR, 'creators_clean.csv'))
    brands   = pd.read_csv(os.path.join(BASE_DIR, 'brands_100.csv'))
    collabs  = pd.read_csv(os.path.join(BASE_DIR, 'collaborations_final.csv'))
    ratings  = pd.read_csv(os.path.join(BASE_DIR, 'ratings_clean.csv'))
    sim_path = os.path.join(BASE_DIR, 'creator_similarity.csv')
    if os.path.exists(sim_path):
        similarity = pd.read_csv(sim_path)
    else:
        similarity = None
    return creators, brands, collabs, ratings, similarity

@st.cache_data
def build_similarity(creators, brands, ratings):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    category_map = {
        '뷰티': ['뷰티'], '패션': ['패션', '라이프스타일'],
        '식품': ['푸드'], '테크': ['테크', '게임'],
        '생활용품': ['라이프스타일'], '피트니스': ['피트니스', '라이프스타일'],
        '교육': ['교육'], '여행': ['여행', '라이프스타일'],
        '게임': ['게임', '테크'], '헬스케어': ['라이프스타일', '피트니스'],
    }
    similar_categories = {
        ('뷰티', '패션'): 0.5, ('패션', '뷰티'): 0.5,
        ('테크', '게임'): 0.5, ('게임', '테크'): 0.5,
        ('라이프스타일', '여행'): 0.5, ('여행', '라이프스타일'): 0.5,
        ('푸드', '라이프스타일'): 0.3, ('라이프스타일', '푸드'): 0.3,
        ('피트니스', '라이프스타일'): 0.5, ('라이프스타일', '피트니스'): 0.5,
    }

    def calc_category_score(brand_industry, creator_category):
        direct = category_map.get(brand_industry, [])
        if creator_category in direct:
            return 1.0
        return similar_categories.get((creator_category, brand_industry), 0.0)

    def calc_context_score(brand, creator):
        score = 0.0
        if brand['Target_Age'] == creator['Target_Age']:
            score += 0.25
        if brand['Target_Gender'] == creator['Target_Gender'] or \
           creator['Target_Gender'] == 'Mixed' or brand['Target_Gender'] == 'Mixed':
            score += 0.25
        if creator['Estimated_CPM'] <= brand['Max_CPM']:
            score += 0.25
        if creator['Risk_Score'] >= 3.0:
            score += 0.25
        return score

    rating_matrix = ratings.pivot_table(
        index='Brand_ID', columns='Creator_ID', values='Score', aggfunc='mean'
    ).fillna(0)
    brand_similarity_mat = pd.DataFrame(
        cosine_similarity(rating_matrix),
        index=rating_matrix.index, columns=rating_matrix.index
    )

    def calc_cf_score(brand_id, creator_id):
        if brand_id not in rating_matrix.index or creator_id not in rating_matrix.columns:
            return 0.0
        sim_scores = brand_similarity_mat[brand_id].drop(brand_id)
        top_similar = sim_scores.nlargest(5)
        numerator = denominator = 0
        for other_brand, sim in top_similar.items():
            if sim > 0 and rating_matrix.loc[other_brand, creator_id] > 0:
                numerator += sim * rating_matrix.loc[other_brand, creator_id]
                denominator += sim
        if denominator == 0:
            return 0.0
        return round(numerator / denominator / 5.0, 4)

    results = []
    for b in brands.to_dict('records'):
        for c in creators.to_dict('records'):
            cat_score = calc_category_score(b['Industry'], c['Category'])
            if cat_score == 0:
                continue
            ctx_score = calc_context_score(b, c)
            cf_score  = calc_cf_score(b['Brand_ID'], c['Creator_ID'])
            matching  = cat_score*0.3 + ctx_score*0.3 + cf_score*0.4 if cf_score > 0 \
                        else cat_score*0.5 + ctx_score*0.5
            results.append({
                'Brand_ID': b['Brand_ID'], 'Creator_ID': c['Creator_ID'],
                'category_score': round(cat_score, 4),
                'context_score':  round(ctx_score, 4),
                'cf_score':       round(cf_score, 4),
                'matching_score': round(matching, 4),
            })
    return pd.DataFrame(results)

def recommend(brand_id, similarity_df, creators, risk_threshold=2.5, top_n=3):
    result = similarity_df[similarity_df['Brand_ID'] == brand_id].copy()
    risk_map    = dict(zip(creators['Creator_ID'], creators['Risk_Score']))
    name_map    = dict(zip(creators['Creator_ID'], creators['Channel_Name']))
    cat_map     = dict(zip(creators['Creator_ID'], creators['Category']))
    plat_map    = dict(zip(creators['Creator_ID'], creators['Platform']))
    follow_map  = dict(zip(creators['Creator_ID'], creators['Followers']))
    er_map      = dict(zip(creators['Creator_ID'], creators['Engagement_Rate']))

    result['Risk_Score'] = result['Creator_ID'].map(risk_map)
    result = result[result['Risk_Score'] >= risk_threshold]
    top = result.nlargest(top_n, 'matching_score').copy()
    top['Channel_Name']    = top['Creator_ID'].map(name_map)
    top['Category']        = top['Creator_ID'].map(cat_map)
    top['Platform']        = top['Creator_ID'].map(plat_map)
    top['Followers']       = top['Creator_ID'].map(follow_map)
    top['Engagement_Rate'] = top['Creator_ID'].map(er_map)
    top['Rank']            = range(1, len(top) + 1)
    return top.reset_index(drop=True)

def get_similar_campaigns(brand_id, top_creator_ids, collabs, creators, brands):
    brand_info = brands[brands['Brand_ID'] == brand_id].iloc[0]
    brand_industry = brand_info['Industry']

    same_industry_brands = brands[brands['Industry'] == brand_industry]['Brand_ID'].tolist()
    cases = collabs[
        (collabs['Brand_ID'].isin(same_industry_brands)) &
        (collabs['Creator_ID'].isin(top_creator_ids))
    ].copy()

    name_map  = dict(zip(creators['Creator_ID'], creators['Channel_Name']))
    brand_map = dict(zip(brands['Brand_ID'], brands['Brand_Name']))
    cases['Creator_Name'] = cases['Creator_ID'].map(name_map)
    cases['Brand_Name']   = cases['Brand_ID'].map(brand_map)
    return cases.head(5)

def grade_label(score):
    if score >= 0.8:
        return "A"
    elif score >= 0.6:
        return "B"
    elif score >= 0.4:
        return "C"
    else:
        return "D"

# ── 앱 시작 ──────────────────────────────────────────────────────────────────
creators, brands, collabs, ratings, similarity_df = load_data()

if similarity_df is None:
    with st.spinner("추천 점수를 계산 중입니다... (최초 1회)"):
        similarity_df = build_similarity(creators, brands, ratings)

# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
            padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0; font-size: 1.8rem;'>기업-크리에이터 매칭 추천 시스템</h1>
    <p style='color: #a8c8e8; margin: 0.3rem 0 0; font-size: 0.95rem;'>
        KAIST BIZ | 비즈니스 애널리틱스 | CBF + CF 하이브리드 추천
    </p>
</div>
""", unsafe_allow_html=True)

# ── ① 브랜드 조건 입력 ────────────────────────────────────────────────────────
st.subheader("① 브랜드 조건 입력")

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        brand_options = brands[['Brand_ID', 'Brand_Name', 'Industry']].copy()
        brand_display = brand_options.apply(
            lambda r: f"{r['Brand_Name']} ({r['Industry']})", axis=1
        ).tolist()
        selected_idx = st.selectbox("브랜드 선택", range(len(brand_display)),
                                     format_func=lambda i: brand_display[i])
        selected_brand = brand_options.iloc[selected_idx]
        brand_id = selected_brand['Brand_ID']

    with col2:
        risk_threshold = st.slider("최소 Risk Score 기준", 1.0, 5.0, 2.5, 0.5,
                                    help="이 값 미만의 크리에이터는 추천에서 제외됩니다")

    with col3:
        top_n = st.slider("추천 크리에이터 수", 1, 10, 3)

    brand_row = brands[brands['Brand_ID'] == brand_id].iloc[0]
    st.markdown(f"""
    <div style='background:#f0f4f8; border-radius:8px; padding:0.8rem 1rem; margin-top:0.5rem;
                display:flex; gap:2rem; flex-wrap:wrap; font-size:0.88rem; color:#444;'>
        <span>🏢 <b>{brand_row['Brand_Name']}</b></span>
        <span>🏷️ {brand_row['Industry']}</span>
        <span>💰 월 예산 {brand_row['Monthly_Budget']:,}원</span>
        <span>🎯 타겟 {brand_row['Target_Age']} / {brand_row['Target_Gender']}</span>
        <span>📱 선호 플랫폼 {brand_row['Preferred_Platform']}</span>
    </div>
    """, unsafe_allow_html=True)

    run = st.button("추천 받기", type="primary", use_container_width=True)

# ── ② Top N 추천 결과 ─────────────────────────────────────────────────────────
if run or 'last_brand_id' in st.session_state:
    if run:
        st.session_state['last_brand_id']        = brand_id
        st.session_state['last_risk_threshold']  = risk_threshold
        st.session_state['last_top_n']           = top_n

    brand_id       = st.session_state['last_brand_id']
    risk_threshold = st.session_state['last_risk_threshold']
    top_n          = st.session_state['last_top_n']

    top_df = recommend(brand_id, similarity_df, creators, risk_threshold, top_n)

    st.subheader("② 추천 결과")

    if top_df.empty:
        st.warning("조건을 만족하는 크리에이터가 없습니다. Risk Score 기준을 낮춰보세요.")
    else:
        GRADE_COLOR = {"A": "#1a7a4a", "B": "#2d6a9f", "C": "#b07c00", "D": "#c0392b"}

        cols = st.columns(len(top_df))
        for col, (_, row) in zip(cols, top_df.iterrows()):
            grade = grade_label(row['matching_score'])
            color = GRADE_COLOR.get(grade, "#888")
            reasons = []
            if row['category_score'] >= 1.0:
                reasons.append("카테고리 일치")
            elif row['category_score'] > 0:
                reasons.append("카테고리 유사")
            if row['context_score'] >= 0.5:
                reasons.append("오디언스 적합")
            if row['Engagement_Rate'] >= 5.0:
                reasons.append("높은 참여율")
            if row['cf_score'] > 0:
                reasons.append("협업 이력 반영")

            followers = row['Followers']
            if followers >= 10000:
                followers_str = f"{followers/10000:.1f}만"
            else:
                followers_str = f"{followers:,}"

            with col:
                st.markdown(f"""
                <div style='border:1px solid #dde3ec; border-radius:10px; padding:1.2rem;
                            text-align:center; background:white; height:100%;'>
                    <div style='font-size:0.85rem; color:#888; margin-bottom:0.3rem;'>
                        {row['Rank']}위
                    </div>
                    <div style='font-size:1.1rem; font-weight:700; color:#1a3a5c;
                                margin-bottom:0.5rem;'>
                        {row['Channel_Name']}
                    </div>
                    <div style='display:inline-block; background:{color}; color:white;
                                border-radius:20px; padding:0.2rem 0.8rem;
                                font-size:1rem; font-weight:700; margin-bottom:0.8rem;'>
                        {row['matching_score']:.2f}점&nbsp;&nbsp;등급 {grade}
                    </div>
                    <div style='font-size:0.82rem; color:#555; line-height:1.8;
                                margin-bottom:0.8rem; text-align:left;'>
                        {"<br>".join(f"✔ {r}" for r in reasons) if reasons else ""}
                    </div>
                    <hr style='border:none; border-top:1px solid #eee; margin:0.5rem 0;'>
                    <div style='font-size:0.82rem; color:#555; text-align:left;
                                line-height:1.9;'>
                        📱 {row['Platform']}<br>
                        🏷️ {row['Category']}<br>
                        👥 구독자 {followers_str}<br>
                        💬 참여율 {row['Engagement_Rate']}%<br>
                        ⚠️ Risk {row['Risk_Score']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── ③ 유사 협업 사례 ──────────────────────────────────────────────────
        st.subheader("③ 유사 협업 사례")
        top_creator_ids = top_df['Creator_ID'].tolist()
        cases = get_similar_campaigns(brand_id, top_creator_ids, collabs, creators, brands)

        if cases.empty:
            st.info("동일 업종의 유사 협업 사례가 없습니다.")
        else:
            with st.container(border=True):
                for _, c in cases.iterrows():
                    success_icon = "✅" if c['is_success'] == 'Y' else "❌"
                    st.markdown(
                        f"{success_icon} **{c['Brand_Name']}** + **{c['Creator_Name']}** "
                        f"→ 예산 {c['Budget_Spent']:,}원 | "
                        f"노출 {c['Impressions']:,}회 | "
                        f"CTR {c['CTR']}% | "
                        f"CVR {c['CVR']}% | "
                        f"{'성공(Y)' if c['is_success']=='Y' else '실패(N)'}"
                    )

        # ── ④ 캠페인 성과 입력 ────────────────────────────────────────────────
        st.subheader("④ 캠페인 성과 입력")
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                creator_options = top_df.apply(
                    lambda r: f"{r['Rank']}위 {r['Channel_Name']}", axis=1
                ).tolist()
                selected_creator_label = st.selectbox("크리에이터 선택", creator_options)
                selected_creator_idx = int(selected_creator_label.split("위")[0]) - 1
                selected_creator_id = top_df.iloc[selected_creator_idx]['Creator_ID']

            with col2:
                impressions_input = st.number_input("실제 노출수", min_value=0, step=1000)
            with col3:
                ctr_input = st.number_input("CTR (%)", min_value=0.0, max_value=100.0,
                                             step=0.1, format="%.2f")
            with col4:
                success_input = st.selectbox("성공 여부", ["Y", "N"])

            if st.button("성과 저장", type="secondary", use_container_width=True):
                new_row = {
                    'Collab_ID':    f"CB_NEW_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                    'Brand_ID':     brand_id,
                    'Creator_ID':   selected_creator_id,
                    'Campaign_Start': pd.Timestamp.now().date(),
                    'Campaign_End':   pd.Timestamp.now().date(),
                    'Budget_Spent':   0,
                    'Impressions':    impressions_input,
                    'Clicks':         int(impressions_input * ctr_input / 100),
                    'CTR':            ctr_input,
                    'Conversions':    0,
                    'CVR':            0,
                    'is_success':     success_input,
                }
                save_path = os.path.join(BASE_DIR, 'collaborations_final.csv')
                updated = pd.concat([collabs, pd.DataFrame([new_row])], ignore_index=True)
                updated.to_csv(save_path, index=False, encoding='utf-8-sig')
                st.success(f"성과가 저장되었습니다! ({new_row['Collab_ID']})")
                st.cache_data.clear()

# ── 푸터 ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "KAIST BIZ | 비즈니스 애널리틱스 2026 | "
    "CBF + CF Hybrid Recommendation System</p>",
    unsafe_allow_html=True
)
