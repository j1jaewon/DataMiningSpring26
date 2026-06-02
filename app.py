import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from recommendation_logic import build_similarity, recommend, grade_label

st.set_page_config(page_title="기업-크리에이터 매칭 추천 시스템", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

GRADE_COLOR = {"A": "#1a7a4a", "B": "#2d6a9f", "C": "#b07c00", "D": "#c0392b"}
GRADE_BG    = {"A": "#e8f7ef", "B": "#e8f0fb", "C": "#fdf6e3", "D": "#fdecea"}

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    creators = pd.read_csv(os.path.join(DATA_DIR, 'creators_clean.csv'))
    brands   = pd.read_csv(os.path.join(DATA_DIR, 'brands_100.csv'))
    collabs  = pd.read_csv(os.path.join(DATA_DIR, 'collaborations_final.csv'))
    ratings  = pd.read_csv(os.path.join(DATA_DIR, 'ratings_clean.csv'))
    sim_path = os.path.join(DATA_DIR, 'creator_similarity.csv')
    similarity = pd.read_csv(sim_path) if os.path.exists(sim_path) else None
    return creators, brands, collabs, ratings, similarity

build_similarity_cached = st.cache_data(build_similarity)

creators, brands, collabs, ratings, similarity_df = load_data()

if similarity_df is None:
    with st.spinner("추천 점수를 계산 중입니다... (최초 1회)"):
        similarity_df = build_similarity_cached(creators, brands, ratings)

# ── 협업 이력 카운트 (크리에이터별) ──────────────────────────────────────────
collab_count = collabs.groupby('Creator_ID').size().to_dict()
collab_success = collabs[collabs['is_success'] == 'Y'].groupby('Creator_ID').size().to_dict()
max_followers  = creators['Followers'].max()

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

# ── 메인 탭 ──────────────────────────────────────────────────────────────────
tab_match, tab_explore, tab_dashboard = st.tabs([
    "🎯 브랜드 매칭", "🔍 크리에이터 탐색", "📊 성과 대시보드"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: 브랜드 매칭
# ════════════════════════════════════════════════════════════════════════════
with tab_match:

    # ── ① 브랜드 조건 입력 ────────────────────────────────────────────────
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

    # ── ② 추천 결과 ────────────────────────────────────────────────────────
    if run or 'last_brand_id' in st.session_state:
        if run:
            st.session_state['last_brand_id']       = brand_id
            st.session_state['last_risk_threshold'] = risk_threshold
            st.session_state['last_top_n']          = top_n

        brand_id       = st.session_state['last_brand_id']
        risk_threshold = st.session_state['last_risk_threshold']
        top_n          = st.session_state['last_top_n']
        brand_row      = brands[brands['Brand_ID'] == brand_id].iloc[0]

        top_df = recommend(brand_id, similarity_df, creators, risk_threshold, top_n)

        st.subheader("② 추천 결과")

        if top_df.empty:
            st.warning("조건을 만족하는 크리에이터가 없습니다. Risk Score 기준을 낮춰보세요.")
        else:
            # 카테고리 필터 탭 ──────────────────────────────────────────────
            all_cats = ["전체"] + sorted(top_df['Category'].unique().tolist())
            cat_tabs = st.tabs(all_cats)

            for cat_tab, cat_label in zip(cat_tabs, all_cats):
                with cat_tab:
                    filtered = top_df if cat_label == "전체" \
                               else top_df[top_df['Category'] == cat_label]

                    if filtered.empty:
                        st.info("해당 카테고리의 추천 결과가 없습니다.")
                        continue

                    cols = st.columns(len(filtered))
                    for col, (_, row) in zip(cols, filtered.iterrows()):
                        grade = row.get('recommendation_grade', grade_label(row['matching_score']))
                        color = GRADE_COLOR.get(grade, "#888")
                        bg    = GRADE_BG.get(grade, "#f9f9f9")

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
                        followers_str = f"{followers/10000:.1f}만" if followers >= 10000 \
                                        else f"{followers:,}"
                        follow_pct = min(int(followers / max_followers * 100), 100)

                        c_id = row['Creator_ID']
                        n_collab   = collab_count.get(c_id, 0)
                        n_success  = collab_success.get(c_id, 0)
                        badge_html = f"<span style='background:#e8f0fb; color:#2d6a9f; " \
                                     f"border-radius:12px; padding:0.1rem 0.5rem; " \
                                     f"font-size:0.75rem; font-weight:600;'>" \
                                     f"협업 {n_collab}회</span>" if n_collab > 0 else ""

                        with col:
                            st.markdown(f"""
                            <div style='border:1px solid #dde3ec; border-radius:10px;
                                        padding:1.2rem; background:white;'>
                                <div style='font-size:0.8rem; color:#888; margin-bottom:0.2rem;'>
                                    {row['Rank']}위 &nbsp; {badge_html}
                                </div>
                                <div style='font-size:1.05rem; font-weight:700; color:#1a3a5c;
                                            margin-bottom:0.4rem;'>
                                    {row['Channel_Name']}
                                </div>
                                <div style='display:inline-block; background:{color};
                                            color:white; border-radius:20px;
                                            padding:0.2rem 0.8rem; font-size:0.95rem;
                                            font-weight:700; margin-bottom:0.6rem;'>
                                    {row['matching_score']:.2f}점 &nbsp; 등급 {grade}
                                </div>
                                <div style='background:#f5f5f5; border-radius:6px;
                                            height:6px; margin-bottom:0.7rem;'>
                                    <div style='background:{color}; height:6px; border-radius:6px;
                                                width:{int(row["matching_score"]*100)}%;'></div>
                                </div>
                                <div style='font-size:0.8rem; color:#555; line-height:1.8;
                                            margin-bottom:0.6rem; text-align:left;'>
                                    {"".join(f"✔ {r}<br>" for r in reasons)}
                                </div>
                                <hr style='border:none; border-top:1px solid #eee; margin:0.4rem 0;'>
                                <div style='font-size:0.8rem; color:#555; text-align:left;
                                            line-height:1.9;'>
                                    📱 {row['Platform']}<br>
                                    🏷️ {row['Category']}<br>
                                    👥 구독자 {followers_str}<br>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # 구독자 규모 progress bar
                            st.progress(follow_pct, text=f"구독자 규모 상위 {100-follow_pct}%")

                            # 크리에이터 상세 expander (유하 스타일)
                            with st.expander("상세 정보 보기"):
                                d1, d2, d3 = st.columns(3)
                                d1.metric("참여율", f"{row['Engagement_Rate']}%")
                                d2.metric("Risk Score", f"{row['Risk_Score']}")
                                d3.metric("성공 협업", f"{n_success}회")

                                st.markdown("**점수 구성**")
                                score_data = pd.DataFrame({
                                    '항목': ['카테고리(CBF)', '조건매칭(CBF)', '협업필터링(CF)'],
                                    '점수': [row['category_score'],
                                             row['context_score'],
                                             row['cf_score']],
                                })
                                st.bar_chart(score_data.set_index('항목'))

                                # 이 크리에이터의 과거 협업 성과
                                past = collabs[collabs['Creator_ID'] == c_id][
                                    ['Brand_ID', 'CTR', 'CVR', 'is_success']
                                ].copy()
                                if not past.empty:
                                    brand_name_map = dict(zip(brands['Brand_ID'], brands['Brand_Name']))
                                    past['브랜드'] = past['Brand_ID'].map(brand_name_map)
                                    past['성공'] = past['is_success'].map({'Y': '✅', 'N': '❌'})
                                    st.markdown("**과거 협업 성과**")
                                    st.dataframe(
                                        past[['브랜드', 'CTR', 'CVR', '성공']].head(5),
                                        use_container_width=True, hide_index=True
                                    )

            # ── ③ 점수 분포 차트 (썸트렌드 스타일) ──────────────────────────
            st.subheader("③ 매칭 점수 분포")
            with st.container(border=True):
                brand_scores = similarity_df[similarity_df['Brand_ID'] == brand_id].copy()
                risk_map_all = dict(zip(creators['Creator_ID'], creators['Risk_Score']))
                brand_scores['Risk_Score'] = brand_scores['Creator_ID'].map(risk_map_all)
                brand_scores = brand_scores[brand_scores['Risk_Score'] >= risk_threshold]

                hist_data = pd.cut(
                    brand_scores['matching_score'],
                    bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    labels=['~0.4', '0.4~0.5', '0.5~0.6', '0.6~0.7', '0.7~0.8', '0.8~0.9', '0.9~']
                ).value_counts().sort_index()

                col_chart, col_info = st.columns([2, 1])
                with col_chart:
                    st.bar_chart(hist_data, color="#2d6a9f")
                with col_info:
                    st.markdown("**현재 브랜드 추천 현황**")
                    total = len(brand_scores)
                    for g, thr in [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.0)]:
                        upper = 1.1 if g == "A" else (0.9 if g == "B" else (0.8 if g == "C" else 0.7))
                        cnt = ((brand_scores['matching_score'] >= thr) &
                               (brand_scores['matching_score'] < upper)).sum()
                        pct = cnt / total * 100 if total > 0 else 0
                        st.markdown(
                            f"<span style='color:{GRADE_COLOR[g]}; font-weight:700;'>등급 {g}</span>"
                            f" &nbsp; {cnt}명 ({pct:.1f}%)",
                            unsafe_allow_html=True
                        )
                    top_ids = top_df['Creator_ID'].tolist()
                    top_scores = brand_scores[brand_scores['Creator_ID'].isin(top_ids)]
                    st.markdown(f"\n**추천된 크리에이터**: "
                                f"점수 {top_scores['matching_score'].min():.2f} ~ "
                                f"{top_scores['matching_score'].max():.2f}")

            # ── ④ 유사 협업 사례 ──────────────────────────────────────────
            st.subheader("④ 유사 협업 사례")
            brand_industry     = brand_row['Industry']
            same_industry_ids  = brands[brands['Industry'] == brand_industry]['Brand_ID'].tolist()
            top_creator_ids    = top_df['Creator_ID'].tolist()
            cases = collabs[
                (collabs['Brand_ID'].isin(same_industry_ids)) &
                (collabs['Creator_ID'].isin(top_creator_ids))
            ].copy()
            name_map_c  = dict(zip(creators['Creator_ID'], creators['Channel_Name']))
            brand_map_b = dict(zip(brands['Brand_ID'], brands['Brand_Name']))
            cases['크리에이터'] = cases['Creator_ID'].map(name_map_c)
            cases['브랜드']     = cases['Brand_ID'].map(brand_map_b)

            if cases.empty:
                st.info("동일 업종의 유사 협업 사례가 없습니다.")
            else:
                with st.container(border=True):
                    for _, c in cases.head(5).iterrows():
                        icon = "✅" if c['is_success'] == 'Y' else "❌"
                        st.markdown(
                            f"{icon} **{c['브랜드']}** + **{c['크리에이터']}** → "
                            f"예산 {c['Budget_Spent']:,}원 | "
                            f"노출 {c['Impressions']:,}회 | "
                            f"CTR {c['CTR']}% | CVR {c['CVR']}%"
                        )

            # ── ⑤ 같은 업종 브랜드 비교 (썸트렌드 스타일) ───────────────────
            st.subheader("⑤ 같은 업종 브랜드 비교")
            with st.container(border=True):
                compare_brands = brands[
                    (brands['Industry'] == brand_industry) &
                    (brands['Brand_ID'] != brand_id)
                ].head(4)

                if compare_brands.empty:
                    st.info("비교할 동일 업종 브랜드가 없습니다.")
                else:
                    comp_rows = []
                    for _, br in compare_brands.iterrows():
                        br_scores = similarity_df[similarity_df['Brand_ID'] == br['Brand_ID']]
                        avg = br_scores['matching_score'].mean()
                        top1 = br_scores.nlargest(1, 'matching_score')
                        top1_name = ""
                        if not top1.empty:
                            cid = top1.iloc[0]['Creator_ID']
                            top1_name = name_map_c.get(cid, cid)
                        comp_rows.append({
                            '브랜드': br['Brand_Name'],
                            '평균 매칭점수': round(avg, 3),
                            'Top 크리에이터': top1_name,
                        })

                    # 현재 브랜드도 포함
                    cur_scores = similarity_df[similarity_df['Brand_ID'] == brand_id]
                    cur_avg = cur_scores['matching_score'].mean()
                    cur_top1_id = cur_scores.nlargest(1, 'matching_score').iloc[0]['Creator_ID'] \
                        if not cur_scores.empty else ""
                    comp_rows.insert(0, {
                        '브랜드': f"⭐ {brand_row['Brand_Name']} (현재)",
                        '평균 매칭점수': round(cur_avg, 3),
                        'Top 크리에이터': name_map_c.get(cur_top1_id, ""),
                    })

                    comp_df = pd.DataFrame(comp_rows)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                    st.bar_chart(comp_df.set_index('브랜드')['평균 매칭점수'], color="#2d6a9f")

            # ── ⑥ 캠페인 성과 입력 ──────────────────────────────────────────
            st.subheader("⑥ 캠페인 성과 입력")
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    creator_options = top_df.apply(
                        lambda r: f"{r['Rank']}위 {r['Channel_Name']}", axis=1
                    ).tolist()
                    sel_label = st.selectbox("크리에이터 선택", creator_options)
                    sel_idx   = int(sel_label.split("위")[0]) - 1
                    sel_cid   = top_df.iloc[sel_idx]['Creator_ID']
                with col2:
                    impressions_input = st.number_input("실제 노출수", min_value=0, step=1000)
                with col3:
                    ctr_input = st.number_input("CTR (%)", min_value=0.0, max_value=100.0,
                                                step=0.1, format="%.2f")
                with col4:
                    success_input = st.selectbox("성공 여부", ["Y", "N"])

                if st.button("성과 저장", type="secondary", use_container_width=True):
                    new_row = {
                        'Collab_ID':      f"CB_NEW_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                        'Brand_ID':       brand_id,
                        'Creator_ID':     sel_cid,
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
                    save_path     = os.path.join(DATA_DIR, 'collaborations_final.csv')
                    fresh_collabs = pd.read_csv(save_path)
                    updated       = pd.concat([fresh_collabs, pd.DataFrame([new_row])],
                                              ignore_index=True)
                    updated.to_csv(save_path, index=False, encoding='utf-8-sig')
                    st.success(f"성과가 저장되었습니다! ({new_row['Collab_ID']})")
                    st.cache_data.clear()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: 크리에이터 탐색 (역방향 — 유하 스타일)
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.subheader("크리에이터 → 맞는 브랜드 탐색")
    st.caption("크리에이터 관점에서 협업 가능성이 높은 브랜드를 역방향으로 조회합니다.")

    with st.container(border=True):
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            cat_filter = st.selectbox("카테고리 필터",
                                      ["전체"] + sorted(creators['Category'].unique().tolist()))
        with ecol2:
            plat_filter = st.selectbox("플랫폼 필터",
                                       ["전체"] + sorted(creators['Platform'].unique().tolist()))
        with ecol3:
            min_risk = st.slider("최소 Risk Score", 1.0, 5.0, 2.5, 0.5)

    filtered_creators = creators.copy()
    if cat_filter != "전체":
        filtered_creators = filtered_creators[filtered_creators['Category'] == cat_filter]
    if plat_filter != "전체":
        filtered_creators = filtered_creators[filtered_creators['Platform'] == plat_filter]
    filtered_creators = filtered_creators[filtered_creators['Risk_Score'] >= min_risk]

    if filtered_creators.empty:
        st.warning("조건에 맞는 크리에이터가 없습니다.")
    else:
        creator_names = filtered_creators['Channel_Name'].tolist()
        creator_ids   = filtered_creators['Creator_ID'].tolist()
        sel_creator   = st.selectbox("크리에이터 선택", creator_names)
        sel_cid_exp   = creator_ids[creator_names.index(sel_creator)]
        creator_info  = filtered_creators[filtered_creators['Creator_ID'] == sel_cid_exp].iloc[0]

        # 크리에이터 프로필 카드
        with st.container(border=True):
            pc1, pc2, pc3, pc4, pc5 = st.columns(5)
            pc1.metric("플랫폼", creator_info['Platform'])
            pc2.metric("카테고리", creator_info['Category'])
            followers_disp = f"{creator_info['Followers']/10000:.1f}만" \
                             if creator_info['Followers'] >= 10000 \
                             else f"{creator_info['Followers']:,}"
            pc3.metric("구독자", followers_disp)
            pc4.metric("참여율", f"{creator_info['Engagement_Rate']}%")
            pc5.metric("Risk Score", f"{creator_info['Risk_Score']}")

            n_c = collab_count.get(sel_cid_exp, 0)
            n_s = collab_success.get(sel_cid_exp, 0)
            st.markdown(
                f"협업 이력 **{n_c}회** | 성공 **{n_s}회** "
                f"({int(n_s/n_c*100) if n_c > 0 else 0}%)"
            )

        st.markdown("#### 이 크리에이터에게 맞는 브랜드 Top 10")
        creator_scores = similarity_df[similarity_df['Creator_ID'] == sel_cid_exp].copy()
        creator_scores = creator_scores.nlargest(10, 'matching_score')
        brand_name_map = dict(zip(brands['Brand_ID'], brands['Brand_Name']))
        brand_ind_map  = dict(zip(brands['Brand_ID'], brands['Industry']))
        brand_bud_map  = dict(zip(brands['Brand_ID'], brands['Monthly_Budget']))
        creator_scores['브랜드']   = creator_scores['Brand_ID'].map(brand_name_map)
        creator_scores['업종']     = creator_scores['Brand_ID'].map(brand_ind_map)
        creator_scores['월예산']   = creator_scores['Brand_ID'].map(brand_bud_map)
        creator_scores['등급']     = creator_scores['recommendation_grade'] \
                                     if 'recommendation_grade' in creator_scores.columns \
                                     else creator_scores['matching_score'].apply(grade_label)
        creator_scores['순위']     = range(1, len(creator_scores) + 1)

        display_cols = ['순위', '브랜드', '업종', '월예산', 'matching_score', '등급']
        st.dataframe(
            creator_scores[display_cols].rename(columns={'matching_score': '매칭점수'}),
            use_container_width=True, hide_index=True
        )

        st.bar_chart(
            creator_scores.set_index('브랜드')['matching_score'],
            color="#1a7a4a"
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: 성과 대시보드 (썸트렌드 스타일)
# ════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.subheader("캠페인 성과 대시보드")

    # KPI 요약
    total_collabs  = len(collabs)
    success_cnt    = (collabs['is_success'] == 'Y').sum()
    success_rate   = success_cnt / total_collabs * 100 if total_collabs > 0 else 0
    avg_ctr        = collabs['CTR'].mean()
    avg_cvr        = collabs['CVR'].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 협업 수",   f"{total_collabs:,}건")
    k2.metric("성공률",       f"{success_rate:.1f}%")
    k3.metric("평균 CTR",    f"{avg_ctr:.2f}%")
    k4.metric("평균 CVR",    f"{avg_cvr:.2f}%")

    st.divider()

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("**업종별 성공률**")
        brand_ind = dict(zip(brands['Brand_ID'], brands['Industry']))
        collabs_ind = collabs.copy()
        collabs_ind['Industry'] = collabs_ind['Brand_ID'].map(brand_ind)
        ind_stats = collabs_ind.groupby('Industry').apply(
            lambda x: round((x['is_success'] == 'Y').mean() * 100, 1)
        ).reset_index()
        ind_stats.columns = ['업종', '성공률(%)']
        ind_stats = ind_stats.sort_values('성공률(%)', ascending=False)
        st.bar_chart(ind_stats.set_index('업종'), color="#1a7a4a")

    with dcol2:
        st.markdown("**카테고리별 평균 CTR**")
        creator_cat = dict(zip(creators['Creator_ID'], creators['Category']))
        collabs_cat = collabs.copy()
        collabs_cat['Category'] = collabs_cat['Creator_ID'].map(creator_cat)
        cat_ctr = collabs_cat.groupby('Category')['CTR'].mean().round(2).reset_index()
        cat_ctr.columns = ['카테고리', '평균CTR(%)']
        cat_ctr = cat_ctr.sort_values('평균CTR(%)', ascending=False)
        st.bar_chart(cat_ctr.set_index('카테고리'), color="#2d6a9f")

    st.divider()

    dcol3, dcol4 = st.columns(2)

    with dcol3:
        st.markdown("**성공 협업 Top 10 크리에이터**")
        top_creators = collabs[collabs['is_success'] == 'Y'] \
            .groupby('Creator_ID').size().nlargest(10).reset_index()
        top_creators.columns = ['Creator_ID', '성공횟수']
        top_creators['크리에이터'] = top_creators['Creator_ID'].map(
            dict(zip(creators['Creator_ID'], creators['Channel_Name']))
        )
        st.dataframe(
            top_creators[['크리에이터', '성공횟수']],
            use_container_width=True, hide_index=True
        )

    with dcol4:
        st.markdown("**노출수 vs CTR 산점도**")
        sample = collabs.sample(min(200, len(collabs)), random_state=42)[
            ['Impressions', 'CTR', 'is_success']
        ].copy()
        sample['색상'] = sample['is_success'].map({'Y': '#1a7a4a', 'N': '#c0392b'})
        st.scatter_chart(
            sample,
            x='Impressions', y='CTR',
            color='색상', size=30
        )

    st.divider()
    st.markdown("**전체 협업 데이터**")
    display_collabs = collabs.copy()
    display_collabs['크리에이터'] = display_collabs['Creator_ID'].map(
        dict(zip(creators['Creator_ID'], creators['Channel_Name']))
    )
    display_collabs['브랜드'] = display_collabs['Brand_ID'].map(
        dict(zip(brands['Brand_ID'], brands['Brand_Name']))
    )
    st.dataframe(
        display_collabs[['브랜드', '크리에이터', 'CTR', 'CVR',
                          'Impressions', 'Budget_Spent', 'is_success']].rename(
            columns={'is_success': '성공'}
        ),
        use_container_width=True, hide_index=True
    )


# ── 푸터 ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "KAIST BIZ | 비즈니스 애널리틱스 2026 | "
    "CBF + CF Hybrid Recommendation System</p>",
    unsafe_allow_html=True
)
