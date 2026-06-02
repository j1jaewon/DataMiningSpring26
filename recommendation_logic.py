import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'

# ===== 등급 기준 (A≥0.9, B≥0.8, C≥0.7, D<0.7) =====
def grade_label(score):
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    else:
        return "D"


# ===== STEP 1: CBF — 카테고리 점수 =====
category_map = {
    '뷰티': ['뷰티'],
    '패션': ['패션', '라이프스타일'],
    '식품': ['푸드'],
    '테크': ['테크', '게임'],
    '생활용품': ['라이프스타일'],
    '피트니스': ['피트니스', '라이프스타일'],
    '교육': ['교육'],
    '여행': ['여행', '라이프스타일'],
    '게임': ['게임', '테크'],
    '헬스케어': ['라이프스타일', '피트니스'],
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


# ===== STEP 2: CBF — 조건 매칭 점수 =====
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


# ===== STEP 3: CF — 협업 필터링 점수 =====
def build_cf_matrix(ratings):
    rating_matrix = ratings.pivot_table(
        index='Brand_ID', columns='Creator_ID', values='Score', aggfunc='mean'
    ).fillna(0)
    brand_similarity = pd.DataFrame(
        cosine_similarity(rating_matrix),
        index=rating_matrix.index,
        columns=rating_matrix.index
    )
    return rating_matrix, brand_similarity


def calc_cf_score(brand_id, creator_id, rating_matrix, brand_similarity):
    if brand_id not in rating_matrix.index or creator_id not in rating_matrix.columns:
        return 0.0
    sim_scores = brand_similarity[brand_id].drop(brand_id)
    top_similar = sim_scores.nlargest(5)
    numerator = denominator = 0
    for other_brand, sim in top_similar.items():
        if sim > 0 and rating_matrix.loc[other_brand, creator_id] > 0:
            numerator += sim * rating_matrix.loc[other_brand, creator_id]
            denominator += sim
    if denominator == 0:
        return 0.0
    return round(numerator / denominator / 5.0, 4)


# ===== STEP 4: 전체 매칭 점수 계산 =====
def build_similarity(creators, brands, ratings):
    rating_matrix, brand_sim = build_cf_matrix(ratings)

    results = []
    for b in brands.to_dict('records'):
        for c in creators.to_dict('records'):
            cat_score = calc_category_score(b['Industry'], c['Category'])
            if cat_score == 0:
                continue
            ctx_score = calc_context_score(b, c)
            cf_score  = calc_cf_score(b['Brand_ID'], c['Creator_ID'],
                                      rating_matrix, brand_sim)
            matching  = cat_score*0.3 + ctx_score*0.3 + cf_score*0.4 if cf_score > 0 \
                        else cat_score*0.5 + ctx_score*0.5
            results.append({
                'Brand_ID':              b['Brand_ID'],
                'Creator_ID':            c['Creator_ID'],
                'category_score':        round(cat_score, 4),
                'context_score':         round(ctx_score, 4),
                'cf_score':              round(cf_score, 4),
                'matching_score':        round(matching, 4),
                'recommendation_grade':  grade_label(round(matching, 4)),
            })
    return pd.DataFrame(results)


# ===== STEP 5: 추천 함수 =====
def recommend(brand_id, similarity_df, creators, risk_threshold=2.5, top_n=3):
    result = similarity_df[similarity_df['Brand_ID'] == brand_id].copy()
    risk_map   = dict(zip(creators['Creator_ID'], creators['Risk_Score']))
    name_map   = dict(zip(creators['Creator_ID'], creators['Channel_Name']))
    cat_map    = dict(zip(creators['Creator_ID'], creators['Category']))
    plat_map   = dict(zip(creators['Creator_ID'], creators['Platform']))
    follow_map = dict(zip(creators['Creator_ID'], creators['Followers']))
    er_map     = dict(zip(creators['Creator_ID'], creators['Engagement_Rate']))

    result['Risk_Score'] = result['Creator_ID'].map(risk_map)
    result = result[result['Risk_Score'] >= risk_threshold]
    top = result.nlargest(top_n, 'matching_score').copy()
    top['Channel_Name']    = top['Creator_ID'].map(name_map)
    top['Category']        = top['Creator_ID'].map(cat_map)
    top['Platform']        = top['Creator_ID'].map(plat_map)
    top['Followers']       = top['Creator_ID'].map(follow_map)
    top['Engagement_Rate'] = top['Creator_ID'].map(er_map)
    top['Rank']            = range(1, len(top) + 1)

    return top[['Rank', 'Channel_Name', 'Category', 'Platform', 'Followers',
                'Engagement_Rate', 'category_score', 'context_score',
                'cf_score', 'matching_score', 'recommendation_grade',
                'Risk_Score', 'Creator_ID']].reset_index(drop=True)


# ===== 독립 실행 시: 점수 사전계산 후 CSV 저장 =====
if __name__ == '__main__':
    print("=" * 60)
    print("  추천 로직 실행")
    print("=" * 60)

    creators_df = pd.read_csv(os.path.join(BASE_DIR, 'creators_clean.csv'))
    brands_df   = pd.read_csv(os.path.join(BASE_DIR, 'brands_100.csv'))
    ratings_df  = pd.read_csv(os.path.join(BASE_DIR, 'ratings_clean.csv'))

    print("\n[STEP 1~4] 전체 매칭 점수 계산 중...")
    similarity_df = build_similarity(creators_df, brands_df, ratings_df)
    print(f"  → 계산 완료: {len(similarity_df):,}건")

    out_path = os.path.join(BASE_DIR, 'creator_similarity.csv')
    similarity_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n[STEP 5] 저장 완료: {out_path}")
    print(f"  matching_score 평균: {similarity_df['matching_score'].mean():.4f}")
    print(f"  CF 점수 있는 조합: {(similarity_df['cf_score']>0).sum():,}건 "
          f"({(similarity_df['cf_score']>0).mean()*100:.1f}%)")

    print("\n" + "=" * 60)
    print("  추천 테스트 실행")
    print("=" * 60)
    test_id   = brands_df.iloc[0]['Brand_ID']
    test_name = brands_df.iloc[0]['Brand_Name']
    print(f"\n브랜드: {test_name} ({test_id})")
    result = recommend(test_id, similarity_df, creators_df)
    print(result.to_string(index=False))
