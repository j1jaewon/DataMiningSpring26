===============================================================
  기업-크리에이터 매칭 추천 시스템 — 데이터 패키지 README
  작성: Dr. Won (KTNET) | 2026-05-31
===============================================================

■ 이 ZIP에 들어있는 파일들

1. creators_clean.csv (490건 / 24컬럼)
   - 국내 유튜브·틱톡 크리에이터 정보
   - 녹스인플루언서 수집 + LLM 증강으로 구성
   - ★ Risk_Score 컬럼: 논란이력·업로드빈도·조회수추이 등 종합 리스크 점수 (1~5점)
   - MS Access → Creator 테이블로 Import

2. brands_100.csv (100건 / 16컬럼)
   - 국내 가상 브랜드(광고주) 정보
   - 업종 10개 × 10개, SMB 60% / 중견 30% / 대기업 10%
   - MS Access → Brand 테이블로 Import

3. collaborations_final.csv (976건 / 12컬럼)
   - 브랜드-크리에이터 과거 협업 이력 및 캠페인 성과
   - ★ is_success 컬럼: CVR이 업종별 기준 이상이면 Y(성공), 미만이면 N(실패)
     예) 뷰티 CVR≥3% → Y / 테크 CVR≥1% → Y
   - MS Access → Campaign 테이블로 Import

4. ratings_clean.csv (976건 / 7컬럼)
   - 협업 후 브랜드가 크리에이터에게 매긴 만족도 평점 (1.0~5.0점)
   - CF(협업 필터링) 추천 알고리즘의 핵심 입력 데이터
   - MS Access → Ratings 테이블로 Import

5. creator_similarity.csv (9,480건 / 7컬럼)
   - 추천 로직을 사전 계산한 결과 테이블 (핵심!)
   - 브랜드 100개 × 카테고리 일치 크리에이터 조합의 점수
   - 컬럼 설명:
     · category_score : 브랜드 업종 ↔ 크리에이터 카테고리 일치도 (1.0/0.5)
     · context_score  : 연령·성별·CPM·리스크 조건 매칭 (0~1.0)
     · cf_score       : 코사인 유사도 기반 협업 필터링 점수 (0~1.0)
     · matching_score : 최종 추천 점수 (높을수록 적합)
     · recommendation_grade : A(강력추천/0.9↑) B(추천/0.8~) C(보통/0.7~) D(미흡)
   - MS Access → CreatorSimilarity 테이블로 Import

6. recommendation_logic.py
   - 위 creator_similarity.csv를 생성한 Python 코드
   - CBF(카테고리+조건 매칭) + CF(코사인 유사도) 하이브리드 알고리즘
   - Streamlit 개발 시 recommend() 함수를 import해서 사용
     사용법: result = recommend('BR_00001')  # Top 3 반환

7. recommendation_top3_sample.csv
   - 업종별 브랜드 5개의 Top 3 추천 결과 샘플
   - Access에서 creator_similarity 쿼리 결과 검증용 정답지

---------------------------------------------------------------

■ 역할별 활용 방법

[IBK — 다솜]
  → CSV 1~5번을 MS Access에 Import
  → ERD 구현 (PK/FK 연결)
     Creator.Creator_ID ← CollaborationsCreator_ID
     Brand.Brand_ID     ← Collaborations.Brand_ID
     Collaborations.Collab_ID ← Ratings.Collab_ID
  → SQL 쿼리 작성·실행 (교수님 제출용)
  → 추천 결과 검증: creator_similarity에서 TOP 3 쿼리 실행 후
    recommendation_top3_sample.csv와 비교

[승모 — Python/Streamlit]
  → CSV 파일들을 Colab 세션에 올리고 SQLite로 변환
  → recommendation_logic.py의 recommend() 함수를 import
  → 브랜드 조건 입력 → recommend(brand_id) 호출 → Top 3 반환
  → Streamlit UI에 결과 카드 출력

[재원 — UI/SQL]
  → Access에서 추천 결과 조회 SQL:
    SELECT TOP 3 *
    FROM CreatorSimilarity
    WHERE Brand_ID = 'BR_00001'
    ORDER BY matching_score DESC

---------------------------------------------------------------

■ 추천 점수 기준 (데이터 기반, 주관 아님)

  A등급 (0.9 이상) → 실제 협업 성공률 75.7%  ← 강력 추천
  B등급 (0.8~0.9)  → 실제 협업 성공률 58.8%  ← 추천
  C등급 (0.7~0.8)  → 실제 협업 성공률 46.6%  ← 보통 (절반 이상 실패)
  D등급 (0.7 미만) → 실제 협업 성공률 20.0%  ← 비추천

  ※ Risk_Score 2.5 미만은 점수와 무관하게 자동 제외
  ※ 이 성공률은 PoC 가상 데이터 기반. 실서비스에서는 재검증 필요

---------------------------------------------------------------

■ 교수님 Q&A 예상 대응

Q: "추천 알고리즘이 뭔가요?"
A: "CBF(콘텐츠 기반 필터링)와 CF(협업 필터링)를 결합한 하이브리드 방식입니다.
   CBF는 카테고리·연령·성별·예산 조건을 매칭하고,
   CF는 ratings 평점 행렬의 코사인 유사도로 유사 브랜드의 경험을 반영합니다."

Q: "0.8 기준은 어떻게 정했나요?"
A: "주관이 아니라 collaborations_final의 is_success와
   creator_similarity의 matching_score를 조인 분석한 결과,
   0.8 기준에서 성공률이 46.6% → 58.8%로 급상승해 임계값으로 설정했습니다."

Q: "is_success 기준은요?"
A: "업계 벤치마크(FirstPageSage 2025, 이커머스 CVR 2~3%)를 참고해
   업종별로 차등 적용했습니다. PoC 단계이므로 실서비스에서는 재검증 예정입니다."

===============================================================
