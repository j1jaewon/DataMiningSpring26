# 기업-크리에이터 매칭 추천 시스템
**Data Mining Spring 2026 | Dr. Won (KTNET)**

---

## 파일 목록

| 파일 | 건수 | 설명 | Access 테이블 |
|---|---|---|---|
| creators_clean.csv | 490건 | 크리에이터 정보. 녹스 수집 + LLM 증강. Risk_Score 포함 | Creator |
| brands_100.csv | 100건 | 가상 브랜드(광고주) 정보. 업종 10개×10개 | Brand |
| collaborations_final.csv | 976건 | 협업 이력 + 성과. is_success(Y/N) 포함 | Campaign |
| ratings_clean.csv | 976건 | 브랜드→크리에이터 평점(1~5점). CF 알고리즘 입력값 | Ratings |
| creator_similarity.csv | 9,480건 | 추천 점수 사전계산 결과. **핵심 파일** | CreatorSimilarity |
| recommendation_logic.py | — | 추천 로직 Python 코드 | — |
| recommendation_top3_sample.csv | — | Top 3 결과 샘플 (검증용 정답지) | — |

---

## 추천 등급 기준

| 등급 | 점수 | PoC 내 성공률 |
|---|---|---|
| A | 0.9 이상 | 75.7% — 강력 추천 |
| B | 0.8 ~ 0.9 | 58.8% — 추천 |
| C | 0.7 ~ 0.8 | 46.6% — 보통 |
| D | 0.7 미만 | 20.0% — 비추천 |

※ Risk_Score 2.5 미만은 점수 무관 자동 제외

---

## ⚠️ PoC 임의 설정값 안내

교수님 Q&A 시 **"PoC 설정이며 실서비스에서 재검증 예정"** 으로 답하세요.

| 항목 | 설정값 | 비고 |
|---|---|---|
| matching_score 공식 | category×0.3 + context×0.3 + cf×0.4 | 가중치 우리 설정 |
| context_score 배점 | 연령·성별·CPM·리스크 각 +0.25 | 동등 배분 우리 설정 |
| is_success 기준 | 뷰티≥3%, 테크≥1%, 식품≥3.5% 등 | 업계 벤치마크 참고 후 우리 설정 |
| 추천 임계값 | 0.8 | PoC 내 자기검증. 외부 실데이터 검증 아님 |
| Risk 제외 기준 | 2.5 미만 제외 | 우리 설정 |

> 공식·가중치·기준 모두 논리적으로 설계했지만 PoC용 설정값입니다.
> 실서비스 전환 시 실데이터 전면 재검증 필요.

---

## Streamlit 사용법 (승모)

```python
from recommendation_logic import recommend
result = recommend('BR_00001')  # Top 3 반환
```
