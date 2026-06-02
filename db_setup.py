"""
CSV 파일을 SQLite(creator.db)로 변환하는 초기 설정 스크립트.
앱 최초 실행 전 1회 실행하거나, app.py가 자동으로 호출합니다.
"""
import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'creator.db')


# ── DDL: 테이블 생성 ─────────────────────────────────────────────────────────
DDL = """
CREATE TABLE IF NOT EXISTS Creator (
    Creator_ID       TEXT PRIMARY KEY,
    Channel_Name     TEXT,
    Platform         TEXT,
    Category         TEXT,
    Followers        INTEGER,
    Engagement_Rate  REAL,
    Estimated_CPM    REAL,
    Risk_Score       REAL,
    Target_Age       TEXT,
    Target_Gender    TEXT
);

CREATE TABLE IF NOT EXISTS Brand (
    Brand_ID            TEXT PRIMARY KEY,
    Brand_Name          TEXT,
    Industry            TEXT,
    Monthly_Budget      INTEGER,
    Max_CPM             REAL,
    Target_Age          TEXT,
    Target_Gender       TEXT,
    Preferred_Platform  TEXT
);

CREATE TABLE IF NOT EXISTS Campaign (
    Collab_ID       TEXT PRIMARY KEY,
    Brand_ID        TEXT,
    Creator_ID      TEXT,
    Campaign_Start  TEXT,
    Campaign_End    TEXT,
    Budget_Spent    INTEGER,
    Impressions     INTEGER,
    Clicks          INTEGER,
    CTR             REAL,
    Conversions     INTEGER,
    CVR             REAL,
    is_success      TEXT,
    FOREIGN KEY (Brand_ID)    REFERENCES Brand(Brand_ID),
    FOREIGN KEY (Creator_ID)  REFERENCES Creator(Creator_ID)
);

CREATE TABLE IF NOT EXISTS Ratings (
    Brand_ID    TEXT,
    Creator_ID  TEXT,
    Score       REAL,
    PRIMARY KEY (Brand_ID, Creator_ID),
    FOREIGN KEY (Brand_ID)    REFERENCES Brand(Brand_ID),
    FOREIGN KEY (Creator_ID)  REFERENCES Creator(Creator_ID)
);

CREATE TABLE IF NOT EXISTS CreatorSimilarity (
    Brand_ID              TEXT,
    Creator_ID            TEXT,
    category_score        REAL,
    context_score         REAL,
    cf_score              REAL,
    matching_score        REAL,
    recommendation_grade  TEXT,
    PRIMARY KEY (Brand_ID, Creator_ID),
    FOREIGN KEY (Brand_ID)    REFERENCES Brand(Brand_ID),
    FOREIGN KEY (Creator_ID)  REFERENCES Creator(Creator_ID)
);
"""

# CSV → 테이블 매핑 (파일명, 테이블명, 필요 컬럼)
CSV_MAP = {
    'creators_clean.csv': {
        'table': 'Creator',
        'cols': ['Creator_ID', 'Channel_Name', 'Platform', 'Category',
                 'Followers', 'Engagement_Rate', 'Estimated_CPM',
                 'Risk_Score', 'Target_Age', 'Target_Gender'],
    },
    'brands_100.csv': {
        'table': 'Brand',
        'cols': ['Brand_ID', 'Brand_Name', 'Industry', 'Monthly_Budget',
                 'Max_CPM', 'Target_Age', 'Target_Gender', 'Preferred_Platform'],
    },
    'collaborations_final.csv': {
        'table': 'Campaign',
        'cols': ['Collab_ID', 'Brand_ID', 'Creator_ID', 'Campaign_Start',
                 'Campaign_End', 'Budget_Spent', 'Impressions', 'Clicks',
                 'CTR', 'Conversions', 'CVR', 'is_success'],
    },
    'ratings_clean.csv': {
        'table': 'Ratings',
        'cols': ['Brand_ID', 'Creator_ID', 'Score'],
    },
}


def setup_db(force=False):
    """
    creator.db 생성 및 CSV 데이터 적재.
    force=True 이면 기존 데이터를 모두 삭제하고 재적재합니다.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)
    conn.commit()

    for filename, meta in CSV_MAP.items():
        table = meta['table']
        cols  = meta['cols']

        if not force:
            cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if cnt > 0:
                print(f"  [{table}] 이미 {cnt:,}건 존재 — 스킵")
                continue

        csv_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(csv_path):
            print(f"  [{table}] 파일 없음: {filename} — 스킵")
            continue

        df = pd.read_csv(csv_path)
        # 테이블 컬럼과 교집합만 사용 (CSV에 추가 컬럼 있어도 무시)
        available = [c for c in cols if c in df.columns]
        df[available].to_sql(table, conn, if_exists='replace', index=False)
        print(f"  [{table}] {len(df):,}건 적재 완료")

    conn.close()
    print(f"\ncreator.db 준비 완료: {DB_PATH}")


# ── SQL 헬퍼 함수 (app.py에서 import) ────────────────────────────────────────

def get_conn():
    return sqlite3.connect(DB_PATH)


# P1-SQL1: Creator + Brand 조인하여 기본 점수 계산용 데이터 조회
P1_SQL1 = """
SELECT
    c.Creator_ID,
    c.Channel_Name,
    c.Platform,
    c.Category,
    c.Followers,
    c.Engagement_Rate,
    c.Estimated_CPM,
    c.Risk_Score,
    c.Target_Age    AS Creator_Age,
    c.Target_Gender AS Creator_Gender
FROM Creator c
WHERE c.Risk_Score >= :risk_threshold
"""

# P1-SQL2: CreatorSimilarity 일괄 INSERT (upsert)
P1_SQL2 = """
INSERT OR REPLACE INTO CreatorSimilarity
    (Brand_ID, Creator_ID, category_score, context_score,
     cf_score, matching_score, recommendation_grade)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

# P1-SQL3: CreatorSimilarity → 특정 브랜드 Top N 반환
P1_SQL3 = """
SELECT
    cs.Brand_ID,
    cs.Creator_ID,
    c.Channel_Name,
    c.Platform,
    c.Category,
    c.Followers,
    c.Engagement_Rate,
    c.Risk_Score,
    cs.category_score,
    cs.context_score,
    cs.cf_score,
    cs.matching_score,
    cs.recommendation_grade
FROM CreatorSimilarity cs
JOIN Creator c ON cs.Creator_ID = c.Creator_ID
WHERE cs.Brand_ID = :brand_id
  AND c.Risk_Score >= :risk_threshold
ORDER BY cs.matching_score DESC
LIMIT :top_n
"""

# P2-SQL1: Campaign 성과 INSERT
P2_SQL1 = """
INSERT INTO Campaign
    (Collab_ID, Brand_ID, Creator_ID, Campaign_Start, Campaign_End,
     Budget_Spent, Impressions, Clicks, CTR, Conversions, CVR, is_success)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# 추가 조회 SQL
SQL_BRANDS       = "SELECT * FROM Brand"
SQL_CREATORS     = "SELECT * FROM Creator"
SQL_CAMPAIGNS    = "SELECT * FROM Campaign"
SQL_RATINGS      = "SELECT * FROM Ratings"
SQL_SIMILARITY   = "SELECT * FROM CreatorSimilarity"

SQL_COLLAB_COUNT = """
SELECT Creator_ID, COUNT(*) AS cnt
FROM Campaign
GROUP BY Creator_ID
"""

SQL_COLLAB_SUCCESS = """
SELECT Creator_ID, COUNT(*) AS cnt
FROM Campaign
WHERE is_success = 'Y'
GROUP BY Creator_ID
"""

SQL_SIMILAR_CASES = """
SELECT
    camp.Brand_ID,
    camp.Creator_ID,
    b.Brand_Name,
    c.Channel_Name  AS Creator_Name,
    camp.Budget_Spent,
    camp.Impressions,
    camp.CTR,
    camp.CVR,
    camp.is_success
FROM Campaign camp
JOIN Brand   b ON camp.Brand_ID   = b.Brand_ID
JOIN Creator c ON camp.Creator_ID = c.Creator_ID
WHERE b.Industry = :industry
  AND camp.Creator_ID IN ({placeholders})
LIMIT 5
"""


def save_similarity(similarity_df):
    """build_similarity() 결과를 CreatorSimilarity 테이블에 저장 (P1-SQL2)."""
    conn = get_conn()
    conn.execute("DELETE FROM CreatorSimilarity")
    rows = [
        (r.Brand_ID, r.Creator_ID, r.category_score, r.context_score,
         r.cf_score, r.matching_score, r.recommendation_grade)
        for r in similarity_df.itertuples(index=False)
    ]
    conn.executemany(P1_SQL2, rows)
    conn.commit()
    conn.close()


def save_campaign(row_dict):
    """캠페인 성과를 Campaign 테이블에 INSERT (P2-SQL1)."""
    conn = get_conn()
    conn.execute(P2_SQL1, (
        row_dict['Collab_ID'],
        row_dict['Brand_ID'],
        row_dict['Creator_ID'],
        str(row_dict['Campaign_Start']),
        str(row_dict['Campaign_End']),
        row_dict['Budget_Spent'],
        row_dict['Impressions'],
        row_dict['Clicks'],
        row_dict['CTR'],
        row_dict['Conversions'],
        row_dict['CVR'],
        row_dict['is_success'],
    ))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    print("creator.db 초기화 시작...")
    setup_db(force=True)
