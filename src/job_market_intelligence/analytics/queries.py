"""DuckDB-backed analytical queries over processed job-posting files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

PROJECT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
DEFAULT_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
DEFAULT_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"


@dataclass(frozen=True)
class DatasetPaths:
    """Processed dataset files used by the analytics layer."""

    train_path: Path = DEFAULT_TRAIN_PATH
    test_path: Path = DEFAULT_TEST_PATH

    def validate(self) -> None:
        """Fail early when the expected processed files are not available."""
        missing_paths = [
            str(path)
            for path in (self.train_path, self.test_path)
            if not path.exists()
        ]
        if missing_paths:
            joined_paths = ", ".join(missing_paths)
            raise FileNotFoundError(f"Processed dataset file(s) not found: {joined_paths}")


class MarketAnalytics:
    """Run SQL analytics over the processed train/test CSV files."""

    def __init__(self, dataset_paths: DatasetPaths | None = None) -> None:
        self.dataset_paths = dataset_paths or DatasetPaths()

    def get_dataset_summary(self) -> dict[str, int | float | None]:
        """Return high-level market dataset metrics."""
        self.dataset_paths.validate()
        query = f"""
            WITH postings AS ({self._postings_relation_sql()})
            SELECT
                COUNT(*) AS total_postings,
                COUNT(DISTINCT job_category) AS total_categories,
                SUM(CASE WHEN has_salary THEN 1 ELSE 0 END) AS postings_with_salary,
                ROUND(
                    SUM(CASE WHEN has_salary THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
                    2
                ) AS salary_coverage_pct,
                SUM(
                    CASE
                        WHEN has_salary AND pay_period = 'YEARLY' THEN 1
                        ELSE 0
                    END
                ) AS yearly_salary_postings,
                ROUND(
                    AVG(
                        CASE
                            WHEN has_salary AND pay_period = 'YEARLY'
                            THEN max_salary
                            ELSE NULL
                        END
                    ),
                    2
                ) AS average_yearly_max_salary
            FROM postings
        """
        with duckdb.connect(database=":memory:") as connection:
            row = connection.execute(query).fetchone()

        if row is None:
            return {
                "total_postings": 0,
                "total_categories": 0,
                "postings_with_salary": 0,
                "salary_coverage_pct": 0.0,
                "yearly_salary_postings": 0,
                "average_yearly_max_salary": None,
            }

        return {
            "total_postings": int(row[0]),
            "total_categories": int(row[1]),
            "postings_with_salary": int(row[2] or 0),
            "salary_coverage_pct": float(row[3] or 0.0),
            "yearly_salary_postings": int(row[4] or 0),
            "average_yearly_max_salary": float(row[5]) if row[5] is not None else None,
        }

    def get_category_distribution(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return posting counts and percentages by job category."""
        self.dataset_paths.validate()
        limit_clause = ""
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be greater than 0.")
            limit_clause = f"LIMIT {limit}"

        query = f"""
            WITH postings AS ({self._postings_relation_sql()}),
            category_counts AS (
                SELECT
                    job_category,
                    COUNT(*) AS total
                FROM postings
                GROUP BY job_category
            ),
            dataset_total AS (
                SELECT COUNT(*) AS total FROM postings
            )
            SELECT
                category_counts.job_category,
                category_counts.total,
                ROUND(category_counts.total * 100.0 / dataset_total.total, 2)
                    AS percentage
            FROM category_counts
            CROSS JOIN dataset_total
            ORDER BY category_counts.total DESC, category_counts.job_category ASC
            {limit_clause}
        """
        with duckdb.connect(database=":memory:") as connection:
            rows = connection.execute(query).fetchall()

        return [
            {
                "job_category": str(row[0]),
                "total": int(row[1]),
                "percentage": float(row[2]),
            }
            for row in rows
        ]

    def get_work_type_distribution(self) -> list[dict[str, Any]]:
        """Return posting counts and percentages by employment type."""
        self.dataset_paths.validate()
        query = f"""
            WITH postings AS ({self._postings_relation_sql()}),
            work_type_counts AS (
                SELECT
                    COALESCE(formatted_work_type, 'Unknown') AS work_type,
                    COUNT(*) AS total
                FROM postings
                GROUP BY COALESCE(formatted_work_type, 'Unknown')
            ),
            dataset_total AS (
                SELECT COUNT(*) AS total FROM postings
            )
            SELECT
                work_type_counts.work_type,
                work_type_counts.total,
                ROUND(work_type_counts.total * 100.0 / dataset_total.total, 2)
                    AS percentage
            FROM work_type_counts
            CROSS JOIN dataset_total
            ORDER BY work_type_counts.total DESC, work_type_counts.work_type ASC
        """
        with duckdb.connect(database=":memory:") as connection:
            rows = connection.execute(query).fetchall()

        return [
            {
                "work_type": str(row[0]),
                "total": int(row[1]),
                "percentage": float(row[2]),
            }
            for row in rows
        ]

    def get_yearly_salary_by_category(self) -> list[dict[str, Any]]:
        """Return yearly max-salary statistics by job category."""
        self.dataset_paths.validate()
        query = f"""
            WITH postings AS ({self._postings_relation_sql()})
            SELECT
                job_category,
                COUNT(*) AS salary_postings,
                ROUND(AVG(max_salary), 2) AS average_max_salary,
                ROUND(MEDIAN(max_salary), 2) AS median_max_salary
            FROM postings
            WHERE has_salary AND pay_period = 'YEARLY' AND max_salary IS NOT NULL
            GROUP BY job_category
            ORDER BY salary_postings DESC, job_category ASC
        """
        with duckdb.connect(database=":memory:") as connection:
            rows = connection.execute(query).fetchall()

        return [
            {
                "job_category": str(row[0]),
                "salary_postings": int(row[1]),
                "average_max_salary": float(row[2]),
                "median_max_salary": float(row[3]),
                "pay_period": "YEARLY",
            }
            for row in rows
        ]

    def _postings_relation_sql(self) -> str:
        """Build the reusable DuckDB relation for processed postings."""
        train_path = _escape_sql_path(self.dataset_paths.train_path)
        test_path = _escape_sql_path(self.dataset_paths.test_path)
        return f"""
            SELECT *, 'train' AS dataset_split
            FROM read_csv_auto('{train_path}', union_by_name = true)
            UNION ALL
            SELECT *, 'test' AS dataset_split
            FROM read_csv_auto('{test_path}', union_by_name = true)
        """


def _escape_sql_path(path: Path) -> str:
    """Escape a local filesystem path for DuckDB SQL string literals."""
    return str(path).replace("'", "''")
