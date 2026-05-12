"""Tests for DuckDB market analytics queries."""

from pathlib import Path

import pytest

from src.analytics.queries import DatasetPaths, MarketAnalytics


CSV_HEADER = (
    "job_id,title,description,max_salary,pay_period,"
    "formatted_work_type,job_category,has_salary\n"
)


def write_processed_csv(path: Path, rows: list[str]) -> None:
    """Write a small processed CSV fixture."""
    path.write_text(CSV_HEADER + "".join(rows), encoding="utf-8")


def test_dataset_summary_uses_train_and_test_files(tmp_path: Path) -> None:
    """Summary metrics should combine both processed splits."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    write_processed_csv(
        train_path,
        [
            "1,Data Engineer,Build pipelines,100000,YEARLY,Full-time,Data Engineering,true\n",
            "2,Frontend Developer,Build UI,,yearly,Full-time,Frontend,false\n",
        ],
    )
    write_processed_csv(
        test_path,
        [
            "3,ML Engineer,Train models,120000,YEARLY,Full-time,AI/ML,true\n",
        ],
    )
    analytics = MarketAnalytics(
        DatasetPaths(train_path=train_path, test_path=test_path)
    )

    summary = analytics.get_dataset_summary()

    assert summary == {
        "total_postings": 3,
        "total_categories": 3,
        "postings_with_salary": 2,
        "salary_coverage_pct": 66.67,
        "yearly_salary_postings": 2,
        "average_yearly_max_salary": 110000.0,
    }


def test_category_distribution_returns_ordered_percentages(tmp_path: Path) -> None:
    """Category distribution should be sorted by count and then category name."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    write_processed_csv(
        train_path,
        [
            "1,Data Engineer,Build pipelines,100000,yearly,Full-time,Data Engineering,true\n",
            "2,Analytics Engineer,Model data,,yearly,Full-time,Data Engineering,false\n",
        ],
    )
    write_processed_csv(
        test_path,
        [
            "3,Frontend Developer,Build UI,,yearly,Full-time,Frontend,false\n",
        ],
    )
    analytics = MarketAnalytics(
        DatasetPaths(train_path=train_path, test_path=test_path)
    )

    categories = analytics.get_category_distribution()

    assert categories == [
        {
            "job_category": "Data Engineering",
            "total": 2,
            "percentage": 66.67,
        },
        {
            "job_category": "Frontend",
            "total": 1,
            "percentage": 33.33,
        },
    ]


def test_category_distribution_rejects_invalid_limit(tmp_path: Path) -> None:
    """Limit must be positive when provided."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    write_processed_csv(train_path, [])
    write_processed_csv(test_path, [])
    analytics = MarketAnalytics(
        DatasetPaths(train_path=train_path, test_path=test_path)
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        analytics.get_category_distribution(limit=0)


def test_work_type_distribution_returns_ordered_percentages(tmp_path: Path) -> None:
    """Work type distribution should be sorted by count and then type name."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    write_processed_csv(
        train_path,
        [
            "1,Data Engineer,Build pipelines,100000,YEARLY,Full-time,Data Engineering,true\n",
            "2,Frontend Developer,Build UI,,YEARLY,Contract,Frontend,false\n",
        ],
    )
    write_processed_csv(
        test_path,
        [
            "3,ML Engineer,Train models,120000,YEARLY,Full-time,AI/ML,true\n",
        ],
    )
    analytics = MarketAnalytics(
        DatasetPaths(train_path=train_path, test_path=test_path)
    )

    work_types = analytics.get_work_type_distribution()

    assert work_types == [
        {"work_type": "Full-time", "total": 2, "percentage": 66.67},
        {"work_type": "Contract", "total": 1, "percentage": 33.33},
    ]


def test_yearly_salary_by_category_uses_only_yearly_salary_rows(
    tmp_path: Path,
) -> None:
    """Salary statistics should avoid mixing yearly and hourly salaries."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    write_processed_csv(
        train_path,
        [
            "1,Data Engineer,Build pipelines,100000,YEARLY,Full-time,Data Engineering,true\n",
            "2,Senior Data Engineer,Lead pipelines,140000,YEARLY,Full-time,Data Engineering,true\n",
            "3,Frontend Developer,Build UI,80,HOURLY,Contract,Frontend,true\n",
        ],
    )
    write_processed_csv(
        test_path,
        [
            "4,ML Engineer,Train models,120000,YEARLY,Full-time,AI/ML,true\n",
            "5,QA Engineer,Test products,,YEARLY,Full-time,QA/Testing,false\n",
        ],
    )
    analytics = MarketAnalytics(
        DatasetPaths(train_path=train_path, test_path=test_path)
    )

    salaries = analytics.get_yearly_salary_by_category()

    assert salaries == [
        {
            "job_category": "Data Engineering",
            "salary_postings": 2,
            "average_max_salary": 120000.0,
            "median_max_salary": 120000.0,
            "pay_period": "YEARLY",
        },
        {
            "job_category": "AI/ML",
            "salary_postings": 1,
            "average_max_salary": 120000.0,
            "median_max_salary": 120000.0,
            "pay_period": "YEARLY",
        },
    ]
