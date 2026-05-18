"""ETL pipeline for tech job market dataset.

Processes raw LinkedIn data into train/test splits with engineered features.
Filters for technology sector roles and categorizes them by domain.
"""

import logging
import re
from pathlib import Path

import pandas as pd
from categories import CATEGORY_PRIORITY, EXCLUDE_KEYWORDS, JOB_CATEGORIES
from sklearn.model_selection import train_test_split

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Path Definitions
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw datasets from CSV files."""
    logger.info("Loading raw files...")
    try:
        cols_postings = [
            "job_id",
            "title",
            "description",
            "max_salary",
            "pay_period",
            "formatted_work_type",
        ]
        df_postings = pd.read_csv(RAW_DATA_DIR / "postings.csv", usecols=cols_postings)

        df_salaries = pd.read_csv(
            RAW_DATA_DIR / "salaries.csv",
            usecols=["job_id", "max_salary", "pay_period"],
        )

        return df_postings, df_salaries

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        raise


def process_data(df_postings: pd.DataFrame, df_salaries: pd.DataFrame) -> pd.DataFrame:
    """Merge postings with salary data into a unified dataset."""
    logger.info("Merging data...")

    # Enrich postings with salary data from the separate salaries table
    df_merged = pd.merge(
        df_postings, df_salaries, on="job_id", how="left", suffixes=("", "_ext")
    )
    df_merged["max_salary"] = df_merged["max_salary"].fillna(
        df_merged["max_salary_ext"]
    )
    df_merged["pay_period"] = df_merged["pay_period"].fillna(
        df_merged["pay_period_ext"]
    )
    df_merged.drop(columns=["max_salary_ext", "pay_period_ext"], inplace=True)

    return df_merged


def _assign_categories_vectorized(df: pd.DataFrame) -> pd.Series:
    """Assign job categories using vectorized string matching.

    Each category is checked in priority order. Once a row is assigned,
    it is not overwritten by lower-priority categories.
    """
    result = pd.Series("Other Tech", index=df.index)
    assigned = pd.Series(False, index=df.index)

    for category in CATEGORY_PRIORITY:
        config = JOB_CATEGORIES[category]
        title_pattern = "|".join(config["title"]) if config["title"] else None
        desc_pattern = "|".join(config["desc"]) if config["desc"] else None

        match_mask = pd.Series(False, index=df.index)
        if title_pattern:
            match_mask |= df["title"].str.contains(
                title_pattern, case=False, na=False, regex=True
            )
        if desc_pattern:
            match_mask |= df["description"].str.contains(
                desc_pattern, case=False, na=False, regex=True
            )

        # Only assign to rows not yet categorized
        new_matches = match_mask & ~assigned
        result[new_matches] = category
        assigned |= new_matches

    return result


def filter_tech_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Filter technology sector job postings and assign categories.

    Matches against all defined tech categories by title and description.
    Excludes clearly non-tech roles.
    """
    logger.info("Filtering tech sector jobs...")

    # Build combined inclusion pattern from all categories
    all_title_keywords: list[str] = []
    all_desc_keywords: list[str] = []
    for config in JOB_CATEGORIES.values():
        all_title_keywords.extend(config["title"])
        all_desc_keywords.extend(config["desc"])

    regex_include_title = "|".join(all_title_keywords)
    regex_include_desc = "|".join(all_desc_keywords)
    regex_exclude = "|".join(EXCLUDE_KEYWORDS)

    mask_include_title = df["title"].str.contains(
        regex_include_title, case=False, na=False, regex=True
    )
    mask_include_desc = df["description"].str.contains(
        regex_include_desc, case=False, na=False, regex=True
    )
    mask_exclude = df["title"].str.contains(
        regex_exclude, case=False, na=False, regex=True
    )

    df_filtered = df[(mask_include_title | mask_include_desc) & (~mask_exclude)].copy()

    logger.info(
        f"Filtered {len(df)} -> {len(df_filtered)} jobs "
        f"(title: {mask_include_title.sum()}, desc: {mask_include_desc.sum()}, "
        f"excluded: {mask_exclude.sum()})"
    )

    # Assign categories (vectorized for performance)
    logger.info("Assigning job categories...")
    df_filtered["job_category"] = _assign_categories_vectorized(df_filtered)

    category_counts = df_filtered["job_category"].value_counts()
    logger.info(f"Category distribution:\n{category_counts.to_string()}")

    return df_filtered


def normalize_dedup_text(text: str) -> str:
    """Normalize text for conservative repost detection."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and invalid entries."""
    initial_len = len(df)

    df = df.drop_duplicates(subset=["job_id"])
    df = df.dropna(subset=["description"])
    df = df[df["description"].str.len() >= 100]

    df = df.copy()
    # Dedup against reposts that only differ in spacing, punctuation, or numeric requisition IDs.
    df["title_norm"] = df["title"].fillna("").map(normalize_dedup_text)
    df["description_norm"] = df["description"].map(normalize_dedup_text)
    duplicate_text_rows = int(
        df.duplicated(subset=["title_norm", "description_norm"]).sum()
    )
    duplicate_text_groups = int(
        (
            df.groupby(["title_norm", "description_norm"], dropna=False).size().gt(1)
        ).sum()
    )
    df = df.drop_duplicates(subset=["title_norm", "description_norm"]).drop(
        columns=["title_norm", "description_norm"]
    )

    logger.info(
        "Cleaned dataset: %s rows removed (%s text duplicates in %s groups)",
        initial_len - len(df),
        duplicate_text_rows,
        duplicate_text_groups,
    )

    return df


def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create analytical features kept for EDA and dashboard views."""
    logger.info("Creating features...")

    df["has_salary"] = df["max_salary"].notna().astype(int)

    logger.info(
        f"Features created - with_salary: {df['has_salary'].mean() * 100:.1f}%"
    )

    return df


def main() -> None:
    """Run the complete ETL pipeline."""
    logger.info("Starting pipeline...")

    df_p, df_s = load_raw_data()
    df = process_data(df_p, df_s)
    df = filter_tech_jobs(df)
    df = clean_data(df)
    df = create_basic_features(df)

    logger.info("Splitting into train/test sets...")
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    logger.info(f"Saved to {PROCESSED_DATA_DIR}")
    logger.info(f"Train: {len(train)} | Test: {len(test)} | Total: {len(df)}")
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
