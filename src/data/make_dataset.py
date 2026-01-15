# src/data/make_dataset.py
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 2. Path Definitions
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"


def load_raw_data():
    """Load raw datasets."""
    logger.info("⏳ Loading RAW files...")
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

        df_job_skills = pd.read_csv(
            RAW_DATA_DIR / "job_skills.csv", usecols=["job_id", "skill_abr"]
        )
        df_skills_map = pd.read_csv(
            RAW_DATA_DIR / "skills.csv", usecols=["skill_abr", "skill_name"]
        )
        df_salaries = pd.read_csv(
            RAW_DATA_DIR / "salaries.csv",
            usecols=["job_id", "max_salary", "pay_period"],
        )

        return df_postings, df_job_skills, df_skills_map, df_salaries
    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        raise


def process_data(df_postings, df_job_skills, df_skills_map, df_salaries):
    """Merge tables into unified dataset."""
    logger.info("🔗 Merging data...")

    # Skills
    df_full_skills = pd.merge(df_job_skills, df_skills_map, on="skill_abr", how="left")
    df_skills_grouped = (
        df_full_skills.groupby("job_id")["skill_name"].apply(list).reset_index()
    )
    df_skills_grouped.rename(columns={"skill_name": "skills"}, inplace=True)

    # Main Merge
    df_merged = pd.merge(df_postings, df_skills_grouped, on="job_id", how="left")

    # Salaries
    df_merged = pd.merge(
        df_merged, df_salaries, on="job_id", how="left", suffixes=("", "_ext")
    )
    df_merged["max_salary"] = df_merged["max_salary"].fillna(
        df_merged["max_salary_ext"]
    )
    df_merged["pay_period"] = df_merged["pay_period"].fillna(
        df_merged["pay_period_ext"]
    )
    df_merged.drop(columns=["max_salary_ext", "pay_period_ext"], inplace=True)

    return df_merged


def filter_ai_jobs(df):
    """
    Filter Generative AI and related job postings using REGEX.
    Searches both title and description for broader coverage.
    """
    logger.info("🔍 Filtering AI & Generative AI sector (Expanded Search)...")

    # 1. Inclusion Keywords - EXPANDED for Generative AI
    # Using \\b for word boundaries
    include_keywords_title = [
        # AI/ML General
        r"\bAI\b",
        r"\bArtificial Intelligence\b",
        r"\bMachine Learning\b",
        r"\bDeep Learning\b",
        r"\bML\b",
        # Data Science
        r"\bData Scientist\b",
        r"\bData Science\b",
        r"\bData Engineer\b",
        r"\bData Analyst\b",
        r"\bData Analytics\b",
        r"\bBig Data\b",
        r"\bAnalytics\b",
        # Generative AI (NEW - More specific)
        r"\bGenerative AI\b",
        r"\bGenAI\b",
        r"\bGen AI\b",
        r"\bLLM\b",
        r"\bLarge Language Model\b",
        r"\bGPT\b",
        r"\bChatGPT\b",
        r"\bOpenAI\b",
        r"\bPrompt Engineer\b",
        r"\bPrompt Engineering\b",
        r"\bAI Engineer\b",
        r"\bML Engineer\b",
        r"\bMLE\b",
        r"\bLangChain\b",
        r"\bTransformer\b",
        r"\bBERT\b",
        r"\bFoundation Model\b",
        r"\bDiffusion Model\b",
        r"\bStable Diffusion\b",
        r"\bMidjourney\b",
        r"\bDALL-E\b",
        r"\bClaude\b",
        r"\bGemini\b",
        r"\bCopilot\b",
        r"\bLlama\b",
        r"\bMistral\b",
        r"\bHugging Face\b",
        r"\bHuggingface\b",
        # NLP / Language Processing
        r"\bNLP\b",
        r"\bNatural Language Processing\b",
        r"\bNatural Language\b",
        r"\bConversational AI\b",
        r"\bChatbot\b",
        r"\bText Analytics\b",
        r"\bSentiment Analysis\b",
        # Computer Vision
        r"\bComputer Vision\b",
        r"\bCV Engineer\b",
        r"\bImage Recognition\b",
        r"\bObject Detection\b",
        # MLOps / Infrastructure
        r"\bMLOps\b",
        r"\bML Ops\b",
        r"\bAIOps\b",
        r"\bAI Ops\b",
        r"\bNeural Network\b",
        r"\bDeep Neural\b",
        # Other related roles
        r"\bRobotics\b",
        r"\bReinforcement Learning\b",
        r"\bRL Engineer\b",
        r"\bResearch Scientist\b",
        r"\bApplied Scientist\b",
    ]

    # More general keywords for description search
    include_keywords_desc = [
        r"\bGenerative AI\b",
        r"\bGenAI\b",
        r"\bLLM\b",
        r"\bLarge Language Model\b",
        r"\bGPT-3\b",
        r"\bGPT-4\b",
        r"\bChatGPT\b",
        r"\bOpenAI API\b",
        r"\bLangChain\b",
        r"\bPrompt Engineering\b",
        r"\bTransformer model\b",
        r"\bHugging Face\b",
        r"\bFine-tuning\b",
        r"\bRAG\b",
        r"\bRetrieval Augmented\b",
        r"\bVector Database\b",
        r"\bPinecone\b",
        r"\bChroma\b",
        r"\bWeaviate\b",
        r"\bEmbedding\b",
        r"\bSemantic Search\b",
        r"\bNeural Network\b",
        r"\bDeep Learning\b",
        r"\bMachine Learning\b",
        r"\bPyTorch\b",
        r"\bTensorFlow\b",
        r"\bKeras\b",
        r"\bScikit-learn\b",
        r"\bMLflow\b",
    ]

    # 2. PROHIBITED Keywords (Exclusion) - Adjusted to avoid over-exclusion
    exclude_keywords = [
        r"\bSales Representative\b",
        r"\bAccount Executive\b",
        r"\bRecruiter\b",
        r"\bHuman Resources\b",
        r"\bNurse\b",
        r"\bDriver\b",
        r"\bCafeteria\b",
        r"\bWarehouse Worker\b",
        r"\bFulfillment\b",
        r"\bStore Manager\b",
        r"\bCashier\b",
        r"\bMechanic\b",
        r"\bElectrician\b",
        r"\bPlumber\b",
        r"\bChef\b",
        r"\bCook\b",
        r"\bWaiter\b",
        r"\bCleaner\b",
        r"\bJanitor\b",
        r"\bCustomer Service Rep\b",
        r"\bProperty Manager\b",
        r"\bTeller\b",
        r"\bRetail\b",
    ]

    # Combine with OR (|) for regex
    regex_include_title = "|".join(include_keywords_title)
    regex_include_desc = "|".join(include_keywords_desc)
    regex_exclude = "|".join(exclude_keywords)

    # Step A: Inclusion by title
    mask_include_title = df["title"].str.contains(
        regex_include_title, case=False, na=False, regex=True
    )

    # Step B: Inclusion by description (to capture more GenAI postings)
    mask_include_desc = df["description"].str.contains(
        regex_include_desc, case=False, na=False, regex=True
    )

    # Step C: Exclusion by title
    mask_exclude = df["title"].str.contains(
        regex_exclude, case=False, na=False, regex=True
    )

    # Logic: (Match in title OR Match in description) AND (NOT junk)
    df_clean = df[(mask_include_title | mask_include_desc) & (~mask_exclude)].copy()

    logger.info(f"📊 Total original postings: {len(df)}")
    logger.info(f"   - Match by title: {mask_include_title.sum()}")
    logger.info(f"   - Match by description: {mask_include_desc.sum()}")
    logger.info(f"   - Excluded: {mask_exclude.sum()}")
    logger.info(f"📉 Final result: {len(df_clean)} clean Generative AI job postings.")
    return df_clean


def clean_data(df):
    """Basic data cleaning."""
    initial_len = len(df)

    # 1. Remove exact duplicates
    df = df.drop_duplicates(subset=["job_id"])

    # 2. Remove postings without description (useless for RAG)
    df = df.dropna(subset=["description"])

    # 3. Filter very short descriptions (<100 chars = probably junk)
    df = df[df["description"].str.len() >= 100]

    logger.info(f"🧹 Cleaning: Removed {initial_len - len(df)} dirty rows.")
    return df


def main():
    logger.info("🚀 Starting Improved Pipeline for Generative AI...")

    # 1. Load and Merge
    df_p, df_js, df_sm, df_s = load_raw_data()
    df = process_data(df_p, df_js, df_sm, df_s)

    # 2. Filter and Clean
    df = filter_ai_jobs(df)
    df = clean_data(df)

    # 3. Train/Test Split (80% / 20%)
    logger.info("✂️ Splitting into Train/Test...")
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # 4. Save
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    logger.info(f"💾 Saved to {PROCESSED_DATA_DIR}")
    logger.info(f"   Train: {len(train)} | Test: {len(test)}")
    logger.info(f"   Total: {len(train) + len(test)} Generative AI job postings")


if __name__ == "__main__":
    main()
