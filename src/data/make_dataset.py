"""ETL pipeline for tech job market dataset.

Processes raw LinkedIn data into train/test splits with engineered features.
Filters for technology sector roles and categorizes them by domain.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"


# ---------------------------------------------------------------------------
# Job category definitions: each category has title and description keywords
# ---------------------------------------------------------------------------
JOB_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "AI/ML": {
        "title": [
            r"\bAI\b",
            r"\bArtificial Intelligence\b",
            r"\bMachine Learning\b",
            r"\bDeep Learning\b",
            r"\bML\b",
            r"\bData Scientist\b",
            r"\bData Science\b",
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
            r"\bNLP\b",
            r"\bNatural Language Processing\b",
            r"\bComputer Vision\b",
            r"\bCV Engineer\b",
            r"\bMLOps\b",
            r"\bML Ops\b",
            r"\bAIOps\b",
            r"\bNeural Network\b",
            r"\bRobotics\b",
            r"\bReinforcement Learning\b",
            r"\bRL Engineer\b",
            r"\bResearch Scientist\b",
            r"\bApplied Scientist\b",
            r"\bConversational AI\b",
            r"\bChatbot\b",
        ],
        "desc": [
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
            r"\bEmbedding\b",
            r"\bNeural Network\b",
            r"\bDeep Learning\b",
            r"\bPyTorch\b",
            r"\bTensorFlow\b",
            r"\bKeras\b",
            r"\bScikit-learn\b",
            r"\bMLflow\b",
        ],
    },
    "Data Engineering": {
        "title": [
            r"\bData Engineer\b",
            r"\bData Architect\b",
            r"\bData Platform\b",
            r"\bETL\b",
            r"\bData Pipeline\b",
            r"\bBig Data\b",
            r"\bData Infrastructure\b",
        ],
        "desc": [
            r"\bApache Spark\b",
            r"\bApache Kafka\b",
            r"\bApache Airflow\b",
            r"\bSnowflake\b",
            r"\bDatabricks\b",
            r"\bRedshift\b",
            r"\bBigQuery\b",
            r"\bETL pipeline\b",
            r"\bdata warehouse\b",
            r"\bdata lake\b",
            r"\bdbt\b",
        ],
    },
    "Data Analytics": {
        "title": [
            r"\bData Analyst\b",
            r"\bData Analytics\b",
            r"\bBusiness Analyst\b",
            r"\bBI Developer\b",
            r"\bBI Analyst\b",
            r"\bBusiness Intelligence\b",
            r"\bAnalytics Engineer\b",
        ],
        "desc": [
            r"\bTableau\b",
            r"\bPower BI\b",
            r"\bLooker\b",
            r"\bdata visualization\b",
        ],
    },
    "Backend": {
        "title": [
            r"\bBackend\b",
            r"\bBack-End\b",
            r"\bBack End\b",
            r"\bServer.Side\b",
            r"\bAPI Developer\b",
            r"\bPython Developer\b",
            r"\bJava Developer\b",
            r"\bGo Developer\b",
            r"\bNode\.js Developer\b",
            r"\b\.NET Developer\b",
            r"\bRuby Developer\b",
            r"\bPHP Developer\b",
        ],
        "desc": [
            r"\bREST API\b",
            r"\bGraphQL\b",
            r"\bFastAPI\b",
            r"\bDjango\b",
            r"\bFlask\b",
            r"\bSpring Boot\b",
            r"\bmicroservices\b",
            r"\bserver-side\b",
        ],
    },
    "Frontend": {
        "title": [
            r"\bFrontend\b",
            r"\bFront-End\b",
            r"\bFront End\b",
            r"\bUI Developer\b",
            r"\bUI Engineer\b",
            r"\bReact Developer\b",
            r"\bAngular Developer\b",
            r"\bVue Developer\b",
            r"\bWeb Developer\b",
        ],
        "desc": [
            r"\bReact\b",
            r"\bAngular\b",
            r"\bVue\.js\b",
            r"\bTypeScript\b",
            r"\bJavaScript\b",
            r"\bCSS\b",
            r"\bHTML\b",
            r"\bNext\.js\b",
            r"\bfrontend framework\b",
        ],
    },
    "Full Stack": {
        "title": [
            r"\bFull.Stack\b",
            r"\bFullstack\b",
            r"\bFull Stack\b",
        ],
        "desc": [],
    },
    "DevOps/SRE": {
        "title": [
            r"\bDevOps\b",
            r"\bSRE\b",
            r"\bSite Reliability\b",
            r"\bPlatform Engineer\b",
            r"\bInfrastructure Engineer\b",
            r"\bRelease Engineer\b",
            r"\bBuild Engineer\b",
        ],
        "desc": [
            r"\bKubernetes\b",
            r"\bDocker\b",
            r"\bTerraform\b",
            r"\bAnsible\b",
            r"\bCI/CD\b",
            r"\bJenkins\b",
            r"\bGitHub Actions\b",
            r"\bArgoCD\b",
            r"\binfrastructure as code\b",
        ],
    },
    "Cloud": {
        "title": [
            r"\bCloud Engineer\b",
            r"\bCloud Architect\b",
            r"\bCloud Developer\b",
            r"\bCloud Consultant\b",
            r"\bAWS\b",
            r"\bAzure\b",
            r"\bGCP\b",
        ],
        "desc": [
            r"\bAWS\b",
            r"\bAzure\b",
            r"\bGoogle Cloud\b",
            r"\bGCP\b",
            r"\bcloud infrastructure\b",
            r"\bcloud-native\b",
            r"\bserverless\b",
            r"\bLambda\b",
        ],
    },
    "Cybersecurity": {
        "title": [
            r"\bCybersecurity\b",
            r"\bCyber Security\b",
            r"\bSecurity Engineer\b",
            r"\bSecurity Analyst\b",
            r"\bInformation Security\b",
            r"\bInfoSec\b",
            r"\bPenetration Tester\b",
            r"\bPentest\b",
            r"\bSOC Analyst\b",
            r"\bSecurity Architect\b",
            r"\bAppSec\b",
        ],
        "desc": [
            r"\bcybersecurity\b",
            r"\bpenetration testing\b",
            r"\bvulnerability\b",
            r"\bSIEM\b",
            r"\bincident response\b",
            r"\bthreat detection\b",
        ],
    },
    "QA/Testing": {
        "title": [
            r"\bQA\b",
            r"\bQuality Assurance\b",
            r"\bTest Engineer\b",
            r"\bSDET\b",
            r"\bTest Automation\b",
            r"\bQA Engineer\b",
            r"\bQA Analyst\b",
        ],
        "desc": [
            r"\bSelenium\b",
            r"\bCypress\b",
            r"\btest automation\b",
            r"\bquality assurance\b",
            r"\bunit testing\b",
            r"\bmanual testing\b",
        ],
    },
    "Mobile": {
        "title": [
            r"\bMobile Developer\b",
            r"\bMobile Engineer\b",
            r"\biOS Developer\b",
            r"\bAndroid Developer\b",
            r"\bFlutter Developer\b",
            r"\bReact Native\b",
            r"\bSwift Developer\b",
            r"\bKotlin Developer\b",
        ],
        "desc": [
            r"\biOS\b",
            r"\bAndroid\b",
            r"\bFlutter\b",
            r"\bReact Native\b",
            r"\bSwift\b",
            r"\bKotlin\b",
            r"\bmobile app\b",
        ],
    },
    "Software Engineer": {
        "title": [
            r"\bSoftware Engineer\b",
            r"\bSoftware Developer\b",
            r"\bSoftware Architect\b",
            r"\bPrincipal Engineer\b",
            r"\bStaff Engineer\b",
            r"\bSystems Engineer\b",
            r"\bApplication Developer\b",
        ],
        "desc": [],
    },
}

# Roles to exclude (clearly non-tech)
EXCLUDE_KEYWORDS: list[str] = [
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
    r"\bPharmacist\b",
    r"\bDentist\b",
    r"\bTherapist\b",
    r"\bTeacher\b",
    r"\bPrincipal\b",
    r"\bConstruction\b",
    r"\bCarpenter\b",
]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        logger.error(f"Error: {e}")
        raise


def process_data(
    df_postings: pd.DataFrame,
    df_job_skills: pd.DataFrame,
    df_skills_map: pd.DataFrame,
    df_salaries: pd.DataFrame,
) -> pd.DataFrame:
    """Merge tables into unified dataset."""
    logger.info("Merging data...")

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


# Priority order for category assignment: most specific first, generic last
_CATEGORY_PRIORITY: list[str] = [
    "AI/ML",
    "Data Engineering",
    "Data Analytics",
    "Cybersecurity",
    "DevOps/SRE",
    "Cloud",
    "QA/Testing",
    "Mobile",
    "Frontend",
    "Backend",
    "Full Stack",
    "Software Engineer",
]


def _assign_categories_vectorized(df: pd.DataFrame) -> pd.Series:
    """Assign job categories using vectorized string matching.

    Each category is checked in priority order. Once a row is assigned,
    it is not overwritten by lower-priority categories.
    """
    result = pd.Series("Other Tech", index=df.index)
    assigned = pd.Series(False, index=df.index)

    for category in _CATEGORY_PRIORITY:
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and invalid entries."""
    initial_len = len(df)

    df = df.drop_duplicates(subset=["job_id"])
    df = df.dropna(subset=["description"])
    df = df[df["description"].str.len() >= 100]

    logger.info(f"Cleaned dataset: removed {initial_len - len(df)} rows")
    return df


def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features."""
    logger.info("Creating features...")

    df["skill_count"] = df["skills"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df["desc_length"] = df["description"].str.split().str.len()
    df["has_salary"] = df["max_salary"].notna().astype(int)

    logger.info(
        f"Features created - avg skills: {df['skill_count'].mean():.1f}, "
        f"avg desc_length: {df['desc_length'].mean():.0f}, "
        f"with_salary: {df['has_salary'].mean()*100:.1f}%"
    )

    return df


def main() -> None:
    """Run the complete ETL pipeline."""
    logger.info("Starting pipeline...")

    df_p, df_js, df_sm, df_s = load_raw_data()
    df = process_data(df_p, df_js, df_sm, df_s)
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
