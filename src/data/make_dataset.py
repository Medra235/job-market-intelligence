# src/data/make_dataset.py
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Configuración de Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 2. Definición de Rutas
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"


def load_raw_data():
    """Carga los datasets crudos."""
    logger.info("⏳ Cargando archivos RAW...")
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
    """Unifica las tablas (Merge)."""
    logger.info("🔗 Unificando datos...")

    # Skills
    df_full_skills = pd.merge(df_job_skills, df_skills_map, on="skill_abr", how="left")
    df_skills_grouped = (
        df_full_skills.groupby("job_id")["skill_name"].apply(list).reset_index()
    )
    df_skills_grouped.rename(columns={"skill_name": "skills"}, inplace=True)

    # Merge Principal
    df_merged = pd.merge(df_postings, df_skills_grouped, on="job_id", how="left")

    # Salarios
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
    Filtra ofertas de IA/Data usando REGEX para evitar falsos positivos
    (ej: evitar que 'Retail' coincida con 'AI').
    """
    logger.info("🔍 Filtrando sector AI & Data (Modo Regex Blindado)...")

    # 1. Palabras clave de Inclusión (Usamos \b para límites de palabra completa)
    # \bAI\b significa: "AI" aislado (no dentro de Retail)
    include_keywords = [
        r"\bAI\b",
        r"\bArtificial Intelligence\b",
        r"\bMachine Learning\b",
        r"\bDeep Learning\b",
        r"\bData Scientist\b",
        r"\bData Science\b",
        r"\bNLP\b",
        r"\bLLM\b",
        r"\bGenerative AI\b",
        r"\bComputer Vision\b",
        r"\bMLOps\b",
        r"\bNeural Networks\b",
        r"\bGPT\b",
        r"\bBig Data\b",
        r"\bData Engineer\b",
        r"\bRobotics\b",
        r"\bData Analyst\b",  # Analyst solo si lleva Data delante
        r"\bAnalytics\b",
    ]

    # 2. Palabras PROHIBIDAS (Exclusión)
    exclude_keywords = [
        "Sales",
        "Account Executive",
        "Representative",
        "Recruiter",
        "HR",
        "Human Resources",
        "Legal",
        "Finance",
        "Nurse",
        "Driver",
        "Cafeteria",
        "Warehouse",
        "Fulfillment",
        "Store",
        "Cashier",
        "Maintenance",
        "Mechanic",
        "Electrician",
        "Plumber",
        "Chef",
        "Cook",
        "Waiter",
        "Cleaner",
        "Janitor",
        "Technician",
        "Customer Service",
        "Property Manager",
        "Banker",
        "Merchandiser",
        "Detailer",
        "Liaison",
    ]

    # Unimos con OR (|) para regex
    regex_include = "|".join(include_keywords)
    regex_exclude = "|".join(exclude_keywords)

    # Paso A: Inclusión (regex=True es vital aquí)
    mask_include = df["title"].str.contains(
        regex_include, case=False, na=False, regex=True
    )

    # Paso B: Exclusión
    mask_exclude = df["title"].str.contains(
        regex_exclude, case=False, na=False, regex=True
    )

    # Lógica: (Es AI) Y (NO es basura)
    df_clean = df[mask_include & (~mask_exclude)].copy()

    logger.info(f"📉 Filtrado: {len(df)} -> {len(df_clean)} ofertas limpias de IA.")
    return df_clean


def clean_data(df):
    """Limpieza básica"""
    initial_len = len(df)

    # 1. Eliminar duplicados exactos
    df = df.drop_duplicates(subset=["job_id"])

    # 2. Eliminar ofertas sin descripción (inútiles para RAG)
    df = df.dropna(subset=["description"])

    logger.info(f"🧹 Limpieza: Se eliminaron {initial_len - len(df)} filas sucias.")
    return df


def main():
    logger.info("🚀 Iniciando Pipeline...")

    # 1. Carga y Merge
    df_p, df_js, df_sm, df_s = load_raw_data()
    df = process_data(df_p, df_js, df_sm, df_s)

    # 2. Filtrado y Limpieza
    df = filter_ai_jobs(df)
    df = clean_data(df)

    # 3. División Train/Test (80% / 20%)
    logger.info("✂️ Dividiendo en Train/Test...")
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # 4. Guardado
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    logger.info(f"💾 Guardado en {PROCESSED_DATA_DIR}")
    logger.info(f"   Train: {len(train)} | Test: {len(test)}")


if __name__ == "__main__":
    main()
