"""Tests for small Streamlit frontend page helpers."""

from job_market_intelligence.frontend.api_client import ApiClientError
from job_market_intelligence.frontend.pages import chatbot, dashboard, profile_evaluator


class FakeExpander:
    """Context manager used to fake Streamlit expanders."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeStreamlit:
    """Small fake for Streamlit calls used by helper functions."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def caption(self, text: str) -> None:
        self.calls.append(("caption", text))

    def subheader(self, text: str) -> None:
        self.calls.append(("subheader", text))

    def info(self, text: str) -> None:
        self.calls.append(("info", text))

    def error(self, text: str) -> None:
        self.calls.append(("error", text))

    def expander(self, label: str) -> FakeExpander:
        self.calls.append(("expander", label))
        return FakeExpander()

    def text_area(
        self,
        label: str,
        value: str,
        height: int,
        disabled: bool,
        key: str,
    ) -> None:
        self.calls.append(("text_area", label, value, height, disabled, key))


def test_dashboard_formats_tables_for_display() -> None:
    """Dashboard table helpers should expose user-facing labels and values."""
    table = dashboard._format_table(
        data=[
            {
                "job_category": "Data Engineering",
                "total": 15315,
                "percentage": 28.345,
            }
        ],
        column_labels={
            "job_category": "Categoría",
            "total": "Ofertas",
            "percentage": "Porcentaje",
        },
        formatters={
            "total": dashboard._format_int,
            "percentage": dashboard._format_percentage,
        },
    )

    assert table.to_dict(orient="records") == [
        {
            "Categoría": "Data Engineering",
            "Ofertas": "15,315",
            "Porcentaje": "28.34%",
        }
    ]


def test_dashboard_formats_salary_table_without_pay_period() -> None:
    """Salary tables should hide constant pay-period data and format USD values."""
    table = dashboard._format_table(
        data=[
            {
                "job_category": "Data Engineering",
                "salary_postings": 1200,
                "average_max_salary": 166036.77,
                "median_max_salary": 145000.0,
                "pay_period": "YEARLY",
            }
        ],
        columns=[
            "job_category",
            "salary_postings",
            "average_max_salary",
            "median_max_salary",
        ],
        column_labels={
            "job_category": "Categoría",
            "salary_postings": "Ofertas con salario",
            "average_max_salary": "Media salario anual máximo",
            "median_max_salary": "Mediana salario anual máximo",
        },
        formatters={
            "salary_postings": dashboard._format_int,
            "average_max_salary": dashboard._format_currency,
            "median_max_salary": dashboard._format_currency,
        },
    )

    assert table.to_dict(orient="records") == [
        {
            "Categoría": "Data Engineering",
            "Ofertas con salario": "1,200",
            "Media salario anual máximo": "$166,037",
            "Mediana salario anual máximo": "$145,000",
        }
    ]


def test_dashboard_fetch_data_maps_api_errors(monkeypatch) -> None:
    """Fetch helper should convert API failures into an empty result for the UI."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(dashboard, "st", fake_st)

    result = dashboard._fetch_data(
        lambda: (_ for _ in ()).throw(ApiClientError("API no disponible"))
    )

    assert result == []
    assert fake_st.calls == [("error", "API no disponible")]


def test_profile_sources_render_full_descriptions(monkeypatch) -> None:
    """Profile sources should be rendered as clickable descriptions."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(profile_evaluator, "st", fake_st)

    profile_evaluator._render_sources(
        [
            {
                "job_id": 123,
                "title": "Data Engineer",
                "job_category": "Data Engineering",
                "score": 0.82345,
                "description": "Python and SQL role.",
            }
        ]
    )

    assert ("subheader", "Fuentes") in fake_st.calls
    assert (
        "expander",
        "1. Data Engineer · Data Engineering · similitud: 0.823",
    ) in fake_st.calls
    assert (
        "text_area",
        "Descripción",
        "Python and SQL role.",
        260,
        True,
        "profile-source-1-123",
    ) in fake_st.calls


def test_profile_sources_handle_empty_results(monkeypatch) -> None:
    """Profile source rendering should show an empty-state message."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(profile_evaluator, "st", fake_st)

    profile_evaluator._render_sources([])

    assert ("info", "La API no devolvió fuentes.") in fake_st.calls


def test_chatbot_sources_render_full_descriptions(monkeypatch) -> None:
    """Chatbot sources should include full descriptions and stable widget keys."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(chatbot, "st", fake_st)

    chatbot._render_sources(
        [
            {
                "job_id": 456,
                "title": "ML Engineer",
                "job_category": "AI/ML",
                "score": 0.91,
                "description": "Model deployment role.",
            }
        ],
        key_prefix="chat-history-0",
    )

    assert ("caption", "Fuentes") in fake_st.calls
    assert ("expander", "1. ML Engineer · AI/ML · similitud: 0.910") in fake_st.calls
    assert (
        "text_area",
        "Descripción",
        "Model deployment role.",
        260,
        True,
        "chat-history-0-source-1-456",
    ) in fake_st.calls
