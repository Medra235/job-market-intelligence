"""Tests for adapting profile text to the classical classifier."""

import pytest

from job_market_intelligence.models.profile_classifier import build_profile_feature_frame


def test_build_profile_feature_frame_matches_training_feature_schema() -> None:
    """Profile text should be adapted to the existing sklearn pipeline schema."""
    features = build_profile_feature_frame(profile_text="Python, SQL, pandas and basic AWS.")

    assert list(features.columns) == ["text_combined"]
    assert features.loc[0, "text_combined"].startswith("User profile Python")


def test_build_profile_feature_frame_uses_generic_title() -> None:
    """The adapter should not use the user-declared target role as model evidence."""
    features = build_profile_feature_frame(profile_text="React and TypeScript.")

    assert features.loc[0, "text_combined"] == "User profile React and TypeScript."


@pytest.mark.parametrize("profile_text", ["", "   "])
def test_build_profile_feature_frame_rejects_empty_profile(profile_text: str) -> None:
    """Empty profile text should fail before model inference."""
    with pytest.raises(ValueError, match="Profile text cannot be empty"):
        build_profile_feature_frame(profile_text=profile_text)
