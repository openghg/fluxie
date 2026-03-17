import pytest

from fluxie.plots.utils import parse_include


@pytest.mark.parametrize(
    ("include", "expected"),
    [
        # Simple str
        ("mf_observed", {"mf_observed": None}),
        (
            # List of str
            ["mf_observed", "mf_posterior"],
            {"mf_observed": None, "mf_posterior": None},
        ),
        (
            # Dict with str values
            {
                "mf_observed": None,
                "mf_posterior": "percentile_mf_posterior",
            },
            {
                "mf_observed": None,
                "mf_posterior": "percentile_mf_posterior",
            },
        ),
        (
            # Other values
            {"a": "b", "c": None},
            {"a": "b", "c": None},
        ),
    ],
)
def test_parse_include(include, expected):
    assert parse_include(include) == expected
