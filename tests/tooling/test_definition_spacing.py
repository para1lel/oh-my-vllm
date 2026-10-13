import pytest

from scripts.check_definition_spacing import python_spacing_gaps


def test_decorators_and_attached_comments_stay_with_their_function():
    source = (
        "def first():\n    return 'def fake(): pass'\n"
        "# Second operation.\n@decorator\ndef second():\n    pass\n"
    )
    assert python_spacing_gaps(source) == {2: 2}


def test_methods_nested_functions_and_import_groups_have_distinct_spacing():
    source = (
        "import os\nimport sys\n\n\nclass Worker:\n"
        "    def first(self):\n        def nested():\n            pass\n"
        "        def other():\n            pass\n"
        "        return nested\n\n"
        "    async def second(self):\n        pass\n"
    )
    assert python_spacing_gaps(source) == {8: 1, 10: 1}


@pytest.mark.parametrize(
    "prefix", ["if True:\n    pass\nelse:\n", "try:\n    pass\nfinally:\n"]
)
def test_definitions_in_alternative_and_finalization_branches(prefix):
    source = (
        prefix + "    def first():\n        pass\n    def second():\n        pass\n"
    )
    assert python_spacing_gaps(source) == {5: 1}
