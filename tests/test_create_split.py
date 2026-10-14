
import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "create_split.py"
)

spec = importlib.util.spec_from_file_location(
    "create_split", SCRIPT_PATH
)
create_split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(create_split)


def test_split_counts_cover_all_samples():
    for count in range(1, 101):
        allocations = create_split._split_counts(count)
        assert sum(allocations.values()) == count


def test_split_counts_never_are_negative():
    for count in range(1, 101):
        allocations = create_split._split_counts(count)
        assert all(value >= 0 for value in allocations.values())


def test_stable_order_key_is_reproducible():
    first = create_split._stable_order_key("1", "image-001")
    second = create_split._stable_order_key("1", "image-001")
    assert first == second


def test_stable_order_key_changes_with_image_id():
    first = create_split._stable_order_key("1", "image-001")
    second = create_split._stable_order_key("1", "image-002")
    assert first != second
