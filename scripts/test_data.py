import unittest

from utils.data_utils import add_items_to_group


class TestMetrics(unittest.TestCase):
    def test_constructing_groups(self):
        groups = []
        add_items_to_group(['#1', '#2'], groups)
        add_items_to_group(['#3', '#4'], groups)

        assert len(groups) == 2

        add_items_to_group(['#5', '#6'], groups)

        assert len(groups) == 3

        add_items_to_group(['#2', '#6'], groups)

        assert len(groups) == 2

