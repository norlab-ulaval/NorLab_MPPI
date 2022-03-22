from src.barebones_mpc.selector.greedy_selector import GreedySelector

from src.barebones_mpc.selector.abstract_selector import AbstractSelector


# ... refactoring ......................................................................................................


def test_config_init(setup_mock_config_dict_CartPole):
    instance = GreedySelector.config_init(config=setup_mock_config_dict_CartPole)
    assert isinstance(instance, AbstractSelector)


def test_manual_init(arbitrary_size_manual_selector_10_5_1_2):
    instance = GreedySelector()
    assert isinstance(instance, AbstractSelector)


# ... implementation ...................................................................................................


def test_greedy_selector_select_arange(arbitrary_size_manual_selector_10_5_1_2):
    greedy_selector = GreedySelector()
    nominal_input, _ = greedy_selector.select_next_input(
        sample_states=arbitrary_size_manual_selector_10_5_1_2.sample_state,
        sample_inputs=arbitrary_size_manual_selector_10_5_1_2.sample_input,
        sample_costs=arbitrary_size_manual_selector_10_5_1_2.sample_cost,
    )
    selected_input = nominal_input[0]
    assert selected_input == 1
