from spiderSolitaire import SpiderSolitaire
from spiderSolitaireEnv import SpiderSolitaireEnv


def test_decode_action():
    env = SpiderSolitaireEnv()

    expected_combinations = set()
    for i in range(10):
        for j in range(10):
            if i != j:
                expected_combinations.add((i, j))

    generated_combinations = set()
    for action in range(90):  # Actions from 0 to 89
        from_pile, to_pile = env._decode_action(action)
        generated_combinations.add((from_pile, to_pile))

    assert (
        expected_combinations == generated_combinations
    ), "Not all pile combinations are covered"

    print("All pile combinations are correctly generated by decode_action.")


# Run the test
test_decode_action()
