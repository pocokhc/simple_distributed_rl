from srl.rl.functions import create_fancy_index_for_invalid_actions


def test_create_fancy_index_for_invalid_actions():
    idx_list = [
        [1, 2, 5],
        [2],
        [2, 3],
    ]
    idx1, idx2 = create_fancy_index_for_invalid_actions(idx_list)
    print(idx1)
    print(idx2)
    assert idx1 == [0, 0, 0, 1, 2, 2]
    assert idx2 == [1, 2, 5, 2, 2, 3]
