from mbo.preproc import jaccard_sim


def test_jaccard_sim():
    assert jaccard_sim([{0, 1, 2}, {0, 1, 2}]) == 1
    assert jaccard_sim([{0, 1, 2}, {3, 4, 5}]) == 0
    assert jaccard_sim([{0, 1, 2, 3}, {0, 4, 5, 6, 7}]) == 0.125
