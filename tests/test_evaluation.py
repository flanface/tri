from types import SimpleNamespace as NS

from tri import evaluation as E


class _A:
    def __init__(self, v):
        self.value = v


def _review(rid, bid, ts, adds=(), rems=()):
    ops = [NS(item_id=i, action=_A("add")) for i in adds] + [NS(item_id=i, action=_A("remove")) for i in rems]
    return NS(review_id=rid, timestamp=ts, operations=ops, original=NS(build_id=bid))


def test_score_and_metrics():
    r = _review("r", "b", 1, adds=["a", "b"], rems=["x"])
    row = E.score([dict(item="a", action="add"), dict(item="c", action="add"), dict(item="x", action="remove")], r)
    assert row == dict(tp_add=1, fp_add=1, tp_rem=1, fp_rem=0, n_add=2, n_rem=1)
    m = E.metrics([row])
    assert m["add_precision"] == 0.5 and m["remove_precision"] == 1.0
    assert abs(m["recall"] - 2 / 3) < 1e-9


def test_units_newest_truth_and_exclusion():
    rs = [_review("r1", "b1", 1), _review("r2", "b1", 2), _review("r3", "b2", 1), _review("r4", "b3", 1)]
    near = {frozenset(("r2", "r3"))}
    sim = lambda a, b: 0.97 if frozenset((a, b)) in near else 0.1
    us = {u.build_id: u for u in E.units(rs, sim)}
    assert us["b1"].truth.review_id == "r2"
    assert us["b1"].exclude == ["r1", "r2", "r3"]
    assert us["b3"].exclude == ["r4"]


def test_bootstrap_has_se():
    rows = [dict(tp_add=i % 2, fp_add=1, tp_rem=1, fp_rem=i % 3, n_add=2, n_rem=2) for i in range(30)]
    b = E.bootstrap(rows, B=200)
    assert b["f1"]["se"] > 0
    g = E.paired_gain(rows, rows, B=50)
    assert g["gain"] == 0 and g["se"] == 0
