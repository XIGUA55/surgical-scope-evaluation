from scope_eval.dfine_adapter import coco_stats_to_metrics


def test_coco_stats_mapping():
    metrics = coco_stats_to_metrics([value / 10 for value in range(12)])
    assert metrics["map"] == 0
    assert metrics["map_75"] == 0.2
    assert metrics["mar_100"] == 0.8
    assert metrics["mar_large"] == 1.1
