from scope_eval.tracking import IoUTracker


def test_iou_tracker_preserves_and_expires_identity():
    tracker = IoUTracker(iou_threshold=0.2, maximum_age=1)
    first = tracker.update([[10, 10, 20, 20]])
    second = tracker.update([[11, 10, 21, 20]])
    tracker.update([])
    tracker.update([])
    third = tracker.update([[10, 10, 20, 20]])
    assert first[0] == second[0]
    assert third[0] != first[0]
