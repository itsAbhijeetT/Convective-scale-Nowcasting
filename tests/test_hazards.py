import pytest
import numpy as np
from hazards.hazards import dbz_to_rain_rate, detect_hazards
from eval.metrics import eval_metrics

def test_dbz_to_rain_rate():
    rr = dbz_to_rain_rate(40)
    assert rr > 0

def test_eval_metrics():
    obs = np.array([[40, 40], [0, 0]])
    pred = np.array([[40, 0], [40, 0]])
    pod, far, csi, bias = eval_metrics(obs, pred, threshold=35.0)
    assert pod == 0.5
    assert far == 0.5
    assert csi == 1/3
    assert bias == 1.0
