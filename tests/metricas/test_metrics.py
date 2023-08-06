import torch

from utils.metrics_utils import create_metrics
from utils.image_utils import upload_images_numpy


def test_metrics_not_zero():
    """
    This test checks that the metrics are correctly calculated,
    since images are not the same metrics should not be zero

    """

    metricHandler = create_metrics("FID")
    metricAB = metricHandler.calculate_metrics(
        "testA2testB", (256, 256), out_domain="B"
    )
    metricBA = metricHandler.calculate_metrics(
        "testA2testB", (256, 256), out_domain="A"
    )

    assert metricAB > 0 and metricBA > 0


def test_metrics_score_zero():
    """
    Check that metrics calculated with the same twp datasets
    are equal to zero since there should not be a difference in
    the inception vectors
    """
    metricHandler = create_metrics("FID")
    name = "testA2testB"
    im_size = (256, 256)
    domain = "B"
    tolerance = 1e-4

    dataA = upload_images_numpy(
        f"./images/{name}/test_{domain}/{domain}/", im_size=im_size
    )
    dataA = torch.from_numpy(dataA).permute(0, 3, 1, 2)

    dataB = dataA

    metricsAB = metricHandler._get_score(dataA, dataB)
    metricsBA = metricHandler._get_score(dataB, dataA)

    assert metricsAB < tolerance
    assert metricsAB - metricsBA < tolerance
