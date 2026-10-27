"""CaptureCheck: predict 3D Gaussian Splatting quality before training.

The pipeline is: load a capture (video or image bundle) -> classical OpenCV
measurements -> DINOv2 embeddings -> rule-based labels with evidence -> a small
trained model that turns the measurements into a quality score.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
