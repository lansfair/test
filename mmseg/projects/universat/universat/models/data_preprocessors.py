"""Data preprocessor for multimodal UniverSat inputs."""

from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


@MODELS.register_module()
class UniverSatDataPreprocessor(BaseDataPreprocessor):
    """Preprocessor that passes through a dict of modality tensors.

    Unlike the standard ``SegDataPreProcessor``, which expects a single image
    tensor, UniverSat receives a dict ``{modality: tensor}``. This preprocessor
    only casts tensors to the model's device/dtype and leaves the dict
    structure untouched.
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Forward function.

        Args:
            data: dict with ``inputs`` (dict of tensors) and optionally
                ``data_samples``.
            training: Whether in training mode.

        Returns:
            dict: The preprocessed data.

        Note:
            ``BaseDataPreprocessor.cast_data`` only moves tensors to the target
            device and preserves their dtypes. This keeps date tensors as
            ``long`` while floating-point modality tensors stay in the model's
            dtype.
        """
        return self.cast_data(data)
