from api import BaseInput

from ...impl.ncnn.model import NcnnModelWrapper


class NcnnModelInput(BaseInput):
    """Input for NcnnModel"""

    def __init__(self, label: str = "Model") -> None:
        super().__init__("NcnnNetwork", label)
        self.associated_type = NcnnModelWrapper
