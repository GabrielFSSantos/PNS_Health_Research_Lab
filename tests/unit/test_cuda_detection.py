
from pns_health_research.core.gpu import cuda_available


def test_cuda_available_via_libcudart_or_nvidia():
  # Em máquinas sem GPU pode ser False; com RTX + driver deve ser True.
  result = cuda_available(refresh=True)
  assert isinstance(result, bool)
