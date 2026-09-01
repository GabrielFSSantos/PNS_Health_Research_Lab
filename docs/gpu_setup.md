# GPU setup (research_v2)

Backend opcional `classification.backend: gpu` para o perfil `research_v2`. **Não** reproduz o sklearn do TCC.

## Requisitos

| Componente | Versão / nota |
|------------|----------------|
| Python | **3.10–3.12** para cuML (não suporta 3.14) |
| CUDA | Compatível com RAPIDS/cuML instalado |
| Extra pip | `pip install -e ".[gpu]"` |

### Instalar Python 3.12 no WSL/Ubuntu (se ausente)

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update && sudo apt install python3.12 python3.12-venv
make install-gpu   # usa python3.12 automaticamente se estiver no PATH
```

Se `cuml-cu12` falhar após o venv:

```bash
source .venv-gpu/bin/activate
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

## Ambiente recomendado

O `.venv` padrão do projeto pode usar Python 3.14. Para GPU, use o target Makefile:

```bash
make install-gpu    # cria .venv-gpu e instala [gpu,dev]
make test-gpu       # pytest -m gpu
make run-v2-gpu     # experiment research_v2 com backend gpu
```

Ou manualmente:

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -e ".[gpu,dev]"
```

Verifique CUDA:

```bash
nvidia-smi
python -c "from pns_health_research.core.gpu import cuda_available; print(cuda_available())"
```

A detecção usa `libcudart` (não só `nvidia-smi`), pois alguns ambientes — inclusive o sandbox do Cursor — bloqueiam `nvidia-smi` mesmo com GPU disponível.

## Configuração

Em `configs/experiments/research_v2.yaml`:

```yaml
classification:
  backend: gpu
  device: cuda
  parallel: true
  max_workers: 4
  models:
    - random_forest   # cuML
    - svc             # cuML SVM
    - mlp             # PyTorch
    - xgboost         # XGBoost device=cuda
```

## Legacy reproduction

- `backend: sklearn` (padrão)
- RF e MLP permanecem em CPU
- Apenas XGBoost usa `device: cuda` quando configurado
- Classificação paralela: `parallel: true`, `max_workers: 3`

## Resume de classificação

```bash
pns-research classification \
  --config configs/experiments/legacy_reproduction.yaml \
  --from-run 20260831T222903Z_legacy_reproduction_0284aca
```

Requer `tables/clustered_categorical.parquet` no run fonte (gerado pelo runner ou via `scripts/backfill_clustered_categorical.py`).

## Testes

Testes marcados `@pytest.mark.gpu` são ignorados sem cuML instalado.
