# Curso de YOLO no Colab

Quatro notebooks curtos para aprender detecção de objetos com YOLOv8, do primeiro `model(imagem)` até treinar com um dataset. Cada um abre direto no Google Colab, instala o que precisa e roda com vídeos pequenos (menos de 10 MB).

| Aula | Abrir |
|---|---|
| 1. Detecção em imagem | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naubergois/curso-yolo-colab/blob/main/notebooks/01-deteccao-em-imagem.ipynb) |
| 2. Detecção em vídeo | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naubergois/curso-yolo-colab/blob/main/notebooks/02-deteccao-em-video.ipynb) |
| 3. Rastreamento e contagem de pessoas | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naubergois/curso-yolo-colab/blob/main/notebooks/03-rastreamento-e-contagem.ipynb) |
| 4. Treino com dataset próprio | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naubergois/curso-yolo-colab/blob/main/notebooks/04-treino-com-dataset-proprio.ipynb) |

No Colab, use **Ambiente de execução → Executar tudo**. As aulas 1 a 3 rodam em CPU. Na aula 4, uma GPU T4 ajuda.

## Vídeos usados

- `home-objects-compressed.mp4`, dos [assets da Ultralytics](https://github.com/ultralytics/assets/releases/tag/v0.0.0)
- `vtest.avi`, dos [exemplos do OpenCV](https://github.com/opencv/opencv/tree/master/samples/data)

Os notebooks já vêm executados: dá para ver imagens, gráficos e vídeos de resultado antes de rodar qualquer coisa.

## Editar

Os notebooks são gerados por `build_notebooks.py`, que cria versões sem saída. Para atualizar, gere, execute e limpe:

```bash
pip install nbformat nbconvert ipykernel ultralytics lap pandas
python build_notebooks.py
mkdir -p /tmp/exec && cp notebooks/*.ipynb /tmp/exec/
(cd /tmp/exec && jupyter nbconvert --to notebook --execute --inplace *.ipynb)
cp /tmp/exec/*.ipynb notebooks/
python clean_outputs.py /tmp/exec notebooks/*.ipynb
```

Mais em [naubergois.com/curso-yolo](https://naubergois.com/curso-yolo/).
