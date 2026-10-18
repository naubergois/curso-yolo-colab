"""Gera um notebook por exercício em exercicios/*.ipynb.

Cada notebook roda sozinho: instala, refaz o trecho da aula de que o exercício
precisa, mostra o enunciado, deixa uma célula para tentar e executa a solução
com a correção automática.
"""

import os
import re
import unicodedata
from pathlib import Path

import nbformat as nbf

from comum import HELPERS, INSTALL, badge, colab_url, novo_notebook
from exercicios import CONFERIR, EXERCICIOS

OUT = Path(os.environ.get("OUT", Path(__file__).parent / "exercicios"))

AULAS = {
    "01-deteccao-em-imagem.ipynb": {
        "nome": "Detecção em imagem",
        "install": INSTALL,
        "contexto": "A preparação carrega o YOLOv8 e roda a detecção na foto do ônibus, como na aula 1. É dela que saem as caixas usadas no exercício.",
        "setup": '''from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")
imagem = baixar("https://ultralytics.com/images/bus.jpg", "bus.jpg")
resultado = model(imagem, verbose=False)[0]

plt.figure(figsize=(6, 8))
plt.imshow(resultado.plot()[:, :, ::-1])
plt.axis("off")
plt.show()
print(len(resultado.boxes), "objetos detectados")''',
    },
    "02-deteccao-em-video.ipynb": {
        "nome": "Detecção em vídeo",
        "install": INSTALL,
        "contexto": "A preparação baixa o vídeo da sala e roda a detecção nos primeiros 150 quadros, como na aula 2. No fim ficam prontos o `contagem`, com as detecções por classe, e o `quadros`, com quantos foram processados.",
        "setup": '''from collections import Counter

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video = baixar(
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/home-objects-compressed.mp4",
    "sala.mp4",
)

MAX_QUADROS = 150
cap = cv2.VideoCapture(video)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
contagem = Counter()
quadros = 0
while quadros < MAX_QUADROS:
    ok, frame = cap.read()
    if not ok:
        break
    r = model(frame, verbose=False)[0]
    contagem.update(model.names[int(c)] for c in r.boxes.cls)
    quadros += 1
cap.release()

print(f"{largura}x{altura} a {fps} FPS, {quadros} quadros processados")
print(dict(contagem.most_common(5)))''',
    },
    "03-rastreamento-e-contagem.ipynb": {
        "nome": "Rastreamento e contagem",
        "install": INSTALL + " lap",
        "contexto": "A preparação rastreia as pessoas nos primeiros 200 quadros do vídeo do pátio e conta quem cruza a linha vertical, como na aula 3. Ficam prontos `por_quadro`, `ids_vistos` e `cruzaram`.",
        "setup": '''import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video = baixar("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi", "calcada.avi")

MAX_QUADROS = 200
cap = cv2.VideoCapture(video)
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
LINHA_X = largura // 2

ids_vistos, ultimo_x, cruzaram, por_quadro = set(), {}, set(), []
for _ in range(MAX_QUADROS):
    ok, frame = cap.read()
    if not ok:
        break
    r = model.track(frame, persist=True, classes=[0], verbose=False)[0]
    if r.boxes.id is None:
        por_quadro.append(0)
        continue
    ids = r.boxes.id.int().tolist()
    for pid, cx in zip(ids, r.boxes.xywh[:, 0].tolist()):
        antes = ultimo_x.get(pid)
        if antes is not None and (antes - LINHA_X) * (cx - LINHA_X) < 0:
            cruzaram.add(pid)
        ultimo_x[pid] = cx
    ids_vistos.update(ids)
    por_quadro.append(len(ids))
cap.release()

print(f"Quadros: {len(por_quadro)} | IDs: {len(ids_vistos)} | cruzaram a linha: {len(cruzaram)}")''',
    },
    "04-treino-com-dataset-proprio.ipynb": {
        "nome": "Treino com dataset",
        "install": INSTALL,
        "contexto": "A preparação treina o YOLOv8 por cinco épocas no `coco8`, como na aula 4. Ficam prontos o `df`, com as métricas de cada época, e o `melhor`, que é o modelo treinado.",
        "setup": '''from pathlib import Path

import pandas as pd
from ultralytics import YOLO
from ultralytics.utils import DATASETS_DIR

model = YOLO("yolov8n.pt")
treino = model.train(
    data="coco8.yaml", epochs=5, imgsz=320, batch=8,
    project="runs", name="coco8", exist_ok=True, plots=False, verbose=False,
)
pasta = Path(treino.save_dir)
df = pd.read_csv(pasta / "results.csv")
df.columns = df.columns.str.strip()
melhor = YOLO(pasta / "weights" / "best.pt")

print(df[["epoch", "train/box_loss", "train/cls_loss", "metrics/mAP50(B)"]].to_string(index=False))''',
    },
}


def slug(texto):
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")


def lista_exercicios():
    """[(aula_nb, numero_aula, n, exercicio, nome_arquivo)] na ordem do curso."""
    saida = []
    for aula_nb, aula in EXERCICIOS.items():
        k = int(aula_nb[:2])
        for n, ex in enumerate(aula["lista"], start=1):
            saida.append((aula_nb, k, n, ex, f"aula{k}-exercicio{n}-{slug(ex['titulo'])}.ipynb"))
    return saida


def gerar():
    todos = lista_exercicios()
    OUT.mkdir(exist_ok=True)
    for i, (aula_nb, k, n, ex, arquivo) in enumerate(todos):
        info = AULAS[aula_nb]
        proximo = todos[i + 1] if i + 1 < len(todos) else None
        aula_url = colab_url(f"notebooks/{aula_nb}")

        if proximo:
            fecho = (f"Próximo: [Aula {proximo[1]}, exercício {proximo[2]}: {proximo[3]['titulo']}]"
                     f"({colab_url('exercicios/' + proximo[4])}).")
        else:
            fecho = "Esse foi o último dos 24. Agora é trocar os vídeos e as fotos pelos seus."

        celulas = [
            ("md", f"""{badge('exercicios/' + arquivo)}

# Aula {k}, exercício {n}: {ex['titulo']}

{ex['enunciado'].strip()}

O exercício é da aula [{k}. {info['nome']}]({aula_url}), e este notebook roda sozinho. Use **Ambiente de execução → Executar tudo**: ele prepara os dados, deixa uma célula para você tentar e, no fim, mostra uma solução já conferida."""),
            ("code", info["install"]),
            ("code", HELPERS),
            ("md", f"## Preparação\n\n{info['contexto']}"),
            ("code", info["setup"]),
            ("form", "# @title Correção automática (rode, não precisa ler)\n"
                     + EXERCICIOS[aula_nb]["preparo"].strip("\n") + "\n" + CONFERIR),
            ("md", f"## Sua vez\n\n{ex['enunciado'].strip()}\n\n*Dica:* {ex['dica']}\n\n"
                   "Troque o `None` pela sua resposta e rode a célula."),
            ("code", f"{ex.get('inicio', '')}resposta = None\n\nconferir({n}, resposta)"),
            ("md", "## Uma solução\n\nCompare com o que você fez. Existem outros caminhos certos: o que vale é a `conferir` aprovar."),
            ("code", f"{ex['solucao'].strip()}\n\nconferir({n}, resposta)"),
            ("md", f"## O que fica\n\n{fecho}"),
        ]

        nb = novo_notebook()
        for tipo, texto in celulas:
            if tipo == "md":
                nb.cells.append(nbf.v4.new_markdown_cell(texto.strip("\n")))
            else:
                cel = nbf.v4.new_code_cell(texto.strip("\n"))
                if tipo == "form":
                    cel.metadata["cellView"] = "form"
                nb.cells.append(cel)
        nbf.write(nb, OUT / arquivo)
    print(f"{len(todos)} notebooks em {OUT}")


if __name__ == "__main__":
    gerar()
