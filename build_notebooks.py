"""Gera os notebooks do curso em notebooks/*.ipynb."""

from pathlib import Path

import nbformat as nbf

REPO = "naubergois/curso-yolo-colab"
OUT = Path(__file__).parent / "notebooks"

INSTALL = "!pip install -q ultralytics"

HELPERS = '''import base64
import subprocess
import urllib.request
from pathlib import Path

from IPython.display import HTML, display


def baixar(url, destino):
    if not Path(destino).exists():
        urllib.request.urlretrieve(url, destino)
    print(f"{destino}: {Path(destino).stat().st_size / 1e6:.1f} MB")
    return destino


def mostrar_video(caminho, largura=720):
    """Converte para H.264 (o navegador não toca mp4v) e embute no notebook."""
    web = str(Path(caminho).with_suffix("")) + "_web.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", caminho,
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", web],
        check=True,
    )
    dados = base64.b64encode(Path(web).read_bytes()).decode()
    display(HTML(f'<video width="{largura}" controls autoplay muted loop>'
                 f'<source src="data:video/mp4;base64,{dados}" type="video/mp4"></video>'))
    return web


def baixar_para_maquina(caminho):
    try:
        from google.colab import files
        files.download(caminho)
    except ImportError:
        print(f"Fora do Colab: o arquivo está em {Path(caminho).resolve()}")'''


def badge(nome):
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{nome}"
    return (f'<a href="{url}" target="_blank">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
            'alt="Abrir no Colab"/></a>')


def notebook(nome, celulas):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    out = []
    for tipo, texto in celulas:
        texto = texto.strip("\n")
        texto = texto.replace("{BADGE}", badge(nome))
        out.append(nbf.v4.new_markdown_cell(texto) if tipo == "md" else nbf.v4.new_code_cell(texto))
    nb.cells = out
    OUT.mkdir(exist_ok=True)
    nbf.write(nb, OUT / nome)
    print("ok", nome)


# ---------------------------------------------------------------- 01
notebook("01-deteccao-em-imagem.ipynb", [
    ("md", """
{BADGE}

# 1. Detecção de objetos em uma imagem

Uma rede de detecção olha a foto inteira de uma vez e devolve caixas: onde está cada objeto, que classe ele é e quanta certeza ela tem. É daí que vem o nome YOLO, *You Only Look Once*.

O modelo usado aqui é o `yolov8n`, a versão *nano*. Tem uns 6 MB, roda bem até sem GPU e já sabe reconhecer as 80 classes do dataset COCO: pessoa, carro, ônibus, cachorro, celular e por aí vai.

> No Colab: **Ambiente de execução → Executar tudo**. GPU é opcional nesta aula.
"""),
    ("code", INSTALL),
    ("code", HELPERS),
    ("md", """
## Carregar o modelo e a imagem

Na primeira chamada, `YOLO("yolov8n.pt")` baixa os pesos já treinados. A imagem é a foto de um ônibus numa rua, que a própria Ultralytics usa como teste.
"""),
    ("code", '''from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")
imagem = baixar("https://ultralytics.com/images/bus.jpg", "bus.jpg")'''),
    ("md", """
## Rodar a detecção

`model(imagem)` devolve uma lista com um resultado por imagem. Cada resultado traz as caixas (`boxes`), com coordenadas, classe e confiança. O `plot()` desenha tudo em cima da foto.
"""),
    ("code", '''resultado = model(imagem, verbose=False)[0]

plt.figure(figsize=(8, 10))
plt.imshow(resultado.plot()[:, :, ::-1])  # BGR -> RGB
plt.axis("off")
plt.show()'''),
    ("md", """
## Ler as caixas como dados

Desenhar é bom para olhar. Para usar num sistema, você quer os números: classe, confiança e posição de cada caixa.
"""),
    ("code", '''for caixa in resultado.boxes:
    classe = model.names[int(caixa.cls)]
    conf = float(caixa.conf)
    x1, y1, x2, y2 = caixa.xyxy[0].tolist()
    print(f"{classe:10s} {conf:5.0%}  ({x1:4.0f},{y1:4.0f}) -> ({x2:4.0f},{y2:4.0f})")

assert len(resultado.boxes) > 0, "nenhum objeto detectado"'''),
    ("md", """
## Ajustar a confiança

Por padrão o modelo descarta o que tem menos de 25% de confiança. Subir o `conf` limpa os falsos positivos, mas também perde objetos difíceis. Compare:
"""),
    ("code", '''for limite in (0.25, 0.5, 0.8):
    n = len(model(imagem, conf=limite, verbose=False)[0].boxes)
    print(f"conf >= {limite:.2f}: {n} objetos")'''),
    ("md", """
## O que fica

Uma linha de código já dá caixas, classes e confianças. O que o modelo **não** faz sozinho é lembrar do objeto de um quadro para o outro, e isso é assunto do rastreamento, na aula 3.

Para experimentar com uma foto sua, troque a URL em `baixar(...)` ou suba o arquivo pelo painel lateral do Colab.
"""),
])

# ---------------------------------------------------------------- 02
notebook("02-deteccao-em-video.ipynb", [
    ("md", """
{BADGE}

# 2. Detecção quadro a quadro em vídeo

Vídeo é só uma sequência de fotos. Para detectar objetos nele, você lê um quadro, roda o modelo, desenha as caixas e grava o quadro num arquivo novo. Repete até o fim.

O exemplo é um clipe de cinco segundos (menos de 1 MB) em que a câmera passeia por uma sala com cadeiras, plantas, mesa e livros. Muita classe ao mesmo tempo, e roda rápido até em CPU.

> GPU deixa bem mais rápido: **Ambiente de execução → Alterar o tipo de ambiente → T4 GPU**.
"""),
    ("code", INSTALL),
    ("code", HELPERS),
    ("code", '''from collections import Counter

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video = baixar(
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/home-objects-compressed.mp4",
    "sala.mp4",
)'''),
    ("md", """
## Ler, detectar, gravar

O OpenCV lê o vídeo com `VideoCapture` e escreve com `VideoWriter`. A resolução e o FPS do arquivo de saída precisam bater com os do original, senão o vídeo sai corrompido.
"""),
    ("code", '''MAX_QUADROS = 150

cap = cv2.VideoCapture(video)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"{largura}x{altura} a {fps} FPS")

out = cv2.VideoWriter("sala_detectada.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))
contagem = Counter()
quadros = 0

while quadros < MAX_QUADROS:
    ok, frame = cap.read()
    if not ok:
        break
    r = model(frame, verbose=False)[0]
    contagem.update(model.names[int(c)] for c in r.boxes.cls)
    out.write(r.plot())
    quadros += 1

cap.release()
out.release()
print(f"{quadros} quadros processados")
assert quadros > 0'''),
    ("md", """
## Assistir o resultado
"""),
    ("code", 'web = mostrar_video("sala_detectada.mp4")'),
    ("md", """
## Quantas detecções de cada classe?

Repare que o número é de **detecções**, não de objetos. A mesma cadeira aparece em dezenas de quadros e é contada em cada um. Separar "a mesma cadeira" de "outra cadeira" é exatamente o que o rastreamento resolve na próxima aula.
"""),
    ("code", '''for classe, n in contagem.most_common():
    print(f"{classe:12s} {n:5d} detecções  (~{n / quadros:.1f} por quadro)")'''),
    ("md", """
## Baixar o vídeo

No Colab, a célula abaixo baixa o vídeo anotado para o seu computador.
"""),
    ("code", 'baixar_para_maquina(web)'),
    ("md", """
## O que fica

Detecção em vídeo é um laço: ler, inferir, desenhar, gravar. O gargalo é a inferência, por isso GPU e modelos menores fazem tanta diferença. Para filtrar só uma classe, passe `classes=[0]` (pessoa) na chamada do modelo.
"""),
])

# ---------------------------------------------------------------- 03
notebook("03-rastreamento-e-contagem.ipynb", [
    ("md", """
{BADGE}

# 3. Rastreamento e contagem de pessoas

Detectar diz *tem uma pessoa aqui*. Rastrear diz *é a mesma pessoa do quadro anterior*. O rastreador (ByteTrack, que já vem no Ultralytics) liga as caixas de um quadro às do próximo e dá a cada pessoa um **ID** que se mantém enquanto ela estiver em cena.

Com IDs estáveis dá para contar gente de verdade: quantas pessoas diferentes passaram e quantas cruzaram uma linha. O vídeo é o clássico `vtest.avi` do OpenCV, um pátio de universidade filmado do alto.
"""),
    ("code", INSTALL + " lap"),
    ("code", HELPERS),
    ("code", '''import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video = baixar("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi", "calcada.avi")'''),
    ("md", """
## Rastrear e contar quem cruza a linha

O segredo está em `model.track(..., persist=True)`: o `persist` faz o rastreador lembrar dos IDs entre uma chamada e outra. `classes=[0]` limita a pessoas.

Para contar cruzamentos, guardamos a posição horizontal do centro de cada ID. Quando ela troca de lado da linha vertical, contamos uma vez aquele ID.
"""),
    ("code", '''MAX_QUADROS = 200

cap = cv2.VideoCapture(video)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
LINHA_X = largura // 2

out = cv2.VideoWriter("calcada_rastreada.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))

ids_vistos = set()
ultimo_x = {}
cruzaram = set()
por_quadro = []
exemplos = []

for i in range(MAX_QUADROS):
    ok, frame = cap.read()
    if not ok:
        break
    r = model.track(frame, persist=True, classes=[0], verbose=False)[0]
    anotado = r.plot()

    if r.boxes.id is not None:
        ids = r.boxes.id.int().tolist()
        centros_x = r.boxes.xywh[:, 0].tolist()
        for pid, cx in zip(ids, centros_x):
            antes = ultimo_x.get(pid)
            if antes is not None and (antes - LINHA_X) * (cx - LINHA_X) < 0:
                cruzaram.add(pid)
            ultimo_x[pid] = cx
        ids_vistos.update(ids)
        por_quadro.append(len(ids))
    else:
        por_quadro.append(0)

    cv2.line(anotado, (LINHA_X, 0), (LINHA_X, altura), (0, 255, 255), 2)
    cv2.putText(anotado, f"cruzaram: {len(cruzaram)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    out.write(anotado)

    if i in (20, 80, 140, 190):
        exemplos.append(anotado[:, :, ::-1])

cap.release()
out.release()

print(f"Quadros: {len(por_quadro)}")
print(f"Pessoas diferentes (IDs): {len(ids_vistos)}")
print(f"Cruzaram a linha: {len(cruzaram)}")
assert len(ids_vistos) > 0'''),
    ("md", """
## Ver quadros de exemplo
"""),
    ("code", '''plt.figure(figsize=(14, 10))
for k, img in enumerate(exemplos):
    plt.subplot(2, 2, k + 1)
    plt.imshow(img)
    plt.axis("off")
plt.suptitle(f"{len(ids_vistos)} IDs, {len(cruzaram)} cruzamentos")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(por_quadro)
plt.title("Pessoas rastreadas por quadro")
plt.xlabel("quadro")
plt.grid(True)
plt.show()'''),
    ("md", """
## Assistir e baixar
"""),
    ("code", '''web = mostrar_video("calcada_rastreada.mp4")
baixar_para_maquina(web)'''),
    ("md", """
## O que fica

O ID é o que transforma detecção em contagem. Mas ele não é perfeito: quando uma pessoa some atrás de outra e volta, o rastreador às vezes dá um ID novo. Por isso o número de IDs costuma passar um pouco do número real de pessoas. A contagem por linha sofre menos com isso, já que só conta quem atravessa.
"""),
])

# ---------------------------------------------------------------- 04
notebook("04-treino-com-dataset-proprio.ipynb", [
    ("md", """
{BADGE}

# 4. Treinar o YOLO com um dataset

O modelo pronto conhece 80 classes. Para reconhecer algo novo (uma peça da sua fábrica, um tipo de placa, uma praga na folha), você treina com fotos rotuladas. O Ultralytics faz isso com uma chamada só.

Para caber numa aula, o treino usa o `coco8`: oito fotos (quatro de treino, quatro de validação), já no formato certo, que baixam sozinhas. Serve para entender o processo, não para ter um modelo bom. Isso pede centenas de imagens por classe.

> Aqui a GPU faz diferença: **Ambiente de execução → Alterar o tipo de ambiente → T4 GPU**.
"""),
    ("code", INSTALL),
    ("md", """
## Como é um dataset YOLO

Um arquivo `.yaml` aponta as pastas de treino e validação e lista as classes. Para cada imagem existe um `.txt` com uma linha por objeto: `classe x_centro y_centro largura altura`, tudo de 0 a 1.
"""),
    ("code", '''from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils import DATASETS_DIR

model = YOLO("yolov8n.pt")'''),
    ("md", """
## Treinar

Começamos dos pesos já treinados (`yolov8n.pt`). Isso se chama *transfer learning*: o modelo já sabe ver bordas e formas, só ajusta o final. Cinco épocas com imagens de 320 px levam poucos minutos.
"""),
    ("code", '''treino = model.train(
    data="coco8.yaml",
    epochs=5,
    imgsz=320,
    batch=8,
    project="runs",
    name="coco8",
    exist_ok=True,
    plots=False,
    verbose=False,
)
pasta = Path(treino.save_dir)
print("Resultados em", pasta)'''),
    ("md", """
## O que o arquivo de rótulos tem
"""),
    ("code", '''rotulo = sorted((DATASETS_DIR / "coco8" / "labels" / "train").glob("*.txt"))[0]
print(rotulo.name)
print(rotulo.read_text())'''),
    ("md", """
## Acompanhar as perdas

O `results.csv` guarda as métricas de cada época. Num dataset de verdade a perda (`loss`) cai época após época e o `mAP50`, que mede acerto na validação, sobe. Aqui são quatro fotos de treino e cinco épocas, então as curvas oscilam. O que importa é ver onde cada número mora.
"""),
    ("code", '''df = pd.read_csv(pasta / "results.csv")
df.columns = df.columns.str.strip()
print(df[["epoch", "train/box_loss", "train/cls_loss", "metrics/mAP50(B)"]].to_string(index=False))

plt.figure(figsize=(8, 3))
plt.plot(df["epoch"], df["train/box_loss"], label="box_loss")
plt.plot(df["epoch"], df["train/cls_loss"], label="cls_loss")
plt.xlabel("época")
plt.legend()
plt.grid(True)
plt.show()'''),
    ("md", """
## Usar o modelo treinado

O melhor peso fica em `weights/best.pt`. Ele carrega igual ao modelo original.
"""),
    ("code", '''melhor = YOLO(pasta / "weights" / "best.pt")
r = melhor("https://ultralytics.com/images/bus.jpg", verbose=False)[0]
plt.figure(figsize=(6, 8))
plt.imshow(r.plot()[:, :, ::-1])
plt.axis("off")
plt.show()
print(len(r.boxes), "objetos detectados")'''),
    ("md", """
## Para treinar com as suas fotos

1. Rotule as imagens no [Roboflow](https://roboflow.com) ou no [CVAT](https://www.cvat.ai) e exporte em **YOLOv8**.
2. Suba o `.zip` no Colab e descompacte.
3. Troque `data="coco8.yaml"` pelo `data.yaml` do seu dataset e aumente `epochs` para 50 ou mais.

O processo é o mesmo desta aula. Muda o tamanho do dataset e o tempo de treino.
"""),
])
