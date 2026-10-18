"""Exercícios com correção automática, seis por aula.

Cada exercício tem enunciado, dica, solução e uma verificação que calcula a
resposta esperada a partir dos dados do próprio notebook.
"""

CONFERIR = '''
_acertos = {}


def conferir(n, resposta):
    if resposta is None:
        print(f"Exercício {n}: ainda sem resposta.")
        return
    try:
        certo = bool(VERIFICAR[n](resposta))
    except Exception as erro:
        certo = False
        print(f"  (a verificação deu erro: {erro})")
    _acertos[n] = certo
    print(f"Exercício {n}: " + ("✅ certo!" if certo else "❌ ainda não. Releia o enunciado e tente de novo."))


def placar():
    print(f"{sum(_acertos.values())} de {len(VERIFICAR)} exercícios certos.")
'''


def celulas_exercicios(aula, solucoes=False):
    celulas = [
        ("md", """
## Exercícios

Seis exercícios para fixar a aula. Escreva a resposta na variável `resposta` e rode a célula: a função `conferir` diz na hora se está certo. Se travar, abra **Ver solução**.

Rode antes a célula de preparação abaixo. Ela só carrega as verificações.
"""),
        ("form", "# @title Preparação dos exercícios (rode, não precisa ler)\n" + aula["preparo"].strip("\n") + "\n" + CONFERIR),
    ]
    for n, ex in enumerate(aula["lista"], start=1):
        solucao = ex["solucao"].strip("\n")
        celulas.append(("md", f"""
### Exercício {n}. {ex["titulo"]}

{ex["enunciado"].strip()}

<details>
<summary>Ver solução</summary>

```python
{solucao}
```

</details>
"""))
        corpo = solucao if solucoes else f"# Dica: {ex['dica']}\n{ex.get('inicio', '')}resposta = None"
        celulas.append(("code", f"{corpo}\n\nconferir({n}, resposta)"))
    celulas.append(("code", "placar()"))
    return celulas


AULA1 = {
    "preparo": '''
import cv2

_cls = resultado.boxes.cls.int().tolist()
_conf = resultado.boxes.conf.tolist()
_xyxy = resultado.boxes.xyxy.tolist()


def _tamanho_onibus():
    i = [k for k, c in enumerate(_cls) if model.names[c] == "bus"][0]
    x1, y1, x2, y2 = map(int, _xyxy[i])
    return y2 - y1, x2 - x1


VERIFICAR = {
    1: lambda r: int(r) == _cls.count(0),
    2: lambda r: set(r) == {model.names[c] for c in _cls},
    3: lambda r: abs(float(r) - max(_conf)) < 1e-3,
    4: lambda r: abs(float(r) - max((c[2] - c[0]) * (c[3] - c[1]) for c in _xyxy)) < 1,
    5: lambda r: int(r) == len(model(imagem, classes=[0], verbose=False)[0].boxes),
    6: lambda r: all(abs(a - b) <= 2 for a, b in zip(r.shape[:2], _tamanho_onibus())),
}
''',
    "lista": [
        {
            "titulo": "Contar pessoas",
            "enunciado": "Quantas pessoas o modelo encontrou em `resultado`? A classe `person` tem índice 0.",
            "dica": "resultado.boxes.cls tem a classe de cada caixa.",
            "solucao": "resposta = int((resultado.boxes.cls == 0).sum())",
        },
        {
            "titulo": "Classes encontradas",
            "enunciado": "Monte um conjunto (`set`) com os **nomes** das classes que apareceram na imagem.",
            "dica": "model.names é um dicionário índice -> nome.",
            "solucao": "resposta = {model.names[int(c)] for c in resultado.boxes.cls}",
        },
        {
            "titulo": "A detecção mais segura",
            "enunciado": "Qual é a maior confiança entre todas as caixas? Responda com um número entre 0 e 1.",
            "dica": "resultado.boxes.conf guarda a confiança de cada caixa.",
            "solucao": "resposta = float(resultado.boxes.conf.max())",
        },
        {
            "titulo": "A maior caixa",
            "enunciado": "Qual é a área, em pixels, da maior caixa? Área é largura vezes altura, e `xyxy` dá os dois cantos.",
            "dica": "x1, y1, x2, y2 = resultado.boxes.xyxy.T separa as quatro colunas.",
            "solucao": '''x1, y1, x2, y2 = resultado.boxes.xyxy.T
resposta = float(((x2 - x1) * (y2 - y1)).max())''',
        },
        {
            "titulo": "Só pessoas",
            "enunciado": "Rode o modelo de novo na mesma imagem, mas pedindo só pessoas com `classes=[0]`. Quantas caixas vieram?",
            "dica": "model(imagem, classes=[...], verbose=False)[0].boxes",
            "solucao": "resposta = len(model(imagem, classes=[0], verbose=False)[0].boxes)",
        },
        {
            "titulo": "Recortar o ônibus",
            "enunciado": "Recorte só o ônibus da imagem original. Leia a foto com `cv2.imread`, ache a caixa da classe `bus` e fatie o array. A resposta é o recorte.",
            "dica": "Uma imagem do OpenCV é um array [linhas, colunas]: img[y1:y2, x1:x2].",
            "solucao": '''img = cv2.imread(imagem)
i = [k for k, c in enumerate(resultado.boxes.cls) if model.names[int(c)] == "bus"][0]
x1, y1, x2, y2 = map(int, resultado.boxes.xyxy[i])
resposta = img[y1:y2, x1:x2]

plt.imshow(resposta[:, :, ::-1])
plt.axis("off")
plt.show()''',
        },
    ],
}

AULA2 = {
    "preparo": '''
from functools import lru_cache

_cap = cv2.VideoCapture(video)
_total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
_fps = _cap.get(cv2.CAP_PROP_FPS)
_cap.release()


@lru_cache(None)
def _classes_por_quadro(n):
    cap = cv2.VideoCapture(video)
    saida = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        saida.append(tuple(model.names[int(c)] for c in model(frame, verbose=False)[0].boxes.cls))
    cap.release()
    return saida


def _quadros_do_arquivo(caminho):
    cap = cv2.VideoCapture(str(caminho))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


VERIFICAR = {
    1: lambda r: abs(float(r) - _total / _fps) < 0.05,
    2: lambda r: r == contagem.most_common(1)[0][0],
    3: lambda r: int(r) == sum(q.count("chair") for q in _classes_por_quadro(60)[:50:5]),
    4: lambda r: int(r) == sum("chair" in q for q in _classes_por_quadro(60)),
    5: lambda r: _quadros_do_arquivo(r) == 60,
    6: lambda r: abs(float(r) - contagem["chair"] / quadros) < 1e-3,
}
''',
    "lista": [
        {
            "titulo": "Duração do vídeo",
            "enunciado": "Quantos segundos tem o vídeo **inteiro**? Use o total de quadros e o FPS do arquivo.",
            "dica": "cap.get(cv2.CAP_PROP_FRAME_COUNT) e cap.get(cv2.CAP_PROP_FPS).",
            "solucao": '''cap = cv2.VideoCapture(video)
resposta = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
cap.release()''',
        },
        {
            "titulo": "A classe campeã",
            "enunciado": "Qual classe teve mais detecções no vídeo processado? Responda com o nome.",
            "dica": "contagem é um Counter, e Counter tem most_common().",
            "solucao": "resposta = contagem.most_common(1)[0][0]",
        },
        {
            "titulo": "Pular quadros",
            "enunciado": "Processar todo quadro é caro. Nos primeiros 50 quadros, rode o modelo só em 1 a cada 5 (quadros 0, 5, 10...) e some quantas cadeiras (`chair`) foram detectadas.",
            "dica": "Leia todos os quadros com cap.read(), mas só chame o modelo quando i % 5 == 0.",
            "solucao": '''cap = cv2.VideoCapture(video)
resposta = 0
for i in range(50):
    ok, frame = cap.read()
    if not ok:
        break
    if i % 5 == 0:
        r = model(frame, verbose=False)[0]
        resposta += sum(model.names[int(c)] == "chair" for c in r.boxes.cls)
cap.release()''',
        },
        {
            "titulo": "Quadros com cadeira",
            "enunciado": "Em quantos dos primeiros 60 quadros aparece **pelo menos uma** cadeira?",
            "dica": "Para cada quadro, monte a lista de nomes e teste \"chair\" in nomes.",
            "solucao": '''cap = cv2.VideoCapture(video)
resposta = 0
for _ in range(60):
    ok, frame = cap.read()
    if not ok:
        break
    nomes = [model.names[int(c)] for c in model(frame, verbose=False)[0].boxes.cls]
    resposta += "chair" in nomes
cap.release()''',
        },
        {
            "titulo": "Vídeo só com plantas",
            "enunciado": "Grave um vídeo novo com os primeiros 60 quadros, desenhando só as detecções de `potted plant`. A resposta é o caminho do arquivo. Descubra o índice da classe em `model.names`.",
            "dica": "Passe classes=[índice] na chamada do modelo e use o mesmo VideoWriter da aula.",
            "solucao": '''planta = [k for k, v in model.names.items() if v == "potted plant"][0]
cap = cv2.VideoCapture(video)
out = cv2.VideoWriter("so_plantas.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))
for _ in range(60):
    ok, frame = cap.read()
    if not ok:
        break
    out.write(model(frame, classes=[planta], verbose=False)[0].plot())
cap.release()
out.release()
resposta = "so_plantas.mp4"
mostrar_video(resposta)''',
        },
        {
            "titulo": "Média por quadro",
            "enunciado": "Em média, quantas cadeiras o modelo detectou por quadro no vídeo processado?",
            "dica": "Você já tem contagem e quadros.",
            "solucao": 'resposta = contagem["chair"] / quadros',
        },
    ],
}

AULA3 = {
    "preparo": '''
from collections import Counter
from functools import lru_cache


@lru_cache(None)
def _trilhas(n):
    m = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video)
    pontos = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        r = m.track(frame, persist=True, classes=[0], verbose=False)[0]
        if r.boxes.id is not None:
            for pid, (cx, cy, _, _) in zip(r.boxes.id.int().tolist(), r.boxes.xywh.tolist()):
                pontos.append((pid, cx, cy))
    cap.release()
    return tuple(pontos)


def _cruzam_horizontal(n):
    linha, ultimo, cruzaram_h = altura // 2, {}, set()
    for pid, _, cy in _trilhas(n):
        antes = ultimo.get(pid)
        if antes is not None and (antes - linha) * (cy - linha) < 0:
            cruzaram_h.add(pid)
        ultimo[pid] = cy
    return len(cruzaram_h)


VERIFICAR = {
    1: lambda r: int(r) == max(por_quadro),
    2: lambda r: int(r) == por_quadro.index(max(por_quadro)),
    3: lambda r: abs(float(r) - sum(por_quadro) / len(por_quadro)) < 1e-3,
    4: lambda r: int(r) == len(ids_vistos - cruzaram),
    5: lambda r: int(r) == sum(n >= 50 for n in Counter(p for p, _, _ in _trilhas(100)).values()),
    6: lambda r: int(r) == _cruzam_horizontal(100),
}
''',
    "lista": [
        {
            "titulo": "Pico de movimento",
            "enunciado": "Qual foi o maior número de pessoas rastreadas **ao mesmo tempo** num quadro?",
            "dica": "por_quadro guarda quantas pessoas havia em cada quadro.",
            "solucao": "resposta = max(por_quadro)",
        },
        {
            "titulo": "Quando foi o pico",
            "enunciado": "Em qual quadro (índice começando em 0) esse pico aconteceu pela primeira vez?",
            "dica": "Listas têm o método index().",
            "solucao": "resposta = por_quadro.index(max(por_quadro))",
        },
        {
            "titulo": "Lotação média",
            "enunciado": "Quantas pessoas, em média, estavam em cena por quadro?",
            "dica": "Soma dividida pela quantidade de quadros.",
            "solucao": "resposta = sum(por_quadro) / len(por_quadro)",
        },
        {
            "titulo": "Quem não cruzou",
            "enunciado": "Quantos IDs apareceram no vídeo mas **não** cruzaram a linha?",
            "dica": "ids_vistos e cruzaram são conjuntos, e conjuntos aceitam subtração.",
            "solucao": "resposta = len(ids_vistos - cruzaram)",
        },
        {
            "titulo": "Quem ficou em cena",
            "enunciado": "Rastreie de novo os primeiros 100 quadros com um modelo novo (`YOLO(\"yolov8n.pt\")`, para os IDs começarem do zero). Quantas pessoas apareceram em **pelo menos 50** desses quadros?",
            "dica": "Um Counter atualizado com os IDs de cada quadro conta as aparições de cada pessoa.",
            "inicio": "novo = YOLO(\"yolov8n.pt\")\n",
            "solucao": '''novo = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video)
aparicoes = Counter()
for _ in range(100):
    ok, frame = cap.read()
    if not ok:
        break
    r = novo.track(frame, persist=True, classes=[0], verbose=False)[0]
    if r.boxes.id is not None:
        aparicoes.update(r.boxes.id.int().tolist())
cap.release()
resposta = sum(n >= 50 for n in aparicoes.values())''',
        },
        {
            "titulo": "Linha horizontal",
            "enunciado": "Troque a linha vertical por uma **horizontal** no meio da imagem (`y = altura // 2`). Rastreando os primeiros 100 quadros com um modelo novo, quantas pessoas cruzaram essa linha?",
            "dica": "É a mesma lógica da aula, trocando o centro x (coluna 0 de xywh) pelo centro y (coluna 1).",
            "inicio": "novo = YOLO(\"yolov8n.pt\")\nLINHA_Y = altura // 2\n",
            "solucao": '''novo = YOLO("yolov8n.pt")
LINHA_Y = altura // 2
cap = cv2.VideoCapture(video)
ultimo_y, cruzaram_h = {}, set()
for _ in range(100):
    ok, frame = cap.read()
    if not ok:
        break
    r = novo.track(frame, persist=True, classes=[0], verbose=False)[0]
    if r.boxes.id is None:
        continue
    for pid, cy in zip(r.boxes.id.int().tolist(), r.boxes.xywh[:, 1].tolist()):
        antes = ultimo_y.get(pid)
        if antes is not None and (antes - LINHA_Y) * (cy - LINHA_Y) < 0:
            cruzaram_h.add(pid)
        ultimo_y[pid] = cy
cap.release()
resposta = len(cruzaram_h)''',
        },
    ],
}

AULA4 = {
    "preparo": '''
from collections import Counter
from functools import lru_cache

import cv2

_raiz = DATASETS_DIR / "coco8"
_linhas = [
    l.split()
    for f in sorted((_raiz / "labels" / "train").glob("*.txt"))
    for l in f.read_text().splitlines()
    if l.strip()
]
_por_classe = Counter(int(l[0]) for l in _linhas)


def _caixa_em_pixels():
    img = cv2.imread(str(_raiz / "images" / "train" / "000000000009.jpg"))
    h, w = img.shape[:2]
    linha = (_raiz / "labels" / "train" / "000000000009.txt").read_text().splitlines()[0]
    _, x, y, bw, bh = map(float, linha.split())
    return ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)


@lru_cache(None)
def _map50():
    return float(melhor.val(data="coco8.yaml", imgsz=320, verbose=False, plots=False).box.map50)


VERIFICAR = {
    1: lambda r: int(r) == len(list((_raiz / "images" / "train").glob("*.jpg"))),
    2: lambda r: int(r) == len(_linhas),
    3: lambda r: r in {model.names[c] for c, n in _por_classe.items() if n == max(_por_classe.values())},
    4: lambda r: all(abs(a - b) <= 2 for a, b in zip(r, _caixa_em_pixels())),
    5: lambda r: int(r) == int(df.loc[df["train/box_loss"].idxmin(), "epoch"]),
    6: lambda r: abs(float(r) - _map50()) < 1e-3,
}
''',
    "lista": [
        {
            "titulo": "Tamanho do treino",
            "enunciado": "Quantas imagens há na pasta de treino do `coco8`?",
            "dica": "As imagens ficam em DATASETS_DIR / \"coco8\" / \"images\" / \"train\".",
            "solucao": 'resposta = len(list((DATASETS_DIR / "coco8" / "images" / "train").glob("*.jpg")))',
        },
        {
            "titulo": "Objetos rotulados",
            "enunciado": "Quantos objetos rotulados existem, somando todos os arquivos `.txt` do treino? Cada linha é um objeto.",
            "dica": "Some o número de linhas de cada arquivo em labels/train.",
            "solucao": '''rotulos = (DATASETS_DIR / "coco8" / "labels" / "train").glob("*.txt")
resposta = sum(len(f.read_text().strip().splitlines()) for f in rotulos)''',
        },
        {
            "titulo": "Classe mais rotulada",
            "enunciado": "Qual classe mais aparece nos rótulos de treino? Responda com o nome, não com o número.",
            "dica": "O primeiro número de cada linha é a classe; model.names traduz para nome.",
            "solucao": '''cont = Counter()
for f in (DATASETS_DIR / "coco8" / "labels" / "train").glob("*.txt"):
    cont.update(int(l.split()[0]) for l in f.read_text().strip().splitlines())
resposta = model.names[cont.most_common(1)[0][0]]''',
        },
        {
            "titulo": "De volta para pixels",
            "enunciado": "Pegue a primeira linha do rótulo `000000000009.txt` e converta para pixels no formato `(x1, y1, x2, y2)`, usando o tamanho real da imagem `000000000009.jpg`.",
            "dica": "x1 = (x_centro - largura / 2) * largura_da_imagem, e assim por diante.",
            "solucao": '''raiz = DATASETS_DIR / "coco8"
img = cv2.imread(str(raiz / "images" / "train" / "000000000009.jpg"))
h, w = img.shape[:2]
linha = (raiz / "labels" / "train" / "000000000009.txt").read_text().splitlines()[0]
_, x, y, bw, bh = map(float, linha.split())
resposta = (int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h))''',
        },
        {
            "titulo": "A melhor época",
            "enunciado": "Em qual época a `train/box_loss` foi a menor?",
            "dica": "df[\"coluna\"].idxmin() dá a linha do menor valor.",
            "solucao": 'resposta = int(df.loc[df["train/box_loss"].idxmin(), "epoch"])',
        },
        {
            "titulo": "Validar o modelo",
            "enunciado": "Rode a validação do modelo treinado com `melhor.val(data=\"coco8.yaml\", imgsz=320)` e responda com o `mAP50`.",
            "dica": "O retorno de val() tem .box.map50.",
            "solucao": '''metricas = melhor.val(data="coco8.yaml", imgsz=320, verbose=False, plots=False)
resposta = metricas.box.map50''',
        },
    ],
}

EXERCICIOS = {
    "01-deteccao-em-imagem.ipynb": AULA1,
    "02-deteccao-em-video.ipynb": AULA2,
    "03-rastreamento-e-contagem.ipynb": AULA3,
    "04-treino-com-dataset-proprio.ipynb": AULA4,
}
