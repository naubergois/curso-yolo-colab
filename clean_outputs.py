"""Limpa as saídas de notebooks executados localmente antes do commit.

Uso: python clean_outputs.py <pasta_de_execucao> notebooks/*.ipynb
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
RUIDO = ("New https://pypi.org/project/ultralytics", "engine/trainer: ", " 0% ─")


def limpar_texto(texto, prefixos):
    linhas = []
    for linha in ANSI.sub("", texto).split("\n"):
        linha = linha.split("\r")[-1]
        if any(r in linha for r in RUIDO):
            continue
        linhas.append(linha)
    texto = "\n".join(linhas)
    for p in prefixos:
        texto = texto.replace(p, "")
    return texto


def main():
    pasta = Path(sys.argv[1]).resolve()
    prefixos = [f"{pasta}/", f"{pasta.parent}/", str(pasta), str(pasta.parent)]
    for caminho in sys.argv[2:]:
        nb = nbf.read(caminho, as_version=4)
        for cel in nb.cells:
            if cel.cell_type != "code":
                continue
            cel.metadata.pop("execution", None)
            for out in cel.get("outputs", []):
                if out.output_type == "stream":
                    out.text = limpar_texto(out.text, prefixos)
                elif "text/plain" in out.get("data", {}):
                    out.data["text/plain"] = limpar_texto(out.data["text/plain"], prefixos)
        nbf.write(nb, caminho)
        print("limpo", caminho)


if __name__ == "__main__":
    main()
