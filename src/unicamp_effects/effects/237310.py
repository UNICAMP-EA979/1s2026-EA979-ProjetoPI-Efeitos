import numpy as np

from unicamp_effects.registry import register


def amostrar(img: np.ndarray, block_size: int) -> np.ndarray:

    # percorre a imagem em blocos de block_size x block_size e amostra o pixel
    # do canto superior esquerdo de cada bloco
    amostragem = img[::block_size, ::block_size]

    # reconstrói a imagem a partir da amostragem, repetindo cada pixel
    # block_size x block_size vezes
    expansao = np.repeat(
        np.repeat(amostragem, block_size, axis=0), block_size, axis=1)

    return expansao


@register(prefix="237310")
def aberracao_cromatica(img: np.ndarray) -> np.ndarray:
    shift = 20
    out = np.zeros_like(img)

    # realiza o deslocamento dos canais de cor via slicing
    # o canal vermelho é deslocado para a esquerda;
    # o canal verde permanece inalterado;
    # o canal azul é deslocado para a direita.
    out[:, :-shift, 0] = img[:, shift:, 0]
    out[:, :, 1] = img[:, :, 1]
    out[:, shift:, 2] = img[:, :-shift, 2]

    return out


@register(prefix="237310")
def pixelizacao(img: np.ndarray) -> np.ndarray:
    block_size = 25
    h, w, _ = img.shape

    # para a pixelização, a imagem é essencialmente subamostrada
    amostragem = amostrar(img, block_size)

    out = amostragem[:h, :w]

    return out


@register(prefix="237310")
def quantizacao(img: np.ndarray) -> np.ndarray:
    block_size = 4

    # para a quantização, a imagem é subamostrada,
    # mantendo a resolução próxima da original
    amostragem = amostrar(img, block_size)

    level = 64
    # em seguida, os valores são divididos e multiplicados pelo nível de quantização
    # para reduzir a quantidade de cores
    out = (amostragem // level) * level

    return out
