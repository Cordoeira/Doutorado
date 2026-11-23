"""A0: benchmark de inferencia bayesiana em EDOs e EDPs.

    python -m a0bench --help            linha de comando
    from a0bench import carregar        leitura dos resultados gravados
"""
__version__ = "3.2"


def carregar(saida, destino=None):
    """Abre uma pasta de resultados (ou um .zip exportado) para reuso. Veja resultados.Arquivo."""
    from .resultados import carregar as _carregar

    return _carregar(saida, destino)
