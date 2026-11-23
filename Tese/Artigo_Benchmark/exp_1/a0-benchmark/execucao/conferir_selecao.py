"""Confere a lista de modelos do lancador: todos existem e todos tem dados reais (A0_PEQ e A0_GRANDES)."""
import os
import sys

from a0bench import config, modelos

pedidos = [m for m in (os.environ.get("A0_PEQ", "") + "," + os.environ.get("A0_GRANDES", "")).split(",") if m]
conhecidos = set(config.PETAB) | set(modelos.POR_NOME)
faltam = [m for m in pedidos if m not in conhecidos]
if faltam:
    sys.exit(f"   modelos desconhecidos: {', '.join(faltam)}")
sinteticos = [m for m in pedidos if not config.dados_reais(m)]
if sinteticos:
    sys.exit(f"   modelos sem dados reais na lista: {', '.join(sinteticos)}")
print(f"   modelos      {len(pedidos)} a rodar, todos com dados reais")
