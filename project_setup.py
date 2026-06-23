"""
Módulo: setup_project.py
Descrição: Automatiza a criação da estrutura de diretórios para o projeto de TCC.
Autor: Eduardo Kanadani Bergamo
"""

from pathlib import Path

def inicializar_estrutura():
    """
    Cria os diretórios necessários para o armazenamento de dados, 
    códigos e análises experimentais.
    """
    # Define a raiz do projeto como o diretório onde este script está localizado
    raiz = Path(__file__).parent.resolve()

    # Lista de diretórios conforme a arquitetura de processamento definida
    diretorios = [
        raiz / "data" / "raw_data",          # Dados imutáveis (.mat)
        raiz / "data" / "processed_data",    # Datasets consolidados (.parquet)
        raiz / "notebooks",                  # Pesquisa exploratória (EDA)
        raiz / "results",    # Garantir que a pasta de resultados exista
        raiz / "src"                         # Scripts de processamento e modelagem
    ]

    print("Criando diretórios...")

    for caminho in diretorios:
        try:
            # parents=True cria diretórios pai se necessário; exist_ok=True evita erros se já existir
            caminho.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Diretório verificado/criado: {caminho.relative_to(raiz)}")
        except Exception as e:
            print(f"[ERRO] Falha ao criar {caminho}: {e}")

    # Criação de arquivos .gitkeep para manter a estrutura no Git sem os arquivos pesados
    for caminho in diretorios:
        gitkeep = caminho / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    print("\nAmbiente configurado com sucesso.")

if __name__ == "__main__":
    inicializar_estrutura()