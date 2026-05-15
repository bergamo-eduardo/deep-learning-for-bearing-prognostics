# Estimativa da Vida Útil Remanescente de Rolamentos via Deep Learning

Este repositório consolida o desenvolvimento técnico do Trabalho de Conclusão de Curso em Engenharia Mecânica, focado na aplicação de Redes Neurais Convolucionais (CNN) para o prognóstico de falhas em elementos de rolamento. A pesquisa aborda a complexidade de prever a *Remaining Useful Life* (RUL) sob regimes de operação dinâmicos, integrando processamento de sinais de alta frequência e variáveis de processo industrial.

---

## 1. Dependências Técnicas

As bibliotecas abaixo são fundamentais para a execução do módulo de pré-processamento e análise de dados. As versões especificadas garantem a compatibilidade com as funcionalidades de *streaming* e processamento digital de sinais utilizadas.

### 1.1 Manipulação de Dados e Performance
*   **Polars (>=0.20.0)**: Biblioteca principal para manipulação de dados *Out-of-Core*. Utilizada para estruturar os dados em arquitetura *Lazy API*, permitindo o processamento de grandes volumes através de *streaming* para disco via `sink_parquet`.
*   **Pandas (>=2.0.0)**: Empregada especificamente no processamento relacional de metadados. A função `merge_asof` é utilizada para a sincronização temporal entre condições de operação e medições de temperatura.
*   **NumPy (>=1.24.0)**: Utilizada para operações vetoriais de baixo nível e criação de matrizes para os parâmetros de setpoint e cálculos de RUL.

### 1.2 Processamento Digital de Sinais (DSP)
*   **SciPy (>=1.10.0)**: Responsável pelo tratamento técnico dos sinais brutos:
    *   `scipy.io`: Carregamento determinístico de arquivos binários do MATLAB (`.mat`).
    *   `scipy.signal`: Aplicação de filtros FIR anti-aliasing e decimação de sinal para normalização da taxa de amostragem em 64 kHz.

### 1.3 Armazenamento e Integração
*   **PyArrow (>=14.0.0)**: Motor de persistência obrigatório para a manipulação de arquivos no formato Apache Parquet, garantindo alta taxa de compressão e velocidade de leitura para o treinamento da rede neural.

### 1.4 Visualização e Diagnóstico
*   **Matplotlib (>=3.7.0)** e **Seaborn (>=0.12.0)**: Ferramentas utilizadas nos módulos de Análise Exploratória de Dados (EDA) para validação visual dos sinais de vibração e tendências de degradação térmica.

---

## 2. Contexto Experimental e Dados

A fundamentação experimental baseia-se no conjunto de dados de ensaios de vida (*run-to-failure*) da **Paderborn University**. O dataset compreende medições de vibração síncrona capturadas a 64 kHz sob quatro condições operacionais distintas, onde foram variados os níveis de torque e a velocidade de rotação do eixo.

Diferente de bases de dados com falhas artificiais, este projeto utiliza sinais provenientes de desgaste natural, o que exige um tratamento rigoroso para a identificação de componentes espectrais de baixa energia que precedem a falha funcional.

---

## 3. Estratégia de Processamento de Sinais

O pipeline de dados foi projetado para transformar sinais brutos de aceleração (medidos em Volts e convertidos para unidades de gravidade, *g*) em tensores de entrada para a rede neural.

### Fluxo de Processamento:
1.  **Segmentação:** Divisão dos sinais em janelas temporais fixas.
2.  **Otimização:** Aplicação de *downsampling* calculado para reduzir o custo computacional sem violar o Teorema de Nyquist.
3.  **Categorização de Variáveis:**
    *   **Controle:** Setpoints de carga e velocidade.
    *   **Processo:** Aceleração real medida pelos sensores (horizontal frontal e traseiro).

Isso garante que o modelo aprenda a relação causal entre o regime de trabalho e a taxa de degradação.

---

## 4. Arquitetura e Implementação

A arquitetura implementada utiliza camadas de convolução para a extração automática de atributos no domínio do tempo e da frequência.

### Organização do Projeto:
```text
.
├── data/
│   ├── raw_data/                # Dados imutáveis de ensaios (University of Paderborn)
│   │   └── B0XX/                # Pastas por experimento (B01 a B17)
│   │       ├── vibrationData/   # Sinais de aceleração em formato .mat
│   │       ├── B0XX_log.pdf     # Documentação do ensaio específico
│   │       ├── *_meanTemperatures.csv
│   │       └── *_operatingConditions.csv
│   ├── processed_data/          # Datasets finais consolidados em formato .parquet
│   └── tmp_parquet_chunks/      # Diretório transiente para processamento Out-of-Core
├── notebooks/                   # Prototipagem e Análise Exploratória de Dados (EDA)
├── references/                  # Literatura de apoio e artigos base (Aimiyenkagbon 2024)
├── results/                     # Saídas do modelo: pesos (.h5), métricas e gráficos
├── src/                         # Núcleo de desenvolvimento
│   ├── model.py                 # Definição da arquitetura da Rede Neural (CNN)
│   └── preprocessing.py         # Pipeline de DSP e estruturação experimental
├── .gitignore                   # Filtros para controle de versão (ignora dados pesados)
├── LICENSE                      # Licença de uso do software (MIT)
├── project_setup.py             # Script para inicialização automática de diretórios
└── README.md                    # Documentação principal do projeto