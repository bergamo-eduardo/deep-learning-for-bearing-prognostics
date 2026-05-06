# Estimativa da Vida Útil Remanescente de Rolamentos via Deep Learning

Este repositório consolida o desenvolvimento técnico do Trabalho de Conclusão de Curso em Engenharia Mecânica focado na aplicação de Redes Neurais Convolucionais (CNN) para o prognóstico de falhas em elementos de rolamento. A pesquisa aborda a complexidade de prever a *Remaining Useful Life* (RUL) sob regimes de operação dinâmicos, integrando processamento de sinais de alta frequência e variáveis de processo industrial.

## Contexto Experimental e Dados

A fundamentação experimental baseia-se no conjunto de dados de ensaios de vida (*run-to-failure*) da Paderborn University. O dataset compreende medições de vibração síncrona capturadas a 64 kHz sob quatro condições operacionais distintas, onde foram variados os níveis de torque e a velocidade de rotação do eixo.

Diferente de bases de dados com falhas artificiais, este projeto utiliza sinais provenientes de desgaste natural, o que exige um tratamento rigoroso para a identificação de componentes espectrais de baixa energia que precedem a falha funcional.

## Estratégia de Processamento de Sinais

O pipeline de dados foi projetado para transformar sinais brutos de aceleração (medidos em Volts e posteriormente convertidos para unidades de gravidade, g) em tensores de entrada para a rede neural.

O processo envolve:
- Segmentação dos sinais em janelas temporais fixas
- Aplicação de downsampling calculado para otimizar o custo computacional sem violar o teorema de Nyquist

Durante o pré-processamento, é mantida a distinção entre:
- Variáveis de controle: setpoints de carga e velocidade
- Variáveis de processo: aceleração real medida pelos sensores (horizontal frontal e traseiro)

Isso garante que o modelo aprenda a relação causal entre o regime de trabalho e a taxa de degradação.

## Arquitetura e Implementação

A arquitetura implementada utiliza camadas de convolução para a extração automática de atributos no domínio do tempo e da frequência.

Organização do projeto:
- `src/`: código fonte principal
  - `preprocessing.py`: leitura de arquivos `.mat` e estruturação dos DataFrames
- `notebooks/`: análise exploratória e validação espectral das frequências de defeito (pista interna, externa e elementos rolantes)

## Orientações de Uso

Para reproduzir os resultados:

1. Alocar os arquivos brutos no diretório `data/raw/`
2. Configurar o ambiente com `requirements.txt`
3. Executar o script principal para:
   - Limpeza dos dados
   - Normalização
   - Exportação em `.parquet` ou `.csv`

Os dados processados são então utilizados para treinamento e validação com métricas de regressão como MAE e RMSE.

## Identificação Acadêmica

Projeto desenvolvido por **Eduardo Kanadani Bergamo**  
Orientação: Prof. Dr. Pedro Fernando Poveda  
Instituição: Instituto Federal de Educação, Ciência e Tecnologia de São Paulo (IFSP) – Campus São Paulo  

Licença: MIT
