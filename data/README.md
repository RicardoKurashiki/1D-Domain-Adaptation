Baixar os dados dos datasets originais para:

`data/original/[DATASET]`

Exemplo:

`data/original/kermany/`

Após o download, usar os scripts de processamento dentro de `scripts/`, onde eles vão gerar os arquivos para treinamento já específicos do estudo. Exemplo:

`data/processed/kermany/`

O script vai gerar os dados com o split já definido do jeito do artigo, fazendo as mudanças necessárias para os datasets utilizados originalmente. Caso seja necessário inserir um novo dataset, é necessário criar um novo script específico para ele.

Fontes dos dados:
- Kermany dataset: https://data.mendeley.com/datasets/rscbjbr9sj/3
- RSNA Pneumonia Detection Challenge: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data