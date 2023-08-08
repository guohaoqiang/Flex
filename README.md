# Flex
SpMM: Sparse-dense matrix multiplication

## Complile
- Run `./compile.sh`


## Run
- Run `./run.sh`


## Example Data:
|  ASpT  |        Size       | #Non-zeros |
|:------:|:-----------------:|:----------:|
| Pubmed |   19717 X 19717   |   108365   |
| Flickr |   89250 X 89250   |   989006   |
| Reddit |  232965 X 232965  |  23446803  |
|   PPI  |   14755 X 14755   |   458973   |
| Amazon | 1569960 X 1569960 |  264339468 |
|  Yelp  |  716847 X 716847  |  13954819  |



## Vertex Ordering
The code for DEG,RCM and Gorder was taken from [here](https://github.com/lecfab/rescience-gorder)

## [ASpT](http://gitlab.hpcrl.cse.ohio-state.edu/chong/ppopp19_ae) results
|    ASpT   |    3090    |        |          |    4090    |         |          |
|:---------:|:----------:|:------:|----------|:----------:|:-------:|----------|
|           | tPre/tElap | GFlops | Errs (%) | tPre/tElap |  GFlops | Errs (%) |
|   Pubmed  |    92.5    | 311.39 | 0.005    |    18.7    |  639.8  | 0.005    |
|   Flickr  |    2.84    |  499.5 | 0.001    |     4.5    |  1308.2 | 0.0011   |
|   Reddit  |    1.149   | 259.35 | 97.071   |    3.97    |  1100.8 | 97.077   |
|    PPI    |    7.77    | 671.01 | 0.0067   |    8.17    |  1182.9 | 0.0067   |
|   Amazon  |   21.415   | 284.49 | 90.25    |   329.07   | 1150.06 | 90.25    |
|    Yelp   |  0.212705  | 470.03 | 0.00014  |     0.3    | 1135.66 | 0.0001   |

