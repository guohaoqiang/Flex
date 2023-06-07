# Flex
SpMM: Sparse-dense matrix multiplication

## Complile
- Run `./compile.sh`


## Run
- Run `./run.sh`


## Example Data:
|                |      size     | #non-zeros |
|:--------------:|:-------------:|:----------:|
| Sparse(Pubmed) | 19717 X 19717 |   108365   |
|      Dense     |  19717 X 128  |            |


## Output Row Mapping
| Row ID | tiling size |
|:------:|:-----------:|
|    0   |    4 X 4    |
|    1   |    8 X 4    |
|    2   |    16 X 4   |
|    3   |    32 X 4   |
|    4   |    64 X 4   |
|    5   |   128 X 4   |
|    6   |   256 X 4   |
|    7   |    4 X 8    |
|    8   |    8 X 8    |
|    9   |    16 X 8   |
|   10   |    32 X 8   |
|   11   |    64 X 8   |
|   12   |   128 X 8   |
|   13   |   256 X 8   |
|   14   |    4 X 16   |
|   15   |    8 X 16   |
|   16   |   16 X 16   |
|   17   |   32 X 16   |
|   18   |   64 X 16   |
|   19   |   128 X 16  |
|   20   |   256 X 16  |
|   21   |    4 X 32   |
|   22   |    8 X 32   |
|   23   |   16 X 32   |
|   24   |   32 X 32   |
|   25   |   64 X 32   |
|   26   |   128 X 32  |
|   27   |   256 X 32  |
