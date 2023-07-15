/*

 This file is used to read mtx (https://sparse.tamu.edu/) and convert it to CSR format.
 It is derived from https://github.com/SuperScientificSoftwareLaboratory/TileSpGEMM 

*/


#include <iostream>
#include <fstream>
#include <string>
#include "mmio.h"

using namespace std;

typedef struct 
{
    int m;
    int n;
    int nnz;
    int isSymmetric;
    float *value;
    int *columnindex;
    int *rowpointer;
    int tilem;
    int tilen;
    int *tile_ptr;
    int *tile_columnidx;
    int *tile_rowidx;
    int *tile_nnz;
    int numtile;
    float *tile_csr_Value;
    unsigned char *tile_csr_Col;
    unsigned char *tile_csr_Ptr;
    unsigned short *mask;
    int *csc_tile_ptr;
    int *csc_tile_rowidx;
}SMatrix;


void exclusive_scan(int *input, int length)
{
    if(length == 0 || length == 1)
        return;
    
    int old_val, new_val;
    
    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}
int mmio_allinone(int *m, int *n, int *nnz, int *isSymmetric, 
                  int **csrRowPtr, int **csrColIdx, float **csrVal, 
                  char *filename)
{
    int m_tmp, n_tmp;
    int nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    float *csrVal_tmp    = (float *)malloc(nnz_mtx_report * sizeof(float));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp+1);

    int *csrRowPtr_alias = (int *)malloc((m_tmp+1) * sizeof(int));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    float *csrVal_alias    = (float *)malloc(nnz_tmp * sizeof(float));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {            
            int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }
    
    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrVal = csrVal_alias;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}
void writeCSR2csv(char* filename, int m, int n, int* rowpointer, 
			int* columnindex, 
			float* value, int nnz){
	std::ofstream myFile(filename);

	for (int i=0; i<m+1; ++i){
    	myFile << rowpointer[i];
    	if (i<m)	myFile << ",";
    }
    myFile << "\n";
	
    for (int i=0; i<nnz; ++i){
    	myFile << columnindex[i];
    	if (i<nnz-1)	myFile << ",";
    }
    myFile << "\n";

    for (int i=0; i<nnz; ++i){
    	myFile << value[i];
    	if (i<nnz-1)	myFile << ",";
    }
    myFile << "\n";
    myFile.close();
}
int main(int argc, char ** argv){
	char* filename = argv[1];
    std::cout<<"MAT: "<<filename<<std::endl;

    SMatrix *matrixA = (SMatrix *)malloc(sizeof(SMatrix));
    mmio_allinone(&matrixA->m, &matrixA->n, &matrixA->nnz, &matrixA->isSymmetric, &matrixA->rowpointer, &matrixA->columnindex, &matrixA->value, filename);


	std::cout<<"m = "<<matrixA->m<<",   n = "<<matrixA->n<<std::endl;
    //for (int i=0; i<20; ++i){
    //	std::cout<<matrixA->rowpointer[i]<<" ";
    //}
    //std::cout<<std::endl;
	
    //for (int i=0; i<30; ++i){
    //	std::cout<<matrixA->columnindex[i]<<" ";
    //}
    //std::cout<<std::endl;

    //for (int i=0; i<30; ++i){
    //	std::cout<<matrixA->value[i]<<" ";
    //}
    //std::cout<<std::endl;

    writeCSR2csv(argv[2], matrixA->m, matrixA->n, matrixA->rowpointer, matrixA->columnindex, matrixA->value, matrixA->nnz);
    return 0;
}











