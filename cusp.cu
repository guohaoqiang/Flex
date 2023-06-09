#include "cusp.cuh"

int run1(DataLoader& input, Metrics& metric){
    float *gpuB = nullptr; // n * c
    CUDA_CHECK(cudaMalloc(&gpuB, sizeof(float) * input.n * input.c));
    float duration, spmm_duration, gemm_duration;
    cudaEvent_t start, stop, spmm_start, spmm_stop, gemm_start, gemm_stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&spmm_start);
	cudaEventCreate(&spmm_stop);
	cudaEventCreate(&gemm_start);
	cudaEventCreate(&gemm_stop);
    // ############################
    cudaEventRecord(start);
    cudaEventRecord(gemm_start);
    // ############################

    //----------  B = XW : sgemm------------
    const float alpha = 1.0;
    const float beta = 0.0;
    
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    /* step 2: compute */
    CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, input.n, input.c, input.dim, &alpha, 
                input.gpuX, input.n, input.gpuW, input.dim, &beta, gpuB, input.n));
    CUBLAS_CHECK(cublasDestroy(cublasH));
    //LOG(INFO) << "step1 of run1 completed ...";
	cudaEventRecord(gemm_stop);
	cudaEventSynchronize(gemm_stop);

    //----------  C = AB : sparsemm------------
    cudaEventRecord(spmm_start);
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, input.n, input.n, input.nnz,
                                      input.rowPtr_dev, input.col_dev, input.vals_dev,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, input.n, input.c, input.n, gpuB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, input.n, input.c, input.n, input.gpuRef1,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, dBuffer) )
	cudaEventRecord(spmm_stop);
	cudaEventSynchronize(spmm_stop);

    //LOG(INFO) << "step2 of run1 completed ...";
    // ############################
	//cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// ############################
	cudaEventElapsedTime(&duration, start, stop);
	cudaEventElapsedTime(&gemm_duration, gemm_start, gemm_stop);
	cudaEventElapsedTime(&spmm_duration, spmm_start, spmm_stop);
    metric.t += duration;
    metric.spmm_t += spmm_duration;
    metric.gemm_t += gemm_duration;
    metric.flops = (input.nnz * input.c + input.n * input.dim * input.c) * 2;
    metric.spmm_flops = (input.nnz * input.c) * 2;
    metric.gemm_flops = (input.n * input.dim * input.c) * 2;
    //                              A                  X                 W                    B
    metric.dataMovement = 4*(input.nnz + input.n*input.dim + input.dim*input.c + 2*input.n*input.c);
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
    CUDA_CHECK( cudaFree(gpuB) );

    CUDA_CHECK(cudaMemcpy(&input.cpuRef1.data(), input.gpuRef1, sizeof(float)*input.n*input.c, cudaMemcpyDeviceToHost));
    //LOG(INFO) << "run1 completed ...";
    return 0;
}

int run2(DataLoader& input, Metrics& metric){
    float *gpuB = nullptr; // n * dim
    CUDA_CHECK(cudaMalloc(&gpuB, sizeof(float) * input.n * input.dim));
    float duration, spmm_duration, gemm_duration;
    cudaEvent_t start, stop, spmm_start, spmm_stop, gemm_start, gemm_stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&spmm_start);
	cudaEventCreate(&spmm_stop);
	cudaEventCreate(&gemm_start);
	cudaEventCreate(&gemm_stop);
    // ############################
    cudaEventRecord(start);
    // ############################

    //----------  B = AX : sparsemm------------
    cudaEventRecord(spmm_start);
    const float alpha = 1.0;
    const float beta = 0.0;
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, input.n, input.n, input.nnz,
                                      input.rowPtr_dev, input.col_dev, input.vals_dev,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix X
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, input.n, input.dim, input.n, input.gpuX,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, input.n, input.dim, input.n, gpuB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, dBuffer) )
    //LOG(INFO) << "step1 of run2 completed ...";
	cudaEventRecord(spmm_stop);
	cudaEventSynchronize(spmm_stop);
    //----------  C = BW : sgemm------------
    
    cudaEventRecord(gemm_start);
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    /* step 2: compute */
    CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, input.n, input.c, input.dim, &alpha, 
                gpuB, input.n, input.gpuW, input.dim, &beta, input.gpuRef2, input.n));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    //LOG(INFO) << "step2 of run2 completed ...";
	cudaEventRecord(gemm_stop);
	cudaEventSynchronize(gemm_stop);

    // ############################
	//cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// ############################
	cudaEventElapsedTime(&duration, start, stop);
	cudaEventElapsedTime(&spmm_duration, spmm_start, spmm_stop);
	cudaEventElapsedTime(&gemm_duration, gemm_start, gemm_stop);
    metric.t += duration;
    metric.spmm_t += spmm_duration;
    metric.gemm_t += gemm_duration;
    metric.flops = (input.nnz * input.dim + input.n * input.dim * input.c) * 2;
    metric.spmm_flops = (input.nnz * input.dim) * 2;
    metric.gemm_flops = (input.n * input.dim * input.c) * 2;
    //                              A                  X                 W                    B
    metric.dataMovement = 4*(input.nnz + input.n*input.dim + input.dim*input.c + 2*input.n*input.dim);
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
    CUDA_CHECK( cudaFree(gpuB) );

    CUDA_CHECK(cudaMemcpy(&input.cpuRef2.data(), input.gpuRef2, sizeof(float)*input.n*input.c, cudaMemcpyDeviceToHost));
    //LOG(INFO) << "run2 completed ...";
    return 0;
}

