#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    //printf("USTH ICT Master 2017, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    //printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    //printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU elapsed %.1fms\n", timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            printf("labwork 1 OpenMP elapsed %.1fms\n", timer.getElapsedTimeInMilliSec());
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            if (labwork.labwork3_GPU()) {
                labwork.saveOutputImage("labwork3-gpu-out.jpg");
                printf("labwork 3 elapsed %.1fms\n", timer.getElapsedTimeInMilliSec());
            }
            break;
        case 4:
            if (labwork.labwork4_GPU()) {
                labwork.saveOutputImage("labwork4-gpu-out.jpg");
                printf("labwork 4 elapsed %.1fms\n", timer.getElapsedTimeInMilliSec());
            }
            break;
        case 5:
            if (labwork.labwork5_GPU()) {
                labwork.saveOutputImage("labwork5-gpu-out.jpg");
                printf("labwork 5 elapsed %.1fms\n", timer.getElapsedTimeInMilliSec());
            } else {
                printf("error\n");
            }
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    //printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int numDevices;
    if (cudaGetDeviceCount(&numDevices) != cudaSuccess) {
        fprintf(stderr, "cannot get number of devices\n");
        return;
    }
    printf("%d devices found\n", numDevices);
    for (int i = 0; i < numDevices; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            fprintf(stderr, "cannot get device props\n");
            return;
        }
        printf("Information for device %d:\n", i);
        printf("Device name: %s\n", prop.name);
        int cores = getSPcores(prop);
        printf("Core count: %d\n", cores);
        printf("Core clock rate: %d kHz\n", prop.clockRate);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Warp size: %d threads\n", prop.warpSize);
        printf("Memory clock rate: %d kHz\n", prop.memoryClockRate);
        printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("\n");
    }
}

__global__ void labwork3(uchar3 * __restrict__ input, uchar3 * __restrict__ output, long long pixelCount) {
    long long i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < pixelCount) {
        output[i].x = (char)(((int)input[i].x + input[i].y + input[i].z) / 3);
        output[i].y = output[i].z = output[i].x;
    }
}

int Labwork::labwork3_GPU() {
    long long pixelCount = inputImage->width * inputImage->height;
    char *blockSizeEnv = getenv("LW3_CUDA_BLOCK_SIZE");
    if (!blockSizeEnv) {
        fprintf(stderr, "invalid block size\n");
        return 0;
    }
    int blockSize = atoi(blockSizeEnv);
    long long numBlocks = pixelCount / blockSize + 1;

    uchar3 *inputCudaBuffer;
    if (cudaMalloc(&inputCudaBuffer, pixelCount * sizeof(uchar3)) != cudaSuccess) {
        fprintf(stderr, "memory allocation error\n");
        return 0;
    }
    uchar3 *outputCudaBuffer;
    if (cudaMalloc(&outputCudaBuffer, pixelCount * sizeof(uchar3)) != cudaSuccess) {
        fprintf(stderr, "memory allocation error\n");
        return 0;
    }

    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    if (cudaMemcpy(inputCudaBuffer, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "input buffer copy error\n");
        return 0;
    }
    for (int j = 0; j < 100; j++) {
        labwork3<<<numBlocks, blockSize>>>(inputCudaBuffer, outputCudaBuffer, pixelCount);
    }
    cudaDeviceSynchronize();
    if (cudaMemcpy(outputImage, outputCudaBuffer, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "output buffer copy error\n");
        return 0;
    }

    cudaFree(inputCudaBuffer);
    cudaFree(outputCudaBuffer);

    return 1;
}

__global__ void labwork4(uchar3 * __restrict__ input, uchar3 * __restrict__ output, long long pixelCount, int width) {
    long row = blockIdx.y * gridDim.y + threadIdx.y;
    long long i = row * width + threadIdx.x;
    if (i < pixelCount) {
        output[i].x = (char)(((int)input[i].x + input[i].y + input[i].z) / 3);
        output[i].y = output[i].z = output[i].x;
    }
}

int Labwork::labwork4_GPU() {
    long long pixelCount = inputImage->width * inputImage->height;
    char *blockSizeEnv = getenv("LW4_CUDA_BLOCK_SIZE");
    if (!blockSizeEnv) {
        fprintf(stderr, "invalid block size\n");
        return 0;
    }
    int blockSize = atoi(blockSizeEnv);

    long gridWidth = (inputImage->width + blockSize - 1) / blockSize;
    long gridHeight = (inputImage->width + blockSize - 1) / blockSize;
    dim3 gdim(gridWidth, gridHeight);
    dim3 bdim(blockSize, blockSize);

    uchar3 *inputCudaBuffer;
    if (cudaMalloc(&inputCudaBuffer, pixelCount * sizeof(uchar3)) != cudaSuccess) {
        fprintf(stderr, "memory allocation error\n");
        return 0;
    }
    uchar3 *outputCudaBuffer;
    if (cudaMalloc(&outputCudaBuffer, pixelCount * sizeof(uchar3)) != cudaSuccess) {
        fprintf(stderr, "memory allocation error\n");
        return 0;
    }

    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    if (cudaMemcpy(inputCudaBuffer, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "input buffer copy error\n");
        return 0;
    }
    for (int j = 0; j < 100; j++) {
        labwork4<<<gdim, bdim>>>(inputCudaBuffer, outputCudaBuffer, pixelCount, inputImage->width);
    }
    cudaDeviceSynchronize();
    if (cudaMemcpy(outputImage, outputCudaBuffer, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "output buffer copy error\n");
        return 0;
    }

    cudaFree(inputCudaBuffer);
    cudaFree(outputCudaBuffer);

    return 1;
}

__global__ void labwork5(uchar3 * __restrict__ input, uchar3 * __restrict__ output, long pixelCount, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height || col >= width) {
        return;
    }
    long i = row * width + col; // position in image pixel stream
    unsigned char kernel[7][7] = { // [r][c]
        { 0, 0, 1, 2, 1, 0, 0 },
        { 0, 3, 13, 22, 13, 3, 0 },
        { 1, 13, 59, 97, 59, 13, 1 },
        { 2, 22, 97, 159, 97, 22, 2 },
        { 1, 13, 59, 97, 59, 13, 1 },
        { 0, 3, 13, 22, 13, 3, 0 },
        { 0, 0, 1, 2, 1, 0, 0 },
    };
    long sum = 0;
    long acc = 0; // sum accumulator
    for (int v = 0; v < 7; v++) { // r
        for (int u = 0; u < 7; u++) { // c
            int p = col + u - 3;
            int q = row + v - 3;
            if (p < 0 || p >= width || q < 0 || q >= height) {
            } else {
                long j = q * width + p;
                int gray = ((int)input[j].x + input[j].y + input[j].z) / 3;
                acc += gray * kernel[v][u];
                sum += kernel[v][u];
            }
        }
    }
    output[i].x = output[i].y = output[i].z = (unsigned char)(acc / sum);
}

void labwork5cpu(uchar3 * __restrict__ input, uchar3 * __restrict__ output, long pixelCount, int width, int height, int blocksize) {
    unsigned char kernel[7][7] = { // [r][c]
        { 0, 0, 1, 2, 1, 0, 0 },
        { 0, 3, 13, 22, 13, 3, 0 },
        { 1, 13, 59, 97, 59, 13, 1 },
        { 2, 22, 97, 159, 97, 22, 2 },
        { 1, 13, 59, 97, 59, 13, 1 },
        { 0, 3, 13, 22, 13, 3, 0 },
        { 0, 0, 1, 2, 1, 0, 0 },
    };
    for (int gy = 0; gy < (height + blocksize - 1) / blocksize; gy++) {
        for (int gx = 0; gx < (width + blocksize - 1) / blocksize; gx++) {
            for (int ty = 0; ty < blocksize; ty++) {
                for (int tx = 0; tx < blocksize; tx++) {
    int row = gy * blocksize + ty;
    int col = gx * blocksize + tx;
    if (row >= height || col >= width) {
        return;
    }
    long i = row * width + col; // position in image pixel stream
    long sum = 0;
    long acc = 0; // sum accumulator
    for (int v = 0; v < 7; v++) { // r
        for (int u = 0; u < 7; u++) { // c
            int p = col + u - 3;
            int q = row + v - 3;
            if (p < 0 || p >= width || q < 0 || q >= height) {
            } else {
                long j = q * width + p;
                int gray = ((int)input[j].x + input[j].y + input[j].z) / 3;
                acc += gray * kernel[v][u];
                sum += kernel[v][u];
            }
        }
    }
    output[i].x = output[i].y = output[i].z = (unsigned char)(acc / sum);
                }
            }
        }
    }
}

int Labwork::labwork5_GPU() {
    long long pixelCount = inputImage->width * inputImage->height;
    char *blockSizeEnv = getenv("LW5_CUDA_BLOCK_SIZE");
    if (!blockSizeEnv) {
        fprintf(stderr, "invalid block size\n");
        return 0;
    }
    int blockSize = atoi(blockSizeEnv);

    long gridWidth = (inputImage->width + blockSize - 1) / blockSize;
    long gridHeight = (inputImage->width + blockSize - 1) / blockSize;
    dim3 gdim(gridWidth, gridHeight);
    dim3 bdim(blockSize, blockSize);

    uchar3 *inputCudaBuffer;
    if (cudaMalloc(&inputCudaBuffer, pixelCount * sizeof(uchar3)) != cudaSuccess) {
        fprintf(stderr, "memory allocation error\n");
        return 0;
    }
    uchar3 *outputCudaBuffer;
    if (cudaMalloc(&outputCudaBuffer, pixelCount * sizeof(uchar3)) != cudaSuccess) {
        fprintf(stderr, "memory allocation error\n");
        return 0;
    }

    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    ///*
    if (cudaMemcpy(inputCudaBuffer, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "input buffer copy error\n");
        return 0;
    }
    //*/
    for (int j = 0; j < 100; j++) {
        labwork5<<<gdim, bdim>>>(inputCudaBuffer, outputCudaBuffer, pixelCount, inputImage->width, inputImage->height);
        //labwork5cpu((uchar3*)inputImage->buffer, (uchar3*)outputImage, pixelCount, inputImage->width, inputImage->height, blockSize);
    }
    ///*
    cudaDeviceSynchronize();
    if (cudaMemcpy(outputImage, outputCudaBuffer, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "output buffer copy error\n");
        return 0;
    }
    //*/

    cudaFree(inputCudaBuffer);
    cudaFree(outputCudaBuffer);

    return 1;
}

void Labwork::labwork6_GPU() {

}

void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
