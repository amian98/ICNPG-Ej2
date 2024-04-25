#include <cuda_runtime.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <stdexcept>
#include "cpu_timer.h"
#include "gpu_timer.h"
#include <fstream>
#include <algorithm>



struct kernel_shape{
    dim3 threads;
    dim3 blocks;
};

template<typename T>
__global__ void kernel1d(T *A, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    int j;

    if(idx < N*N){
        j = idx/N;
        i = idx%N;
        A[i+N*j] = cosf(i)*sinf(j);
//idx += blockDim.x * gridDim.x;
    }
}

template<typename T>
__global__ void kernel2d(T *A, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

//    while (i < N){
//        while (j<N){
    if (i < N && j < N){
            A[i+N*j] = cosf(i)*sinf(j);
//            j += blockDim.y * gridDim.y;
//        }
//        i += blockDim.x * gridDim.x;
    }
}

__host__ kernel_shape get_kernel_shape(dim3 system_shape, bool forzar_1_d = false){
    kernel_shape shape;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    dim3 gpu_threads = dim3(prop.maxThreadsPerBlock);
    dim3 gpu_blocks = dim3(prop.multiProcessorCount);

    if (forzar_1_d){
        shape.threads = dim3(1);
        shape.blocks = dim3(1);
        int N = system_shape.x * system_shape.y * system_shape.z;
        if (N > gpu_threads.x){
            shape.blocks.x = N / gpu_threads.x + (N % gpu_threads.x == 0 ? 0 : 1);
            shape.threads.x = gpu_threads.x;
        } else {
            shape.threads.x = N;
        }
        
    } else {
        // Si el sistema cabe en un bloque, lanzar solo un bloque
        if (system_shape.x <= gpu_threads.x && system_shape.y <= gpu_threads.y && system_shape.z <= gpu_threads.z){
            shape.threads = system_shape;
            shape.blocks = dim3(1);
        // Si no cabe en un bloque, entonces calcular la cantidad de bloques necesarios
        } 
        else {
            shape.blocks.x = system_shape.x / gpu_threads.x + (system_shape.x % gpu_threads.x == 0 ? 0 : 1);
            shape.blocks.y = system_shape.y / gpu_threads.y + (system_shape.y % gpu_threads.y == 0 ? 0 : 1);
            shape.blocks.z = system_shape.z / gpu_threads.z + (system_shape.z % gpu_threads.z == 0 ? 0 : 1);
            shape.threads = gpu_threads;
        }
    }
    
    return shape;
}

template<typename T>
__host__ void calcular_host(T *A, int N){
    for (int i=0; i<N*N; i++){
        int x = i%N;
        int y = i/N;
        A[i] = cosf(x)*sinf(y);
    }
}

int main(int argc, char *argv[]){

    printf("Verificando tarjeta de video...\n");
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No se encontraron dispositivos CUDA" << std::endl;
        return -1;
    }

    



    // Verificar argumentos
    // if (argc < 2 || argc > 4){
    //     printf("Usage: %s X\n", argv[0]);
    //     printf("       %s X Y\n", argv[0]);
    //     printf("       %s X Y Z\n", argv[0]);
    //     return 1;
    // }

    printf("Verificando argumentos...\n");

    if (argc != 2){
        printf("Usage: %s X\n", argv[0]);
        return 1;
    }
    std::string length;
    int X=1, Y=1, Z=1;
    if (argc >= 2){
        length = argv[1];
        try{
            X = std::stoi(length);
        } catch (std::invalid_argument){
            printf("Invalid argument for X\n");
            return 1;
        } catch (std::out_of_range){
            printf("Out of range for X\n");
            return 1;
        } catch (...){
            printf("Unknown error for X\n");
            return 1;
        } 
    } else if (argc == 3){
        length = argv[2];
        try{
            Y = std::stoi(length);
        } catch (std::invalid_argument){
            printf("Invalid argument for Y\n");
            return 1;
        } catch (std::out_of_range){
            printf("Out of range for Y\n");
            return 1;
        } catch (...){
            printf("Unknown error for Y\n");
            return 1;
        } 
    } else {
        length = argv[3];
        try{
            Z = std::stoi(length);
        } catch (std::invalid_argument){
            printf("Invalid argument for Z\n");
            return 1;
        } catch (std::out_of_range){
            printf("Out of range for Z\n");
            return 1;
        } catch (...){
            printf("Unknown error for Z\n");
            return 1;
        } 
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::string nombreArchivo = prop.name;
    std::replace_if(nombreArchivo.begin(), nombreArchivo.end(), [](char c) {
        return c == ' ';
    }, '_');

    // nombreArchivo += (std::string) "_info.txt";

    std::ofstream gpu_data(nombreArchivo + (std::string) "_info.txt");

    // Comprobar si se abrió correctamente
    if (!gpu_data.is_open()) {
        std::cerr << "Error al abrir el archivo" << std::endl;
        return 1; // Salir del programa con código de error
    }
    
    gpu_data << "Nombre: " << prop.name << std::endl;
    gpu_data << "Capacidad de cómputo: " << prop.major << "." << prop.minor << std::endl;
    gpu_data << "Memoria global: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    gpu_data << "Reloj de núcleo: " << prop.clockRate / 1000 << " MHz" << std::endl;
    gpu_data << "Hilos por bloque: " << prop.maxThreadsPerBlock << std::endl;
    gpu_data << "Multiprocesadores: " << prop.multiProcessorCount << std::endl;
    gpu_data.close();

    Y = X; // para patchear la generalidad de arriba

    printf("Se generará una matriz de %d x %d\n", X, Y);

////////////////////////////////////////////////////////////////////////////////////////////
//                                       Caso float                                       //
////////////////////////////////////////////////////////////////////////////////////////////
    printf("Caso float\n");
    printf("Iniciando...\n");
    printf("Reservando memoria...\n");
    int N = X*Y*Z;
    float *h_A1 = (float*)malloc(N*sizeof(float));
    float *h_A2 = (float*)malloc(N*sizeof(float));
    float *d_A;
    float *h_C = (float*)malloc(N*sizeof(float));
    cudaMalloc(&d_A, N*sizeof(float));

    gpu_timer gt1, gt2, gt3, gt4;
    cpu_timer ct1, ct2;

    // Lanzar kernel
    printf("Lanzando kernel...\n");
    kernel_shape shape1 = get_kernel_shape(dim3(X,Y,Z), true);
    kernel_shape shape2 = get_kernel_shape(dim3(X,Y,Z));
    
    gt1.tic();
    kernel1d<float><<<shape1.blocks, shape1.threads>>>(d_A, X);
    // Copiar resultados
    cudaMemcpy(h_A1, d_A, N*sizeof(float), cudaMemcpyDeviceToHost);
    double t1 = gt1.tac();
    gt2.tic();
    kernel2d<float><<<shape2.blocks, shape2.threads>>>(d_A, X);
    // Copiar resultados
    cudaMemcpy(h_A2, d_A, N*sizeof(float), cudaMemcpyDeviceToHost);
    double t2 = gt2.tac();
    // cudaDeviceSynchronize();

    printf("Tiempo 1D: %f\n", t1);
    printf("Tiempo 2D: %f\n", t2);



    // Calcular en el host
    printf("Calculando en el host...\n");
    ct1.tic();
    calcular_host(h_C, X);
    double t3 = ct1.tac();
    printf("Tiempo host: %f\n", t3);

    // Calcular error
    printf("Calculando error...\n");
    float error = 0;
    for (int i=0; i<X; i++){
        for (int j=0; j<X; j++){
                error += abs(h_A1[i+X*j] - h_C[i+X*j]) + abs(h_A2[i+X*j] - h_C[i+X*j]);
        }
    }

    printf("Error: %f\n", error);
    
    // Liberar memoria
    free(h_A1);
    free(h_A2);
    free(h_C);
    cudaFree(d_A);
    printf("\n\n");

////////////////////////////////////////////////////////////////////////////////////////////
//                                       Caso double                                      //
////////////////////////////////////////////////////////////////////////////////////////////
    printf("Caso double\n");
    printf("Iniciando...\n");
    printf("Reservando memoria...\n");

    double *h_B1 = (double*)malloc(N*sizeof(double));
    double *h_B2 = (double*)malloc(N*sizeof(double));
    double *h_B = (double*)malloc(N*sizeof(double));
    double *d_B;
    cudaMalloc(&d_B, N*sizeof(double));

    // Lanzar kernel
    printf("Lanzando kernel...\n");
    gt3.tic();
    kernel1d<double><<<shape1.blocks, shape1.threads>>>(d_B, X);
    cudaMemcpy(h_B1, d_B, N*sizeof(double), cudaMemcpyDeviceToHost);
    double t4 = gt3.tac();
    gt4.tic();
    kernel2d<double><<<shape2.blocks, shape2.threads>>>(d_B, X);
    cudaMemcpy(h_B2, d_B, N*sizeof(double), cudaMemcpyDeviceToHost);
    double t5 = gt4.tac();
    // cudaDeviceSynchronize();

    printf("Tiempo 1D: %f\n", t4);
    printf("Tiempo 2D: %f\n", t5);



    // Calcular en el host
    printf("Calculando en el host...\n");
    ct2.tic();
    calcular_host(h_B, X);
    double t6 = ct2.tac();
    printf("Tiempo host: %f\n", t6);

    float error2 = 0;
    printf("Calculando error...\n");
    for (int i=0; i<X; i++){
        for (int j=0; j<Y; j++){
            error2 += abs(h_B1[i+Y*j] - h_B[i+Y*j]) + abs(h_B2[i+Y*j] - h_B[i+Y*j]);
        }
    }

    printf("Error: %f\n", error2);

    // Liberar memoria
    free(h_B);
    free(h_B1);
    free(h_B2);
    cudaFree(d_B);

    // sacar datos por fichero
    std::ofstream data(nombreArchivo + (std::string) "_data.txt", std::ios::app);
    
    // Comprobar si se abrió correctamente
    if (!data.is_open()) {
        std::cerr << "Error al abrir el archivo" << std::endl;
        return 1; // Salir del programa con código de error
    }

    data << X << " " << t1 << " " << t2 << " " << t3 << " " << error << " " << t4 << " " << t5 << " " << t6 << " " << error2 << std::endl;
    data.close();
    return 0;
}