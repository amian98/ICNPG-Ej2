import numpy as np
import os
from matplotlib import pyplot as plt

def graficar_gpu(path:str):
    gpu_data = np.loadtxt(path)

    gpu_str = "\n".join(open(path[:-8]+"info.txt").readlines())

    L = gpu_data[:, 0]
    gpu_float_1d = gpu_data[:, 1]
    gpu_float_2d = gpu_data[:, 2]
    cpu_float = gpu_data[:, 3]
    error_float = gpu_data[:, 4]
    gpu_double_1d = gpu_data[:, 5]
    gpu_double_2d = gpu_data[:, 6]
    cpu_double = gpu_data[:, 7]
    error_double = gpu_data[:, 8]


    fig, axs = plt.subplots(2,2,figsize=(14, 8))

    axs[0,0].loglog(L, gpu_float_1d,".", label="GPU-Float-1D", color="blue")
    axs[0,0].loglog(L, gpu_float_2d,".", label="GPU-Float-2D", color="green")
    axs[0,0].loglog(L, cpu_float,".", label="CPU-Float", color="red")
    axs[0,0].set_title("Comparación de los tiempos de ejecución del calculo\nen CPU y GPU para  diferentes modelados con tipo de dato float")
    axs[0,0].set_xlabel("$L[u]$")
    axs[0,0].set_ylabel("$t[ms]$")
    axs[0,0].legend()

    axs[0,1].loglog(L, gpu_double_1d,"-", label="GPU-Double-1D", color="blue")
    axs[0,1].loglog(L, gpu_double_2d,"-", label="GPU-Double-2D", color="green")
    axs[0,1].loglog(L, cpu_double,"-", label="CPU-Double", color="red")
    axs[0,1].set_title("Comparación de los tiempos de ejecución del calculo\nen CPU y GPU para  diferentes modelados con tipo de dato double")
    axs[0,1].set_xlabel("$L[u]$")
    axs[0,1].set_ylabel("$t[ms]$")
    axs[0,1].legend()

    axs[1,0].loglog(L, gpu_float_1d,".", label="GPU-Float-1D", color="blue")
    axs[1,0].loglog(L, gpu_double_1d,"-", label="GPU-Double-1D", color="blue")
    axs[1,0].loglog(L, gpu_float_2d,".", label="GPU-Float-2D", color="green")
    axs[1,0].loglog(L, gpu_double_2d,"-", label="GPU-Double-2D", color="green")
    axs[1,0].set_title("Comparación de los tiempos de ejecución del calculo\nen la GPU para diferentes tipos de datos\n y diferentes modelados")
    axs[1,0].set_xlabel("$L[u]$")
    axs[1,0].set_ylabel("$t[ms]$")
    axs[1,0].legend()


    axs[1,1].loglog(L, error_float/L,".", label="$e$ Float", color="yellow")
    axs[1,1].loglog(L, error_double/L,"-", label="$e$ Double", color="orange")
    axs[1,1].set_title("Comparacion del error por elemento en el cálculo para ambos tipos de dato")
    axs[1,1].set_xlabel("$L[u]$")
    axs[1,1].set_ylabel("$e[ms]$")
    axs[1,1].legend()

    fig.suptitle(gpu_str)
    #plt.subplots_adjust(wspace=0.5, hspace=0.5, top=4)
    plt.tight_layout()
    plt.show()


path = os.getcwd()
#archivos = []
for archivo in os.listdir(path):
   if archivo.endswith("data.txt"):
      graficar_gpu(archivo)


def comparar_float():
    gpu_list = []
    for archivo in os.listdir(os.getcwd()):
        if archivo.endswith("data.txt"):
            gpu_list.append(archivo)
    
    fig, axs = plt.subplots(1,1,figsize=(14, 8))
    for archivo in gpu_list:
        gpu_data = np.loadtxt(archivo)

        L = gpu_data[:, 0]
        gpu_float_1d = gpu_data[:, 1]
        axs.loglog(L, gpu_float_1d, label=" ".join(archivo.split("_")[:-1]))

    axs.set_title("Comparación de los tiempos de ejecución del calculo\nen CPU y GPU para  diferentes modelados con tipo de dato float")
    axs.set_xlabel("$L[u]$")
    axs.set_ylabel("$t[ms]$")
    axs.legend()

    plt.show()


def comparar_double():
    gpu_list = []
    for archivo in os.listdir(os.getcwd()):
        if archivo.endswith("data.txt"):
            gpu_list.append(archivo)
    
    fig, axs = plt.subplots(1,1,figsize=(14, 8))
    for archivo in gpu_list:
        gpu_data = np.loadtxt(archivo)

        L = gpu_data[:, 0]
        gpu_double_1d = gpu_data[:, 5]
        axs.loglog(L, gpu_double_1d, label=" ".join(archivo.split("_")[:-1]))

    axs.set_title("Comparación de los tiempos de ejecución del calculo\nen CPU y GPU para  diferentes modelados con tipo de dato double")
    axs.set_xlabel("$L[u]$")
    axs.set_ylabel("$t[ms]$")
    axs.legend()

    plt.show()

comparar_float()
comparar_double()