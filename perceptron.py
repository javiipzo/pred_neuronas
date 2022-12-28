import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
#-------------------------------------  
#    OBSERVACIONES Y PREDICCIONES  
#-------------------------------------  

observaciones_entradas = ([
[1, 0],
[1, 1],
[0, 1],
[1, 0]])

predicciones = ([[0],[1], [0],[0]]) 

random.seed(1)
limiteMin = -1
limiteMax = 1

w11 = (limiteMax-limiteMin) * random.random() + limiteMin
w21 = (limiteMax-limiteMin) * random.random() + limiteMin
w31 = (limiteMax-limiteMin) * random.random() + limiteMin

sesgo = 1
wb = 0 

#Tasa de aprendizaje  
tsAprendizaje = 0.1

#Cantidad de epoch  
epochs = 300

#--------------------------------------  
#       FUNCIONES ÚTILES  
#--------------------------------------  

def suma_ponderada(X1,W11,X2,W21,B,WB):
    return (B*WB+( X1*W11 + X2*W21))


def funcion_activacion_sigmoide(valor_suma_ponderada):
    return (1 / (1 + 10^(-valor_suma_ponderada)))

def funcion_activacion_relu(valor_suma_ponderada):
    return (max(0,valor_suma_ponderada))


def error_lineal(valor_esperado, valor_predicho):
    return (valor_esperado-valor_predicho)


def calculo_gradiente(valor_entrada,prediccion,error):
    return (-1 * error * prediccion * (1-prediccion) *
valor_entrada)


def calculo_valor_ajuste(valor_gradiente,
tasa_aprendizaje):
    return (valor_gradiente*tasa_aprendizaje)


def calculo_nuevo_peso (valor_peso, valor_ajuste):
    return (valor_peso - valor_ajuste)


def calculo_MSE(predicciones_realizadas, 
predicciones_esperadas):
    i=0
    for prediccion in predicciones_esperadas:
        diferencia = predicciones_esperadas [i] - predicciones_realizadas[i]
        cuadradoDiferencia = diferencia * diferencia
        suma = suma + cuadradoDiferencia
        media_cuadratica = 1 / (len(predicciones_esperadas)) * suma
    return media_cuadratica 

#--------------------------------------  
#    APRENDIZAJE  
#--------------------------------------  
Grafica_MSE=[]
for epoch in range(0,epochs):
    print("EPOCH ("+str(epoch)+"/"+str(epochs)+")")
    predicciones_realizadas_durante_epoch = []
    predicciones_esperadas = []
    numObservacion = 0
    for observacion in observaciones_entradas:
        #Carga de la capa de entrada  
        x1 = observacion[0]
        x2 = observacion[1]

        #Valor de predicción esperado  
        valor_esperado = predicciones[numObservacion][0]

        #Etapa 1: Cálculo de la suma ponderada  
        valor_suma_ponderada = suma_ponderada(x1,w11,x2,w21,sesgo,wb)

        #Etapa 2: Aplicación de la función de activación  
        valor_predicho = funcion_activacion_sigmoide(valor_suma_ponderada)


        #Etapa 3: Cálculo del error  
        valor_error = error_lineal(valor_esperado,valor_predicho)


        #Actualización del peso 1  
        #Cálculo del gradiente del valor de ajuste y del  
        gradiente_W11 = calculo_gradiente(x1,valor_predicho,valor_error)
        valor_ajuste_W11 = calculo_valor_ajuste(gradiente_W11,tsAprendizaje)
        w11 = calculo_nuevo_peso(w11,valor_ajuste_W11)

        # Actualización del peso 2  
        gradiente_W21 = calculo_gradiente(x2, valor_predicho, 
        valor_error)
        valor_ajuste_W21 = calculo_valor_ajuste(gradiente_W21, tsAprendizaje)
        w21 = calculo_nuevo_peso(w21, valor_ajuste_W21)

        # Actualización del peso del sesgo  
        gradiente_Wb = calculo_gradiente(sesgo, valor_predicho,
        valor_error)
        valor_ajuste_Wb = calculo_valor_ajuste(gradiente_Wb, tsAprendizaje)
        wb = calculo_nuevo_peso(wb, valor_ajuste_Wb)

        print("EPOCH (" + str(epoch) + "/" + str(epochs) + ") -Observacion: " + str(numObservacion+1) + "/" + str(len(observaciones_entradas)))

        #Almacenamiento de la predicción realizada:  
        predicciones_realizadas_durante_epoch.append(valor_predicho)

        predicciones_esperadas.append(predicciones[numObservacion][0])
        #Paso a la observación siguiente 
        numObservacion = numObservacion+1

    MSE = calculo_MSE(predicciones_realizadas_durante_epoch,
    predicciones)
    Grafica_MSE.append(MSE[0])
    print("MSE: "+str(MSE)) 

plt.plot(Grafica_MSE)
plt.ylabel('MSE')
plt.show() 

#Cantidad de épocas  
epochs = 300000 

print ("Pesos finales: " )
print ("W11 = "+str(w11))
print ("W21 = "+str(w21))
print ("Wb = "+str(wb)) 

print()
print("--------------------------")
print ("PREDICCIÓN ")
print("--------------------------")
x1 = 0
x2 = 1

#Etapa 1: Cálculo de la suma ponderada  
valor_suma_ponderada = suma_ponderada(x1,w11,x2,w21,sesgo,wb)

#Etapa 2: Aplicación de la función de activación  
valor_predicho = funcion_activacion_sigmoide(valor_suma_ponderada)

print("Predicción del [" + str(x1) + "," + str(x2) + "]")
print("Predicción = " + str(valor_predicho)) 


