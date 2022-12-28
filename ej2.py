import tensorflow as tf
import matplotlib.pyplot as plt
#-------------------------------------  
#    DATOS DE APRENDIZAJE  
#-------------------------------------  

valores_entradas_X = [[1., 0.], [1., 1.], [0., 1.], [0., 0.]]
valores_a_predecir_Y = [[0.], [1.], [0.], [0.]]

#-------------------------------------  
#    PARÁMETROS DE LA RED  
#-------------------------------------  
#Variable TensorFLow correspondiente a los valores neuronas 
#de entrada  
tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])

#Variable TensorFlow correspondiente a la neurona de salida  

tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])

#-- Pesos --  
#Creación de una variable TensorFlow del tipo tabla  
#que contiene 3 líneas con valores de tipo decimal  
#Estos valores se inicializan de manera aleatoria  
pesos = tf.Variable(tf.random_normal([2, 1]), tf.float32)

#-- Sesgo inicializado a 0 --  
sesgo = tf.Variable(tf.zeros([1, 1]), tf.float32)

#La suma ponderada es en la práctica una multiplicación de matrices 
#entre los valores en la entrada X y los distintos pesos  
#la función matmul se encarga de hacer esta multiplicación  
sumaponderada = tf.matmul(tf_neuronas_entradas_X,pesos)

#Adición del sesgo a la suma ponderada  
sumaponderada = tf.add(sumaponderada,sesgo)

#Función de activación de tipo sigmoide que permite calcular 
#la predicción  
prediccion = tf.sigmoid(sumaponderada)

#Función de error de media cuadrática MSE  
funcion_error = tf.reduce_sum(tf.pow(tf_valores_reales_Y-prediccion,2))

#Descenso de gradiente con una tasa de aprendizaje fijada en 0,1  
optimizador =tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(funcion_error) 


#-------------------------------------  
#    APRENDIZAJE  
#-------------------------------------  

#Cantidad de epochs  
epochs = 10000

#Inicialización de las variables  
init = tf.global_variables_initializer()

#Inicio de una sesión de aprendizaje  
sesion = tf.Session()
sesion.run(init)

#Para la realización de la gráfica para la MSE  
Grafica_MSE=[]


#Para cada epoch  
for i in range(epochs):

    #Realización del aprendizaje con actualización de los pesos  
    sesion.run(optimizador, feed_dict = {tf_neuronas_entradas_X:valores_entradas_X, tf_valores_reales_Y:valores_a_predecir_Y})

    #Calcular el error  
    MSE = sesion.run(funcion_error, feed_dict = 
    {tf_neuronas_entradas_X: valores_entradas_X,
    tf_valores_reales_Y:valores_a_predecir_Y})

    #Visualización de las informaciones  
    Grafica_MSE.append(MSE)
    print("EPOCH (" + str(i) + "/" + str(epochs) + ") - MSE: "+
    str(MSE))


#Visualización gráfica  
plt.plot(Grafica_MSE)
plt.ylabel('MSE')
plt.show() 

print("--- VERIFICACIONES ----")

for i in range(0,4):
    print("Observación:"+str(valores_entradas_X[i])+ " - Esperado: "+str(valores_a_predecir_Y[i])+" - Predicción: "+str(session.run(prediccion, feed_dict={tf_neuronas_entradas_X: [valores_entradas_X[i]]}))) 


