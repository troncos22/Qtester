# Idea 1
### código que permite evaluar Quantum circuits de manera de saber el tiempo que se demorará en ejecutar en distintos tipos de Quantum processor, evaluando costos, cantidad de Qbits utilizados, a la vez que evalua el hardware local para saber si es posible hacer una simulació previa para el ajuste del Quantum Circuit.

este estimador analiza el circuito que se quiere mandar al hardware cuántico, de manera de tener un estimado de cuanto se demorará dependiendo del modelo de procesador, por defoult si no se declara un procesador en específico ¿, se utilizará el heleron 