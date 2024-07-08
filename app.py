import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Image, ttk
from tkinter import filedialog, messagebox
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image, ImageTk

class KMeansApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Implementación propia de KMeans")
        self.root.geometry("1300x720")
        self.set_icon("icon.png")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Botón para cargar el archivo
        self.btnCargar = tk.Button(top_frame, text="Abrir archivo...", command=self.cargarArchivo, fg="black")
        self.btnCargar.grid(row=0, column=0, sticky=tk.W, pady=10)

        # Campo de entrada para mostrar el nombre del archivo seleccionado
        self.nombreArchivo = tk.StringVar()
        self.lblNombreArchivo = tk.Entry(top_frame, textvariable=self.nombreArchivo, state='readonly', width=50)
        self.lblNombreArchivo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=10)

        # Etiqueta para datos crudos
        self.lblDatosCrudos = tk.Label(main_frame, text="Datos Crudos", fg="black", font=("Helvetica", 12))
        self.lblDatosCrudos.grid(row=1, column=0, sticky=tk.W)

        # Marco para mostrar los datos crudos
        self.frmDatosCrudos = ttk.Frame(main_frame)
        self.frmDatosCrudos.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frmDatosCrudos.columnconfigure(0, weight=1)
        self.frmDatosCrudos.rowconfigure(0, weight=1)

        # Tabla para mostrar los datos crudos
        self.tblDatosCrudos = ttk.Treeview(self.frmDatosCrudos)
        self.tblDatosCrudos.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Barra de desplazamiento para la tabla de datos crudos
        self.scrollDatosCrudos = ttk.Scrollbar(self.frmDatosCrudos, orient="vertical", command=self.tblDatosCrudos.yview)
        self.scrollDatosCrudos.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tblDatosCrudos.configure(yscroll=self.scrollDatosCrudos.set)

        # Marco para mostrar la información del atributo seleccionado (Crudo)
        self.frmInfoDatosCrudos = ttk.Labelframe(main_frame, text="Atributo seleccionado (Crudo)", padding="10")
        self.frmInfoDatosCrudos.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        self.frmInfoDatosCrudos.columnconfigure(1, weight=1)

        # Etiquetas y campos de entrada para mostrar estadísticas del atributo seleccionado (Crudo)
        lblsDatosCrudos = ["Nombre del atributo", "Valor mínimo", "Valor máximo", "Media", "Desviación estándar"]
        self.entDatosCrudos = {}
        for i, label in enumerate(lblsDatosCrudos):
            tk.Label(self.frmInfoDatosCrudos, text=label, fg="black", font=("Helvetica", 12)).grid(row=i, column=0, sticky=tk.W)
            var = tk.StringVar()
            entry = tk.Entry(self.frmInfoDatosCrudos, textvariable=var, state='readonly', width=20)
            entry.grid(row=i, column=1, sticky=(tk.W, tk.E))
            self.entDatosCrudos[label] = var

        # Etiqueta para datos normalizados
        self.lblDatosNormalizados = tk.Label(main_frame, text="Datos Normalizados", fg="black", font=("Helvetica", 12))
        self.lblDatosNormalizados.grid(row=1, column=1, sticky=tk.W)

        # Marco para mostrar los datos normalizados
        self.frmDatosNormalizados = ttk.Frame(main_frame)
        self.frmDatosNormalizados.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frmDatosNormalizados.columnconfigure(0, weight=1)
        self.frmDatosNormalizados.rowconfigure(0, weight=1)

        # Tabla para mostrar los datos normalizados
        self.tblDatosNormalizados = ttk.Treeview(self.frmDatosNormalizados)
        self.tblDatosNormalizados.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Barra de desplazamiento para la tabla de datos normalizados
        self.scrollDatosNormalizados = ttk.Scrollbar(self.frmDatosNormalizados, orient="vertical", command=self.tblDatosNormalizados.yview)
        self.scrollDatosNormalizados.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tblDatosNormalizados.configure(yscroll=self.scrollDatosNormalizados.set)

        # Marco para mostrar la información del atributo seleccionado (Normalizado)
        self.frmInfoDatosNormalizados = ttk.Labelframe(main_frame, text="Atributo seleccionado (Normalizado)", padding="10")
        self.frmInfoDatosNormalizados.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=10)
        self.frmInfoDatosNormalizados.columnconfigure(1, weight=1)

        # Etiquetas y campos de entrada para mostrar estadísticas del atributo seleccionado (Normalizado)
        lblsDatosNormalizados = ["Nombre del atributo", "Valor mínimo", "Valor máximo", "Media", "Desviación estándar"]
        self.entDatosNormalizados = {}
        for i, label in enumerate(lblsDatosNormalizados):
            tk.Label(self.frmInfoDatosNormalizados, text=label, fg="black", font=("Helvetica", 12)).grid(row=i, column=0, sticky=tk.W)
            var = tk.StringVar()
            entry = tk.Entry(self.frmInfoDatosNormalizados, textvariable=var, state='readonly', width=20)
            entry.grid(row=i, column=1, sticky=(tk.W, tk.E))
            self.entDatosNormalizados[label] = var

        # Botón para normalizar los datos
        self.btnNormalizar = tk.Button(main_frame, text="Normalizar datos...", command=self.normalizarDatos, fg="black")
        self.btnNormalizar.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=10)

        # Etiqueta y campo de entrada para el número de clusters (k)
        self.lblCluster = tk.Label(main_frame, text="Número de Clusters (k)", fg="black", font=("Helvetica", 12))
        self.lblCluster.grid(row=5, column=0, sticky=tk.W)
        self.entCluster = tk.Entry(main_frame)
        self.entCluster.grid(row=5, column=1, sticky=tk.W)

        # Radio Butto seleccionar la versión del algoritmo K-means
        self.lblAlgoritmo = tk.Label(main_frame, text="Seleccione la versión del algoritmo:", fg="black", font=("Helvetica", 12))
        self.lblAlgoritmo.grid(row=6, column=0, sticky=tk.W, pady=(10, 0))

        self.varAlgoritmo = tk.StringVar(value="vectorizada")
        self.rdioVectorizado = tk.Radiobutton(main_frame, text="Vectorizada", variable=self.varAlgoritmo, value="vectorizada")
        self.rdioVectorizado.grid(row=7, column=0, sticky=tk.W, padx=(20, 5), pady=(5, 0))

        self.rdioNoVectorizado = tk.Radiobutton(main_frame, text="No Vectorizada", variable=self.varAlgoritmo, value="no_vectorizada")
        self.rdioNoVectorizado.grid(row=7, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))

        # Botón para ejecutar K-means
        self.btnKmeans = tk.Button(main_frame, text="Ejecutar K-Means...", command=self.ejecutarKmeans, fg="black")
        self.btnKmeans.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Botón para salir del programa
        self.btnSalir = tk.Button(main_frame, text="Salir", command=self.confirmarSalida, fg="black")
        self.btnSalir.grid(row=9, column=1, sticky=tk.E, pady=10)

        # Variables de clase para almacenar el archivo seleccionado, datos crudos y datos normalizados
        self.rutaArchivo = ''
        self.dataCruda = None
        self.dataNormalizada = None
        
        # Variables para almacenar los Checkbuttons y las variables de los atributos seleccionados
        self.attrSeleccionados = []
        self.frmAtributos = None
        self.tblDatosCrudos.bind("<<TreeviewSelect>>", self.mostrarAtribCrudo)
        self.tblDatosNormalizados.bind("<<TreeviewSelect>>", self.mostrarAtribNormalizado)

    def confirmarSalida(self):
        respuesta = messagebox.askyesno("Confirmar salida", "¿Está seguro que desea salir?")
        if respuesta:
            self.root.quit()
            
    def set_icon(self, icon_path):
        try:
            self.root.iconphoto(False, tk.PhotoImage(file=icon_path))
        except Exception as e:
            print(f"Error setting icon: {e}")
    
    # Función para cargar el archivo ARFF
    def cargarArchivo(self):
        self.rutaArchivo = filedialog.askopenfilename(filetypes=[("ARFF files", "*.arff")])
        if self.rutaArchivo:
            self.nombreArchivo.set(self.rutaArchivo.split("/")[-1])
            data, _ = arff.loadarff(self.rutaArchivo)
            dataFrame = pd.DataFrame(data)
            if 'class' in dataFrame.columns:
                dataFrame = dataFrame.drop(columns=['class'])
            self.dataCruda = dataFrame
            self.mostrarData(self.dataCruda, self.tblDatosCrudos)
            self.mostrarSeleccionAttrs()

    # Función para mostrar datos en la tabla
    def mostrarData(self, df, table):
        table.delete(*table.get_children())
        table["columns"] = ["Número de Atributo", "Nombre del Atributo"]
        table["show"] = "headings"
        table.heading("Número de Atributo", text="Número de Atributo")
        table.heading("Nombre del Atributo", text="Nombre del Atributo")

        for i, column in enumerate(df.columns, start=1):
            table.insert("", "end", values=(i, column))

        table_height = min(20, len(df) + 1)
        table.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), rowspan=table_height)

        scrollbar = ttk.Scrollbar(table.master, orient="vertical", command=table.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S), rowspan=table_height)
        table.configure(yscroll=scrollbar.set)
        
    # Función para mostrar los Checkbuttons de selección de características
    def mostrarSeleccionAttrs(self):
        if self.frmAtributos:
            self.frmAtributos.destroy()

        self.frmAtributos = ttk.Frame(self.root)
        self.frmAtributos.grid(row=0, column=2, rowspan=9, padx=10, pady=10, sticky=(tk.N, tk.S, tk.W, tk.E))

        canvas = tk.Canvas(self.frmAtributos)
        scrollbar = ttk.Scrollbar(self.frmAtributos, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        title_label = ttk.Label(scrollable_frame, text="Atributos a utilizar para la ejecución del algoritmo")
        title_label.pack(anchor=tk.W, pady=(0, 10))

        self.attrSeleccionados = []
        for column in self.dataCruda.columns:
            var = tk.BooleanVar()
            checkbutton = ttk.Checkbutton(scrollable_frame, text=column, variable=var)
            checkbutton.pack(anchor=tk.W)
            self.attrSeleccionados.append((column, var))
        
    # Función para normalizar los datos
    def normalizarDatos(self):
        if self.dataCruda is not None:
            scaler = StandardScaler()
            self.dataNormalizada = scaler.fit_transform(self.dataCruda)
            normalized_df = pd.DataFrame(self.dataNormalizada, columns=self.dataCruda.columns)
            self.mostrarData(normalized_df, self.tblDatosNormalizados)
        else:
            mostrarMensaje(titulo="Error", mensaje="Cargue un archivo antes de normalizar los datos.")

    # Función para mostrar información del atributo seleccionado (Crudo)
    def mostrarAtribCrudo(self, event):
        atributoSeleccionado = self.tblDatosCrudos.selection()
        if atributoSeleccionado:
            indiceAtributo = int(self.tblDatosCrudos.item(atributoSeleccionado, "values")[0]) - 1
            nombreAtributo = self.dataCruda.columns[indiceAtributo]

            dataAtributo = self.dataCruda[nombreAtributo]

            self.entDatosCrudos["Nombre del atributo"].set(nombreAtributo)
            self.entDatosCrudos["Valor mínimo"].set(round(dataAtributo.min(), 5))
            self.entDatosCrudos["Valor máximo"].set(round(dataAtributo.max(), 5))
            self.entDatosCrudos["Media"].set(round(dataAtributo.mean(), 5))
            self.entDatosCrudos["Desviación estándar"].set(round(dataAtributo.std(), 5))

    # Función para mostrar información del atributo seleccionado (Normalizado)
    def mostrarAtribNormalizado(self, event):
        atributoSeleccionado = self.tblDatosNormalizados.selection()
        if atributoSeleccionado:
            indiceAtributo = int(self.tblDatosNormalizados.item(atributoSeleccionado, "values")[0]) - 1
            nombreAtributo = self.dataCruda.columns[indiceAtributo]

            dataAtributo = self.dataNormalizada[:, indiceAtributo]

            self.entDatosNormalizados["Nombre del atributo"].set(nombreAtributo)
            self.entDatosNormalizados["Valor mínimo"].set(round(dataAtributo.min(), 5))
            self.entDatosNormalizados["Valor máximo"].set(round(dataAtributo.max(), 5))
            self.entDatosNormalizados["Media"].set(round(dataAtributo.mean(), 5))
            self.entDatosNormalizados["Desviación estándar"].set(round(dataAtributo.std(), 5))

    # Función para ejecutar el algoritmo K-Means
    def ejecutarKmeans(self):
        if self.dataNormalizada is not None:
            atributosSeleccionados = [column for column, var in self.attrSeleccionados if var.get()]
            if len(atributosSeleccionados) < 2:
                mostrarMensaje(titulo="Error", mensaje="Seleccione al menos dos atributos para ejecutar K-means.")
                return

            indicesSeleccionados = [self.dataCruda.columns.get_loc(col) for col in atributosSeleccionados] # Columnas de los atributos
            dataClusterizar = self.dataNormalizada[:, indicesSeleccionados] # Todas las filas de las columnas indicadas

            try:
                entrCluster = self.entCluster.get()
                if not entrCluster:
                    raise ValueError("El número de clusters no puede estar vacío.")
                
                k = int(entrCluster)
                if k <= 0:
                    raise ValueError("El número de clusters debe ser mayor que 0.")

                algoritmo = self.varAlgoritmo.get()
                if algoritmo == "vectorizada":
                    centroides = self.inicializarCentroides(dataClusterizar, k)
                    print(f"Centroides iniciales:\n{centroides}")

                    maxIteraciones = 500
                    iteracion = 0
                    convergencia = False
                    centroidesAnteriores = None

                    while not convergencia and iteracion < maxIteraciones:
                        asignacionesCluster, centroides = self.iterarKmeans(dataClusterizar, centroides)
                        if centroidesAnteriores is not None and np.array_equal(centroides, centroidesAnteriores): # Si los centroides actuales son iguales a los anteriores, se alcanza la convergencia
                            convergencia = True
                        centroidesAnteriores = centroides
                        iteracion += 1
                        print(f"Iteración {iteracion}")

                    # Calculos para mostrar en consola
                    inercia = self.calcularInercia(dataClusterizar, asignacionesCluster, centroides)
                    print(f"Error de KMeans: {inercia}")

                    print("Centroides finales:")
                    for i, centroide in enumerate(centroides): # Valores de los centroides finales
                        print(f"Centroide {i+1}: {centroide}")

                    totalPuntos = dataClusterizar.shape[0] 
                    print("Asignaciones de puntos:") # Puntos asignados a cada cluster
                    for i in range(k):
                        puntosCluster = dataClusterizar[asignacionesCluster == i]
                        numPuntosCluster = len(puntosCluster)
                        percentage = (numPuntosCluster / totalPuntos) * 100
                        print(f"Cluster {i+1}: {numPuntosCluster} puntos ({percentage:.2f}%)")

                    if dataClusterizar.shape[1] == 2:
                        self.graficarClusters2D(dataClusterizar, asignacionesCluster, centroides)
                    elif dataClusterizar.shape[1] == 3:
                        self.graficarClusters3D(dataClusterizar, asignacionesCluster, centroides)
                    else: 
                        self.graficarClustersConPCA(dataClusterizar, asignacionesCluster, centroides)
                else:
                    centroides = self.inicializarCentroidesNoVect(dataClusterizar, k)
                    print(f"Centroides iniciales:\n{centroides}")

                    maxIteraciones = 500    
                    iteracion = 0
                    convergencia = False
                    centroidesAnteriores = None

                    while not convergencia and iteracion < maxIteraciones:
                        asignacionesCluster = self.asignarPuntosaClustersNoVect(dataClusterizar, centroides)
                        centroidesActualizados = self.actualizarCentroidesNoVect(dataClusterizar, asignacionesCluster, k)
                        if centroidesAnteriores is not None and np.array_equal(centroides, centroidesAnteriores):
                            convergencia = True
                        centroidesAnteriores = centroides
                        iteracion += 1
                        print(f"Iteración {iteracion}")

                    centroides = centroidesActualizados

                    self.evaluarClusters(dataClusterizar, asignacionesCluster, centroides)

                    print("Centroides finales:")
                    print(centroides)
                    print("Asignaciones de puntos:")

                    totalPuntos = dataClusterizar.shape[0]
                    for i in range(k):
                        puntosCluster = dataClusterizar[asignacionesCluster == i]
                        numPuntosCluster = len(puntosCluster)
                        percentage = (numPuntosCluster / totalPuntos) * 100
                        print(f"Cluster {i+1}: {numPuntosCluster} puntos ({percentage:.2f}%)")

                    if dataClusterizar.shape[1] == 2:
                        self.graficarClusters2D(dataClusterizar, asignacionesCluster, centroides)
                    elif dataClusterizar.shape[1] == 3:
                        self.graficarClusters3D(dataClusterizar, asignacionesCluster, centroides)
                    else:
                        self.graficarClustersConPCA(dataClusterizar, asignacionesCluster, centroides)
            except ValueError as e:
                mostrarMensaje(titulo="Error", mensaje=f"{e}")
        else:
            mostrarMensaje(titulo="Error", mensaje="Primero debe cargar y normalizar los datos.")
    
    # --------------- VERSION VECTORIZADA ---------------
    # Función para inicializar los centroides aleatoriamente
    def inicializarCentroides(self, data, k):
        indices = np.random.choice(data.shape[0], k, replace=False)
        centroides = data[indices, :]
        return centroides
        
    # Función para asignar puntos a clusters
    def asignarPuntosaClusters(self, data, centroides):
        k = centroides.shape[0]
        asignacionesCluster = np.zeros(data.shape[0]) # Arreglo para manejar las asignaciones de clusters para cada punto
        for i, punto in enumerate(data):
            distancias = np.linalg.norm(centroides - punto, axis=1) # Calcula la distancia euclideana desde el punto a todos los centroides
            asignacionesCluster[i] = np.argmin(distancias) # Asigna el punto al centroide con la distancia minima
        return asignacionesCluster
    
    # Función para actualizar los centroides
    def actualizarCentroides(self, data, asignacionesCluster): 
        k = len(np.unique(asignacionesCluster))
        centroidesActualizados = np.zeros((k, data.shape[1]))
        for i in range(k):
            puntosCluster = data[asignacionesCluster == i] # Puntos asignados al cluster
            if len(puntosCluster) > 0:
                centroidesActualizados[i] = np.mean(puntosCluster, axis=0) # Calcula el nuevo centroide como la media de todos los puntos del cluster
        return centroidesActualizados
    
    def evaluarClusters(self, data, asignacionesCluster, centroides):
        inercia = self.calcularInercia(data, asignacionesCluster, centroides)
        print(f"Inercia: {inercia}")

    def calcularInercia(self, data, asignacionesCluster, centroides):
        inercia = 0
        for cluster_idx, centroide in enumerate(centroides):
            puntosCluster = data[asignacionesCluster == cluster_idx] 
            distancias = np.linalg.norm(puntosCluster - centroide, axis=1) # Calculo de distancias al centroide
            inercia += np.sum(distancias ** 2) # Suma de las distancias cuadradas 
        return inercia
    
    # Función para realizar una iteración de K-means
    def iterarKmeans(self, data, centroides):
        asignacionesCluster = np.zeros(data.shape[0])
        for i, punto in enumerate(data):
            distancias = np.linalg.norm(punto - centroides, axis=1) 
            clusterCercano = np.argmin(distancias) 
            asignacionesCluster[i] = clusterCercano # Asigna el punto al cluster mas cercano
        
        nuevosCentroides = np.zeros_like(centroides)
        for cluster in range(len(centroides)):
            puntosCluster = data[asignacionesCluster == cluster]
            if len(puntosCluster) > 0:
                nuevosCentroides[cluster] = np.mean(puntosCluster, axis=0) # Nuevo centroide como la media de los puntos
        
        return asignacionesCluster, nuevosCentroides

    # -------------- FIN VERSION VECTORIZADA ---------------

    # --------------- VERSION NO VECTORIZADA ---------------
    
    # Función para inicializar los centroides aleatoriamente    
    def inicializarCentroidesNoVect(self, data, k):
        indices = np.random.choice(data.shape[0], k, replace=False)
        centroides = data[indices, :]
        return centroides

    # Función para asignar puntos a clusters
    def asignarPuntosaClustersNoVect(self, data, centroides):
        k = centroides.shape[0]
        asignacionesCluster = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            punto = data[i]
            distancias = np.zeros(k)
            for j in range(k):
                distancias[j] = np.linalg.norm(punto - centroides[j])
            asignacionesCluster[i] = np.argmin(distancias)
        return asignacionesCluster

    def actualizarCentroidesNoVect(self, data, asignacionesCluster, k):
        centroidesActualizados = np.zeros((k, data.shape[1]))
        for i in range(k):
            puntosCluster = []
            for j in range(data.shape[0]):
                if asignacionesCluster[j] == i:
                    puntosCluster.append(data[j])
            if len(puntosCluster) > 0:
                centroidesActualizados[i] = np.mean(puntosCluster, axis=0)
        return centroidesActualizados
    
    # --------------- FIN VERSION NO VECTORIZADA ---------------
    
    # -------------- FUNCIONES PARA REALIZAR LOS GRAFICOS --------------
    
    # Función para graficar los clusters y centroides en 2D (cuando usamos 2 atributos)
    def graficarClusters2D(self, data, asignacionesCluster, centroides):
        plt.figure(figsize=(8, 6))
        for i in range(len(centroides)):
            puntosCluster = data[asignacionesCluster == i]
            plt.scatter(puntosCluster[:, 0], puntosCluster[:, 1], label=f'Cluster {i}')
        plt.scatter(centroides[:, 0], centroides[:, 1], marker='x', color='black', label='Centroides')
        plt.title('Clusters y Centroides (2D)')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.legend()
        plt.show()

    # Función para graficar los clusters y centroides en 3D (cuando usamos 3 atributos)
    def graficarClusters3D(self, data, asignacionesCluster, centroides):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(centroides)):
            puntosCluster = data[asignacionesCluster == i]
            ax.scatter(puntosCluster[:, 0], puntosCluster[:, 1], puntosCluster[:, 2], label=f'Cluster {i}')
        ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], marker='x', color='black', s=100, label='Centroides')
        ax.set_title('Clusters y Centroides (3D)')
        ax.set_xlabel('Dimensión 1')
        ax.set_ylabel('Dimensión 2')
        ax.set_zlabel('Dimensión 3')
        ax.legend()
        plt.show()
    
    # Función para graficar los clusters y centroides en 2D (cuando usamos 4 o más atributos)
    def graficarClustersConPCA(self, data, asignacionesCluster, centroides):
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroides2D = pca.transform(centroides)

        plt.figure(figsize=(8, 6))
        for i in range(len(centroides)):
            puntosCluster = data_2d[asignacionesCluster == i]
            plt.scatter(puntosCluster[:, 0], puntosCluster[:, 1], label=f'Cluster {i}')
        plt.scatter(centroides2D[:, 0], centroides2D[:, 1], marker='x', color='black', label='Centroides')
        plt.title('Clusters y Centroides con PCA (2D)')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend()
        plt.show()
        
    # ------------- FIN FUNCIONES PARA REALIZAR LOS GRAFICOS ------------

def mostrarMensaje(titulo, mensaje):
    root = tk.Tk()
    root.withdraw()

    if titulo.lower() == "advertencia":
        messagebox.showwarning(titulo, mensaje)
    elif titulo.lower() == "error":
        messagebox.showerror(titulo, mensaje)
    else:
        messagebox.showinfo(titulo, mensaje)

    root.destroy()

class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - 500) // 2
        y = (screen_height - 300) // 2

        self.root.geometry("500x300+{}+{}".format(x, y))

        original_image = Image.open("UBP.png")
        resized_image = original_image.resize((200, 200))
        self.photo = ImageTk.PhotoImage(resized_image)
        self.label = tk.Label(root, image=self.photo)
        self.label.pack()

        self.description = tk.Label(root, text="Trabajo Final - SIA", font=("Helvetica", 16))
        self.description.pack(pady=20)

    def close(self):
        self.root.destroy()

# Funcion main
if __name__ == "__main__":
    root = tk.Tk()
    splash = SplashScreen(root)

    root.after(3000, root.destroy)  
    root.mainloop()
    
    main_root = tk.Tk()
    app = KMeansApp(main_root)
    main_root.mainloop()