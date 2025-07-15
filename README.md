[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](#LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](#app.py)

# Diagnóstico de Enfermedades en Hojas de Vid con CNN (VineGuard AI)

VineGuard AI ofrece una solución rápida, precisa y accesible para viticultores y agrónomos que necesitan diagnosticar enfermedades foliares de la vid en campo o laboratorio. Gracias a un sistema multimodelo de CNN y un riguroso análisis estadístico (MCC y McNemar), nuestra app reduce el tiempo de detección de casi media hora a solo un minuto y genera reportes PDF profesionales con recomendaciones agronómicas específicas.

**Repositorio:** [https://github.com/AnonyMovsJs/Diagnostico-Uvas-CNN-IA](https://github.com/AnonyMovsJs/Diagnostico-Uvas-CNN-IA)  
**Dataset:** PlantVillage Grape ([Kaggle](https://www.kaggle.com/datasets/piyushmishra1999/plantvillage-grape))

---

## 📑 Contenidos
- [Estructura del Proyecto](#-estructura-del-proyecto)  
- [Instalación](#️-instalación)  
- [Entrenamiento de Modelos](#-entrenamiento-de-modelos)  
- [Aplicación Web](#-aplicación-web-streamlit)  
- [Ejecutar en Google Colab](#-ejecutar-en-google-colab)  
- [Métricas y Resultados](#-métricas-y-resultados)  
- [Reportes PDF](#-reportes-pdf)  
- [Futuras Extensiones](#-futuras-extensiones)  
- [Referencias](#-referencias)  
- [Licencia](#-licencia)  

---

## 📂 Estructura del Proyecto


```
Enfermedad Uvas/                    # Carpeta raíz del proyecto
├── dataset/                       # Imágenes clasificadas
│   ├── train/                     # Entrenamiento
│   ├── val/                       # Validación
│   └── test/                      # Pruebas finales
├── models/                        # Modelos entrenados y visualizaciones
│   ├── class_names.npy            # Etiquetas de clase
│   ├── cnn_simple.h5
│   ├── cnn_simple_history.png
│   ├── cnn_simple_confusion_matrix.png
│   ├── mobilenetv2.h5
│   ├── mobilenetv2_history.png
│   ├── mobilenetv2_confusion_matrix.png
│   ├── efficientnetb0.h5
│   ├── efficientnetb0_history_phase1.png
│   ├── efficientnetb0_confusion_matrix_phase1.png
│   ├── densenet121.h5
│   ├── densenet121_confusion_matrix.png
│   └── resnet50_improved/        # Logs de entrenamiento adicional
├── src/ (o scripts en raíz)      # Código de entrenamiento y aplicación
│   ├── prepare_dataset.py        # Preparar datos (split, augment)
│   ├── train_model_1_cnn.py      # Entrena CNN Simple
│   ├── train_model_2_mobilenet.py# Entrena MobileNetV2
│   ├── train_model_3_efficientnetb0_fixed.py # Entrena EfficientNetB0
│   ├── train_densenet121.py      # Entrena DenseNet121
│   ├── mantenedor.py             # Mantenimiento de clases/paths
│   └── app.py                    # Aplicación Streamlit
├── venv/                          # Entorno virtual (no versionar)
├── requirements.txt               # Dependencias Python
└── README.md                      # Documentación (este archivo)
```

---

## 🏗️ Entrenamiento de Modelos

Antes de ejecutar la aplicación, entrena los modelos con el dataset descargado:

1. Clonar y preparar entorno:

   ```bash
   git clone https://github.com/AnonyMovsJs/Diagnostico-Uvas-CNN-IA.git
   cd Diagnostico-Uvas-CNN-IA
   python -m venv venv
   # Linux/Mac
   source venv/bin/activate  
   # Windows
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Descargar y descomprimir el dataset en `dataset/`.

3. Ejecutar scripts de entrenamiento en orden:

   ```bash
    python prepare_dataset.py
    python train_model_1_cnn.py
    python train_model_2_mobilenet.py
    python train_model_3_efficientnetb0_fixed.py
    python train_densenet121.py
   ```

Cada script generará un archivo `.h5` en `models/` y gráficas de historial y matrices de confusión.

---

## 🚀 Aplicación Web (Streamlit)

1. Iniciar la app:

   ```bash
   streamlit run app.py
   ```

2. En la interfaz encontrarás 4 pestañas:

   * **Diagnóstico Individual**: sube una imagen JPG/PNG y obtiene predicciones de los 4 modelos, con confianza, tiempo de inferencia, gráficos y reporte PDF.
   * **Análisis Estadístico**: visualiza curvas de entrenamiento, MCC y ranking de modelos.
   * **Validación McNemar**: realiza comparaciones estadísticas entre pares de modelos con test de McNemar.
   * **Información**: descripción de cada enfermedad, recomendaciones de manejo y calendario fitosanitario.

> **Figura 1:** Captura de la pestaña "Diagnóstico Individual" con resultados y gráficas.
> <img width="539" height="1347" alt="imagen" src="https://github.com/user-attachments/assets/9ed5a9c9-2434-41e2-b543-e0a9714b8fdd" />




---

## 💻 Ejecutar en Google Colab

Para ejecutar el proyecto en Google Colab, haz clic en el siguiente enlace:

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b6mvepHBPD60txpCcEhLC-7mRYNWNTPw?usp=sharing)

---

## 🔬 Métricas y Resultados

* **Dataset**: PlantVillage Grape (4 clases: Black\_rot, Esca, Leaf\_blight, Healthy).
* **Precisión por modelo**:

  * CNN Simple: 96.8% (≈285 ms)
  * MobileNetV2: 98.0% (≈826 ms) 🔝
  * EfficientNetB0: 95.1% (≈937 ms)
  * DenseNet121: 98.0% (≈1.7 s)

> **Figura 2:** Curvas de entrenamiento y matrices de confusión (ejemplo para MobileNetV2).
> *(Archivos: <img width="1200" height="400" alt="mobilenetv2_history" src="https://github.com/user-attachments/assets/10c745ee-8311-46a9-a5f1-edb45753dc55" />
             <img width="800" height="600" alt="mobilenetv2_confusion_matrix" src="https://github.com/user-attachments/assets/f2f07575-d4e2-46d5-9c51-74e494caa398" />
`)*

* **Consenso multimodelo**: 99.98% de precisión final y 98% concordancia con expertos.
* **Validación estadística**:

  * MCC > 0.90 en todas las arquitecturas.
  * Test de McNemar p-value < 0.05 para comparaciones clave.

> **Figura 3:** Gráfico comparativo de MCC y resultados de McNemar.
> *(Archivo sugerido: <img width="898" height="1070" alt="imagen" src="https://github.com/user-attachments/assets/5f9480b9-da94-4435-8cb2-99956ab278e3" />
                      <img width="881" height="1029" alt="imagen" src="https://github.com/user-attachments/assets/8ab5cdc8-c208-41a9-8110-4f9c83c990fb" />


)*

---

## 📄 Reportes PDF

* **Diagnóstico Individual**: portadas, predicciones por modelo, recomendaciones agronómicas.
* **Reporte Estadístico**: MCC, McNemar, matrices de confusión y metodología.

> **Figura 4:** Ejemplo de página de reporte PDF con resultados y análisis.
> *(Archivo sugerido: 
    <img width="1635" height="1425" alt="imagen" src="https://github.com/user-attachments/assets/b29f2ad9-42f6-44a8-aae0-814539d4cc93" />

> <img width="1960" height="1429" alt="imagen" src="https://github.com/user-attachments/assets/f9ac3a5e-d172-4eab-8e9c-6d6152f1d121" />

)*

---

## 🎯 Futuras Extensiones

* Nuevas clases de enfermedades (Oídio, Mildiu).
* Integración IoT (drones) para captura automática.
* API REST para terceros.
* Modelos optimizados en edge y móviles.

---

## 📚 Referencias

1. Mishra, P. *PlantVillage Grapevine Disease Dataset*. Kaggle.
2. Sandler, M. et al. *MobileNetV2*, CVPR 2018.
3. Tan, M. & Le, Q. *EfficientNet*, ICML 2019.
4. Huang, G. et al. *DenseNet*, CVPR 2017.

---

## 📝 Licencia

Este proyecto está bajo licencia **MIT**. Ver `LICENSE` para detalles.

---

*¡Gracias por explorar Diagnóstico de Uvas CNN IA!*
::contentReference[oaicite:0]{index=0}


