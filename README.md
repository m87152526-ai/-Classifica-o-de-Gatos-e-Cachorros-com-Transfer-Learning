
# Classificação de Gatos e Cachorros com Transfer Learning.

Projeto desenvolvido por **Márcio Roberto B. de Sá** no ambiente **Google Colab**, aplicando **Transfer Learning** com o modelo **MobileNetV2** para classificar imagens de **gatos e cachorros**.

---

# Descrição do Projeto

Este projeto utiliza **Deep Learning** e **Transfer Learning** em uma base de dados contendo imagens de gatos e cachorros.

O objetivo é demonstrar o poder da reutilização de redes neurais pré-treinadas, reduzindo o tempo de treinamento e melhorando a precisão em um conjunto de dados pequeno.

O modelo escolhido foi o **MobileNetV2**, uma rede leve e eficiente originalmente treinada no ImageNet.

---

#Tecnologias Utilizadas

- **Python 3**
- **TensorFlow / Keras**
- **MobileNetV2 (Transfer Learning)**
- **Google Colab**
- **Matplotlib / NumPy**

---

# Estrutura do Projeto
transfer-learning-gatos-cachorros/
│
├── README.md
├── notebook/
│ └── transfer_learning_gatos_cachorros.ipynb
├── images/
│ ├── exemplo_gato.png
│ ├── exemplo_cachorro.png
│ └── resultados.png
└── dataset/
├── gatos/
│ ├── gato1.jpg
│ └── ...
└── cachorros/
├── cachorro1.jpg
└── ...

---

# Como Executar o Projeto

1. Abra o [Google Colab](https://colab.research.google.com)
2. Faça upload do arquivo do notebook localizado em:
3. notebook/transfer_learning_gatos_cachorros.ipynb
4. Faça upload do dataset no mesmo diretório do Colab:
5. dataset/
├── gatos/
└── cachorros/
4. Execute todas as células do notebook.
5. Ao final, o modelo será treinado e salvo como:
model_gatos_cachorros.h5

---

# Modelo Utilizado

O projeto utiliza o modelo **MobileNetV2** com pesos pré-treinados no **ImageNet**.  
A camada base é congelada (não treinável), e camadas densas adicionais são adicionadas para a classificação binária (gato vs. cachorro).

```python
from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
