# Software de Manutenção Preditiva para Pulverizadores 🌱⚙️

Este projeto é um **software de manutenção preditiva** para pulverizadores agrícolas, utilizando **Machine Learning (Random Forest)** a partir de um **banco de dados sintético**.

A aplicação foi desenvolvida em **Python** e executa uma **interface web com Streamlit**.

---
### Clonar o repositório
```
git clone https://github.com/Igor-mariano213/SOFTWARE-DE-MANUTEN-O-PREDITIVA-PARA-PULVERIZADORES.git
cd SOFTWARE-DE-MANUTEN-O-PREDITIVA-PARA-PULVERIZADORES
```

## 🚀 Como executar o projeto

### 1️⃣ Instalar as bibliotecas
Antes de tudo, instale as dependências do projeto com o comando:

```
pip install -r requirements.txt
```

2️⃣ Rodar este arquivo primeiro (geração do modelo)

Antes de executar a aplicação, é obrigatório rodar primeiro o arquivo responsável por gerar o modelo de Machine Learning:

```
python gerador_modelos.py
```

3️⃣ Executar a aplicação

Após rodar o arquivo acima, execute a aplicação com o Streamlit:

```
streamlit run app.py
```
