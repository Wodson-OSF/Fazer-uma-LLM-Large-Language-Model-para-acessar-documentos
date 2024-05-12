# Fazer-uma-LLM-Large-Language-Model-para-acessar-documentos
# Importações e configurações iniciais
!pip install -u -q google-generativeai

import numpy as np
import pandas as pd
import google.generativeai as genai

# Definindo a chave da API do Google Generative AI
GOOGLE_API_KEY = "PASTE_YOUR_KEY_HERE"  # Substitua por sua chave de API válida
genai.configure(api_key=GOOGLE_API_KEY)

# Listando modelos disponíveis com o método `generateContent`
print("Modelos disponíveis com o método `generateContent`:")
for model in genai.list_models():
  if 'generateContent' in model.supported_generation_methods:
    print(model.name)

# Exemplo de embedding de texto
title = "A próxima geração de IA para desenvolvedores e Google Workspace"
sample_text = ("Título: A próxima geração de IA para desenvolvedores e Google Workspace"
            "\n"
            "Artigo completo:\n"
            "\n"
            "Gemini API & Google AI Studio: Uma maneira acessível de explorar e criar protótipos com aplicações de IA generativa")

# Criando um embedding para o texto de exemplo
embeddings = genai.embed_content(model="models/embedding-001",
                                 content=sample_text,
                                 title=title,
                                 task_type="RETRIEVAL_DOCUMENT")

print("Embedding do texto de exemplo:", embeddings)

# Definindo os documentos a serem buscados
DOCUMENT1 = {
    "Título": "Operação do sistema de controle climático",
    "Conteúdo": "O Googlecar tem um sistema de controle climático que permite ajustar a temperatura e o fluxo de ar no carro. Para operar o sistema de controle climático, use os botões e botões localizados no console central.  Temperatura: O botão de temperatura controla a temperatura dentro do carro. Gire o botão no sentido horário para aumentar a temperatura ou no sentido anti-horário para diminuir a temperatura. Fluxo de ar: O botão de fluxo de ar controla a quantidade de fluxo de ar dentro do carro. Gire o botão no sentido horário para aumentar o fluxo de ar ou no sentido anti-horário para diminuir o fluxo de ar. Velocidade do ventilador: O botão de velocidade do ventilador controla a velocidade do ventilador. Gire o botão no sentido horário para aumentar a velocidade do ventilador ou no sentido anti-horário para diminuir a velocidade do ventilador. Modo: O botão de modo permite que você selecione o modo desejado. Os modos disponíveis são: Auto: O carro ajustará automaticamente a temperatura e o fluxo de ar para manter um nível confortável. Cool (Frio): O carro soprará ar frio para dentro do carro. Heat: O carro soprará ar quente para dentro do carro. Defrost (Descongelamento): O carro soprará ar quente no para-brisa para descongelá-lo."}

DOCUMENT2 = {
    "Título": "Touchscreen",
    "Conteúdo": "O seu Googlecar tem uma grande tela sensível ao toque que fornece acesso a uma variedade de recursos, incluindo navegação, entretenimento e controle climático. Para usar a tela sensível ao toque, basta tocar no ícone desejado.  Por exemplo, você pode tocar no ícone \"Navigation\" (Navegação) para obter direções para o seu destino ou tocar no ícone \"Music\" (Música) para reproduzir suas músicas favoritas."}

DOCUMENT3 = {
    "Título": "Mudança de marchas",
    "Conteúdo": "Seu Googlecar tem uma transmissão automática. Para trocar as marchas, basta mover a alavanca de câmbio para a posição desejada.  Park (Estacionar): Essa posição é usada quando você está estacionado. As rodas são travadas e o carro não pode se mover. Marcha à ré: Essa posição é usada para dar ré. Neutro: Essa posição é usada quando você está parado em um semáforo ou no trânsito. O carro não está em marcha e não se moverá a menos que você pressione o pedal do acelerador. Drive (Dirigir): Essa posição é usada para dirigir para frente. Low: essa posição é usada para dirigir na neve ou em outras condições escorregadias."}

# Criando um DataFrame com os documentos
documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
df = pd.DataFrame(documents)

# Renomeando as colunas do DataFrame
df.columns = ["Título", "Conteudo"]

# Visualizando o DataFrame
print(df)

# Definindo o modelo de embedding
model = "models/embedding-001"
def embed_fn(title, text):
  """
  Esta função gera um embedding (representação vetorial) para um determinado texto.

  Args:
      title (str): Título do texto.
      text (str): Conteúdo do texto.

  Returns:
      numpy.ndarray: Embedding do texto (representação vetorial).
  """
  return genai.embed_content(model=model,
                                 content=text,
                                 title=title,
                                 task_type="RETRIEVAL_DOCUMENT")["embedding"]

# Cria uma nova coluna 'Embeddings' no dataframe 'df'
df["Embeddings"] = df.apply(lambda row: embed_fn(row["Titulo"], row["Conteudo"]), axis=1)

# Exibe o dataframe 'df' (opcional)
# print(df)
# Função para criar embeddings de documentos
def embed_fn(title, text):
# Função para gerar e buscar consulta
def gerar_e_buscar_consulta(consulta, base, model):
  """
  Esta função recebe uma consulta, um dataframe contendo embeddings de documentos e um modelo de geração de texto.
  Ela gera um embedding para a consulta utilizando o modelo e então calcula a similaridade entre o embedding da consulta e os embeddings dos documentos no dataframe.
  Por fim, retorna o conteúdo do documento mais similar à consulta.
  #trade-off(troca):maior segurança na busca da informação, já que a busca é feita dentro do sistema de biblioteca que você montou. RAG - combinar IA com a base de conhecimento que você e/ou alimentou na sua base. 

  Argumentos:
    consulta: A string contendo a consulta a ser realizada.
    base: Um dataframe contendo embeddings de documentos.
    model: Um modelo de geração de texto do Google Generative AI.

  Retorno:
    A string contendo o conteúdo do documento mais similar à consulta.
  """

  # Gera o embedding da consulta utilizando o modelo
  embedding_da_consulta = genai.embed_content(model=model,
                                              content=consulta,
                                              task_type="RETRIEVAL_QUERY")["embedding"]

  # Calcula a similaridade entre o embedding da consulta e os embeddings dos documentos no dataframe
  produtos_escalares = np.dot(np.stack(base["Embeddings"]), embedding_da_consulta)

  # Encontra o índice do documento mais similar à consulta
  indice = np.argmax(produtos_escalares)

  # Retorna o conteúdo do documento mais similar à consulta
  return base.iloc[indice]["Conteudo"]

# Consulta a ser realizada
consulta = "Como faço para trocar marchas em um carro do Google?"

# Gera o texto mais similar à consulta utilizando o modelo
trecho = gerar_e_buscar_consulta(consulta, df, model)

# Imprime o texto mais similar à consulta
print(trecho)

# Configurações de geração de texto
generation_config = {
#temperatura pode ser mudada e demanda a liguagem das resposta.
  "temperature": 0, 
  "candidate_count": 1
}

# Prompt para gerar texto a partir do trecho encontrado
prompt = f"Reescreva esse texto de uma forma mais descontraída, sem adicionar informações que não façam parte do texto: {trecho}"

# Carrega o modelo de geração de texto
model_2 = genai.GenerativeModel("gemini-1.0-pro",
                                generation_config=generation_config)

# Gera o texto a partir do prompt
response = model_2.generate_content(prompt)

# Imprime o texto gerado
print(response.text)


