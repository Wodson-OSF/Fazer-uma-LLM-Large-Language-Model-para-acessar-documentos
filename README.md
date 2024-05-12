# Fazer-uma-LLM-Large-Language-Model-para-acessar-documentos
# Importações e configurações iniciais
!pip install -u -q google-generativeai

import numpy as np
import pandas as pd
import google.generativeai as genai

# Substitua "PASTE YOUR KEY HERE" pela sua chave API do Google Generative AI
GOOGLE_API_KEY = "PASTE YOUR KEY HERE" #CTRL C CTRL V DA CHAVE#
genai.configure(api_key=GOOGLE_API_KEY)

# Função para listar modelos que suportam "embedContent"
def listar_modelos_embed_content():
  for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
      print(m.name)

# Exemplo de embedding de conteúdo
titulo = "A próxima geração de IA para desenvolvedores e Google Workspace"
texto_de_exemplo = ("Título: A próxima geração de IA para desenvolvedores e Google Workspace"
                    "\n"
                    "Artigo completo:\n"
                    "\n"
                    "Gemini API & Google AI Studio: Uma maneira acessível de explorar e criar protótipos com aplicações de IA generativa")

embeddings = genai.embed_content(model="models/embedding-001",
                                 content=texto_de_exemplo,
                                 title=titulo,
                                 task_type="RETRIEVAL_DOCUMENT")

print(embeddings)

# Definição dos documentos a serem pesquisados
DOCUMENT1 = {
    "Título": "Operação do sistema de controle climático",
    "Conteúdo": "O Googlecar tem um sistema de controle climático que permite ajustar a temperatura e o fluxo de ar no carro. Para operar o sistema de controle climático, use os botões e botões localizados no console central.  Temperatura: O botão de temperatura controla a temperatura dentro do carro. Gire o botão no sentido horário para aumentar a temperatura ou no sentido anti-horário para diminuir a temperatura. Fluxo de ar: O botão de fluxo de ar controla a quantidade de fluxo de ar dentro do carro. Gire o botão no sentido horário para aumentar o fluxo de ar ou no sentido anti-horário para diminuir o fluxo de ar. Velocidade do ventilador: O botão de velocidade do ventilador controla a velocidade do ventilador. Gire o botão no sentido horário para aumentar a velocidade do ventilador ou no sentido anti-horário para diminuir a velocidade do ventilador. Modo: O botão de modo permite que você selecione o modo desejado. Os modos disponíveis são: Auto: O carro ajustará automaticamente a temperatura e o fluxo de ar para manter um nível confortável. Cool (Frio): O carro soprará ar frio para dentro do carro. Heat: O carro soprará ar quente para dentro do carro. Defrost (Descongelamento): O carro soprará ar quente no para-brisa para descongelá-lo."}

DOCUMENT2 = {
    "Título": "Touchscreen",
    "Conteúdo": "O seu Googlecar tem uma grande tela sensível ao toque que fornece acesso a uma variedade de recursos, incluindo navegação, entretenimento e controle climático. Para usar a tela sensível ao toque, basta tocar no ícone desejado.  Por exemplo, você pode tocar no ícone \"Navigation\" (Navegação) para obter direções para o seu destino ou tocar no ícone \"Music\" (Música) para reproduzir suas músicas favoritas."}

DOCUMENT3 = {
    "Título": "Mudança de marchas",
    "Conteúdo": "Seu Googlecar tem uma transmissão automática. Para trocar as marchas, basta mover a alavanca de câmbio para a posição desejada.  Park (Estacionar): Essa posição é usada quando você está estacionado. As rodas são travadas e o carro não pode se mover. Marcha à ré: Essa posição é usada para dar ré. Neutro: Essa posição é usada quando você está parado em um semáforo ou no trânsito. O carro não está em marcha e não se moverá a menos que você pressione o pedal do acelerador. Drive (Dirigir): Essa posição é usada para dirigir para frente. Low: essa posição é usada para dirigir na neve ou em outras condições escorregadias."}

documentos = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
df = pd.DataFrame(documentos)
df.columns = ["Título", "Conteudo"]

# Função para gerar embeddings de documentos
def gerar_embeddings_documentos(df, model):
  df["Embeddings"] = df.apply(lambda row: genai.embed_content(model=model,
                                                              content=row["Conteudo"],
                                                              title=row["Titulo"],
                                                              task_type="RETRIEVAL_DOCUMENT")["embedding"], axis
