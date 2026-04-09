import os
import re
import pandas as pd
from openai import OpenAI
from google import genai
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score

# =========================================================
# 1. Configuração das APIs
# =========================================================
load_dotenv()

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client_pplx = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =========================================================
# 2. Função para extrair apenas a letra da resposta
# =========================================================
def extrair_letra(resposta):
    if not resposta:
        return "ERRO"

    resposta = str(resposta).strip().upper()

    # Procura letra isolada: A, B, C, D...
    match = re.search(r"\b([A-Z])\b", resposta)
    if match:
        return match.group(1)

    # Se não encontrar, pega a primeira letra maiúscula encontrada
    match = re.search(r"[A-Z]", resposta)
    return match.group(0) if match else "ERRO"

# =========================================================
# 3. Prompt padrão
# =========================================================
def montar_prompt(pergunta, opcoes):
    return f"""
Responda à questão de múltipla escolha abaixo.

Retorne APENAS a letra da alternativa correta.
Não explique.
Não escreva texto adicional.
Não repita a pergunta.

Pergunta:
{pergunta}

Opções:
{opcoes}
"""

# =========================================================
# 4. Funções de consulta aos modelos
# =========================================================
def get_openai_response(pergunta, opcoes):
    prompt = montar_prompt(pergunta, opcoes)

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    texto = response.choices[0].message.content
    return extrair_letra(texto)

def get_gemini_response(pergunta, opcoes):
    prompt = montar_prompt(pergunta, opcoes)

    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    texto = response.text if response.text else ""
    return extrair_letra(texto)

def get_perplexity_response(pergunta, opcoes):
    prompt = montar_prompt(pergunta, opcoes)

    response = client_pplx.chat.completions.create(
        model="sonar-pro",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    texto = response.choices[0].message.content
    return extrair_letra(texto)

# =========================================================
# 5. Carregamento do CSV
# =========================================================
caminho_arquivo = "/home/marcelo-west/Atividade1/dataset_usmle_questoes_de_217_a_243_Marcelo_West.csv"

df = pd.read_csv(
    caminho_arquivo,
    sep=";",
    encoding="utf-8-sig"
)

# Ajuste o nome da coluna exatamente como está no arquivo
df = df[["Identificador", "Pergunta", "Opcoesrespostas", "Resposta Correta"]].copy()

# Renomear para o nome desejado no resultado
df = df.rename(columns={"Resposta Correta": "resposta_correta"})

# Padronizar a resposta correta
df["resposta_correta"] = df["resposta_correta"].apply(extrair_letra)

# =========================================================
# 6. Inferência com múltiplos modelos
# =========================================================
def inferir_perguntas_multiplos_modelos(dataframe, limite=None):
    resultados = []

    if limite is not None:
        dataframe = dataframe.head(limite)

    for contador, (_, linha) in enumerate(dataframe.iterrows(), start=1):
        identificador = linha["Identificador"]
        pergunta = linha["Pergunta"]
        opcoes = linha["Opcoesrespostas"]
        resposta_correta = linha["resposta_correta"]

        print(f"{contador} - Processando Identificador: {identificador}")

        try:
            resposta_gpt = get_openai_response(pergunta, opcoes)
        except Exception as e:
            print(f"Erro GPT no Identificador {identificador}: {e}")
            resposta_gpt = "ERRO"

        try:
            resposta_gemini = get_gemini_response(pergunta, opcoes)
        except Exception as e:
            print(f"Erro Gemini no Identificador {identificador}: {e}")
            resposta_gemini = "ERRO"

        try:
            resposta_perplexity = get_perplexity_response(pergunta, opcoes)
        except Exception as e:
            print(f"Erro Perplexity no Identificador {identificador}: {e}")
            resposta_perplexity = "ERRO"

        resultados.append({
            "Identificador": identificador,
            "Pergunta": pergunta,
            "resposta_correta": resposta_correta,
            "resposta_GPT": resposta_gpt,
            "resposta_Gemini": resposta_gemini,
            "resposta_Perplexity": resposta_perplexity
        })

    return pd.DataFrame(resultados)

# =========================================================
# 7. Função para calcular métricas
# =========================================================
def calcular_metricas(df_resultados):
    metricas = []

    y_true = df_resultados["resposta_correta"]

    modelos = {
        "GPT": "resposta_GPT",
        "Gemini": "resposta_Gemini",
        "Perplexity": "resposta_Perplexity"
    }

    for nome_modelo, coluna_pred in modelos.items():
        y_pred = df_resultados[coluna_pred]

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        metricas.append({
            "modelo": nome_modelo,
            "accuracy": accuracy,
            "f1_macro": f1
        })

    return pd.DataFrame(metricas)

# =========================================================
# 8. Executar inferência
# =========================================================
df_resultados = inferir_perguntas_multiplos_modelos(df, limite=None)

# Se quiser testar primeiro:
# df_resultados = inferir_perguntas_multiplos_modelos(df, limite=3)

# =========================================================
# 9. Salvar resultados
# =========================================================
arquivo_saida_resultados = "/home/marcelo-west/Atividade1/resultados_modelos.csv"

df_resultados.to_csv(
    arquivo_saida_resultados,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ Arquivo gerado com sucesso: {arquivo_saida_resultados}")
print(df_resultados.head())

# =========================================================
# 10. Calcular e salvar métricas
# =========================================================
df_metricas = calcular_metricas(df_resultados)

arquivo_saida_metricas = "/home/marcelo-west/Atividade1/metricas_modelos.csv"

df_metricas.to_csv(
    arquivo_saida_metricas,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ Arquivo de métricas gerado com sucesso: {arquivo_saida_metricas}")
print(df_metricas)