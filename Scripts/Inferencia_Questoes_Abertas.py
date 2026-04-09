import os
import re
import string
import pandas as pd
from openai import OpenAI
from google import genai
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

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
# 2. Função de normalização
# =========================================================
def normalizar_texto(texto):
    if texto is None:
        return ""

    texto = str(texto).strip().lower()
    texto = re.sub(r"\s+", " ", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    return texto.strip()

# =========================================================
# 3. Função BLEU
# =========================================================
def calcular_bleu(referencia, predicao):
    ref = normalizar_texto(referencia).split()
    pred = normalizar_texto(predicao).split()

    if not ref or not pred:
        return 0.0

    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref], pred, smoothing_function=smoothie)

# =========================================================
# 4. Prompt padrão em inglês
# =========================================================
def montar_prompt(pergunta):
    return f"""
Answer the following question in English.

Provide a concise, direct, and correct answer.
Do not translate the question.
Do not answer in Portuguese.

Question:
{pergunta}
"""

# =========================================================
# 5. Funções de consulta aos modelos
# =========================================================
def get_openai_response(pergunta):
    prompt = montar_prompt(pergunta)

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    texto = response.choices[0].message.content
    return texto.strip() if texto else "ERRO"

def get_gemini_response(pergunta):
    prompt = montar_prompt(pergunta)

    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    texto = response.text if response.text else ""
    return texto.strip() if texto else "ERRO"

def get_perplexity_response(pergunta):
    prompt = montar_prompt(pergunta)

    response = client_pplx.chat.completions.create(
        model="sonar-pro",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    texto = response.choices[0].message.content
    return texto.strip() if texto else "ERRO"

# =========================================================
# 6. Carregamento do CSV
# =========================================================
caminho_arquivo = "/home/marcelo-west/Atividade1/questions_w_answers_questoes_de_135_a_151_Marcelo_West_new.csv"

df = pd.read_csv(
    caminho_arquivo,
    sep=";",
    encoding="utf-8-sig"
)

df = df[["Question", "Free_form_answer"]].copy()

df = df.rename(columns={
    "Question": "pergunta",
    "Free_form_answer": "resposta_correta"
})

df.insert(0, "identificador", range(1, len(df) + 1))

# =========================================================
# 7. Inferência com múltiplos modelos
# =========================================================
def inferir_perguntas_multiplos_modelos(dataframe, limite=None):
    resultados = []

    if limite is not None:
        dataframe = dataframe.head(limite)

    for _, linha in dataframe.iterrows():
        identificador = linha["identificador"]
        pergunta = linha["pergunta"]
        resposta_correta = linha["resposta_correta"]

        print(f"{identificador} - Processing question")

        try:
            resposta_gpt = get_openai_response(pergunta)
        except Exception as e:
            print(f"GPT error on identifier {identificador}: {e}")
            resposta_gpt = "ERRO"

        try:
            resposta_gemini = get_gemini_response(pergunta)
        except Exception as e:
            print(f"Gemini error on identifier {identificador}: {e}")
            resposta_gemini = "ERRO"

        try:
            resposta_perplexity = get_perplexity_response(pergunta)
        except Exception as e:
            print(f"Perplexity error on identifier {identificador}: {e}")
            resposta_perplexity = "ERRO"

        resultados.append({
            "identificador": identificador,
            "pergunta": pergunta,
            "resposta_correta": resposta_correta,
            "resposta_gpt": resposta_gpt,
            "resposta_gemini": resposta_gemini,
            "resposta_perplexity": resposta_perplexity
        })

    return pd.DataFrame(resultados)

# =========================================================
# 8. Cálculo das métricas BLEU e ROUGE-L
# =========================================================
def calcular_metricas_respostas_abertas(df_resultados):
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    modelos = {
        "GPT": "resposta_gpt",
        "Gemini": "resposta_gemini",
        "Perplexity": "resposta_perplexity"
    }

    metricas = []

    for nome_modelo, coluna_modelo in modelos.items():
        bleus = []
        rougel_f = []

        for _, linha in df_resultados.iterrows():
            referencia = str(linha["resposta_correta"]) if pd.notna(linha["resposta_correta"]) else ""
            predicao = str(linha[coluna_modelo]) if pd.notna(linha[coluna_modelo]) else ""

            bleus.append(calcular_bleu(referencia, predicao))

            rouge_scores = scorer.score(referencia, predicao)
            rougel_f.append(rouge_scores["rougeL"].fmeasure)

        metricas.append({
            "modelo": nome_modelo,
            "bleu": round(sum(bleus) / len(bleus), 4) if bleus else 0.0,
            "rougeL_f1": round(sum(rougel_f) / len(rougel_f), 4) if rougel_f else 0.0
        })

    return pd.DataFrame(metricas)

# =========================================================
# 9. Executar inferência
# =========================================================
df_resultados = inferir_perguntas_multiplos_modelos(df, limite=None)

# Para teste:
# df_resultados = inferir_perguntas_multiplos_modelos(df, limite=3)

# =========================================================
# 10. Salvar resultados
# =========================================================
arquivo_saida_resultados = "/home/marcelo-west/Atividade1/resultados_respostas_abertas_modelos.csv"

df_resultados.to_csv(
    arquivo_saida_resultados,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print(f"\n✅ Result file generated successfully: {arquivo_saida_resultados}")
print(df_resultados.head())

# =========================================================
# 11. Calcular e salvar métricas
# =========================================================
df_metricas = calcular_metricas_respostas_abertas(df_resultados)

arquivo_saida_metricas = "/home/marcelo-west/Atividade1/metricas_respostas_abertas_modelos.csv"

df_metricas.to_csv(
    arquivo_saida_metricas,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print(f"\n✅ Metrics file generated successfully: {arquivo_saida_metricas}")
print(df_metricas)