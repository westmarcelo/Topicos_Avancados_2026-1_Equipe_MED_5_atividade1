import os
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

# =========================================================
# 1. API configuration
# =========================================================
load_dotenv()

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client_pplx = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =========================================================
# 2. Function to normalize classification
# =========================================================
def extrair_classificacao(resposta):
    if not resposta:
        return "ERROR"

    resposta_limpa = str(resposta).strip().lower()

    mapa = {
        "triagem": "Triagem",
        "generalista": "Generalista",
        "especialista": "Especialista",
        "expert": "Expert",
        "triage": "Triagem",
        "generalist": "Generalista",
        "specialist": "Especialista"
    }

    for chave, valor in mapa.items():
        if chave in resposta_limpa:
            return valor

    return "ERROR"

# =========================================================
# 3. Classification prompt
# =========================================================
def montar_prompt_classificacao(pergunta):
    return f"""
Classify the question below into only one of the following four categories:

1. Triagem -> questions that any layperson or basic system can solve.
2. Generalista -> questions that require basic technical knowledge, but not deep specialization.
3. Especialista -> questions that require advanced technical knowledge, specific protocols, or a history of complex cases.
4. Expert -> questions involving cases without clear precedents or that require longer experience and research time.

Answer ONLY with one of these words:
Triagem
Generalista
Especialista
Expert

Question:
{pergunta}
"""

# =========================================================
# 4. Model calls
# =========================================================
def get_openai_classificacao(pergunta):
    prompt = montar_prompt_classificacao(pergunta)

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    texto = response.choices[0].message.content
    return extrair_classificacao(texto)

def get_gemini_classificacao(pergunta):
    prompt = montar_prompt_classificacao(pergunta)

    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    texto = response.text if response.text else ""
    return extrair_classificacao(texto)

def get_perplexity_classificacao(pergunta):
    prompt = montar_prompt_classificacao(pergunta)

    response = client_pplx.chat.completions.create(
        model="sonar-pro",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    texto = response.choices[0].message.content
    return extrair_classificacao(texto)

# =========================================================
# 5. Load CSV
# =========================================================
caminho_arquivo = "/home/marcelo-west/Atividade1/questions_w_answers_questoes_de_135_a_151_Marcelo_West_new.csv"

df = pd.read_csv(
    caminho_arquivo,
    sep=";",
    encoding="utf-8-sig"
)

df = df[["Question"]].copy()
df.insert(0, "Identificator", range(1, len(df) + 1))

# =========================================================
# 6. Multi-model classification
# =========================================================
def classificar_perguntas_multiplos_modelos(dataframe, limite=None):
    resultados = []

    if limite is not None:
        dataframe = dataframe.head(limite)

    for _, linha in dataframe.iterrows():
        identificador = linha["Identificator"]
        pergunta = linha["Question"]

        print(f"{identificador} - Processing Question")

        try:
            classificacao_gpt = get_openai_classificacao(pergunta)
        except Exception as e:
            print(f"GPT error on Identificator {identificador}: {e}")
            classificacao_gpt = "ERROR"

        try:
            classificacao_gemini = get_gemini_classificacao(pergunta)
        except Exception as e:
            print(f"Gemini error on Identificator {identificador}: {e}")
            classificacao_gemini = "ERROR"

        try:
            classificacao_perplexity = get_perplexity_classificacao(pergunta)
        except Exception as e:
            print(f"Perplexity error on Identificator {identificador}: {e}")
            classificacao_perplexity = "ERROR"

        resultados.append({
            "Identificator": identificador,
            "Question": pergunta,
            "Classification_GPT": classificacao_gpt,
            "Classification_Gemini": classificacao_gemini,
            "Classification_Perplexity": classificacao_perplexity
        })

    return pd.DataFrame(resultados)

# =========================================================
# 7. Agreement and percentage distribution analysis
# =========================================================
def gerar_analise_concordancia(df_resultados):
    total_questoes = len(df_resultados)

    concordancia_total = (
        (df_resultados["Classification_GPT"] == df_resultados["Classification_Gemini"]) &
        (df_resultados["Classification_GPT"] == df_resultados["Classification_Perplexity"])
    ).sum()

    divergencia = total_questoes - concordancia_total

    linhas_analise = [
        {
            "analysis_type": "general_summary",
            "model": "All",
            "category": "Total_Questions",
            "quantity": total_questoes,
            "percentage": 100.0
        },
        {
            "analysis_type": "general_summary",
            "model": "All",
            "category": "Agreement_3_Models",
            "quantity": int(concordancia_total),
            "percentage": round((concordancia_total / total_questoes) * 100, 2) if total_questoes > 0 else 0.0
        },
        {
            "analysis_type": "general_summary",
            "model": "All",
            "category": "Divergence",
            "quantity": int(divergencia),
            "percentage": round((divergencia / total_questoes) * 100, 2) if total_questoes > 0 else 0.0
        }
    ]

    categorias = ["Triagem", "Generalista", "Especialista", "Expert", "ERROR"]

    modelos = {
        "GPT": "Classification_GPT",
        "Gemini": "Classification_Gemini",
        "Perplexity": "Classification_Perplexity"
    }

    for nome_modelo, coluna in modelos.items():
        contagem = df_resultados[coluna].value_counts()

        for categoria in categorias:
            qtd = int(contagem.get(categoria, 0))
            percentual = round((qtd / total_questoes) * 100, 2) if total_questoes > 0 else 0.0

            linhas_analise.append({
                "analysis_type": "percentage_distribution",
                "model": nome_modelo,
                "category": categoria,
                "quantity": qtd,
                "percentage": percentual
            })

    return pd.DataFrame(linhas_analise)

# =========================================================
# 8. Charts
# =========================================================
def gerar_grafico_barras(df_resultados, pasta_saida):
    categorias = ["Triagem", "Generalista", "Especialista", "Expert", "ERROR"]

    modelos = {
        "GPT": df_resultados["Classification_GPT"],
        "Gemini": df_resultados["Classification_Gemini"],
        "Perplexity": df_resultados["Classification_Perplexity"]
    }

    df_plot = pd.DataFrame(index=categorias)

    for nome_modelo, serie in modelos.items():
        contagem = serie.value_counts()
        df_plot[nome_modelo] = [contagem.get(cat, 0) for cat in categorias]

    ax = df_plot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Distribution of classifications by model")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of questions")
    plt.xticks(rotation=45)
    plt.tight_layout()

    caminho = os.path.join(pasta_saida, "grafico_barras_distribuicao_questions_135_151.png")
    plt.savefig(caminho, dpi=300)
    plt.close()

def gerar_grafico_pizza(df_resultados, pasta_saida, coluna_modelo, nome_modelo):
    contagem = df_resultados[coluna_modelo].value_counts()

    if contagem.empty:
        return

    plt.figure(figsize=(7, 7))
    plt.pie(
        contagem.values,
        labels=contagem.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title(f"Percentage distribution of classifications - {nome_modelo}")
    plt.tight_layout()

    nome_arquivo = f"grafico_pizza_{nome_modelo.lower()}_questions_135_151.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300)
    plt.close()

def gerar_heatmap_concordancia(df_resultados, pasta_saida):
    modelos = ["GPT", "Gemini", "Perplexity"]
    colunas = {
        "GPT": "Classification_GPT",
        "Gemini": "Classification_Gemini",
        "Perplexity": "Classification_Perplexity"
    }

    matriz = pd.DataFrame(index=modelos, columns=modelos, dtype=float)

    total = len(df_resultados)

    for modelo_linha in modelos:
        for modelo_coluna in modelos:
            concordantes = (
                df_resultados[colunas[modelo_linha]] == df_resultados[colunas[modelo_coluna]]
            ).sum()

            percentual = (concordantes / total) * 100 if total > 0 else 0
            matriz.loc[modelo_linha, modelo_coluna] = percentual

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(matriz.values, aspect="auto")

    ax.set_xticks(range(len(modelos)))
    ax.set_yticks(range(len(modelos)))
    ax.set_xticklabels(modelos)
    ax.set_yticklabels(modelos)
    ax.set_title("Agreement heatmap between models (%)")

    for i in range(len(modelos)):
        for j in range(len(modelos)):
            ax.text(j, i, f"{matriz.values[i, j]:.1f}%",
                    ha="center", va="center")

    fig.colorbar(cax)
    plt.tight_layout()

    caminho = os.path.join(pasta_saida, "heatmap_concordancia_questions_135_151.png")
    plt.savefig(caminho, dpi=300)
    plt.close()

# =========================================================
# 9. Run classification
# =========================================================
df_resultados = classificar_perguntas_multiplos_modelos(df, limite=None)

# For testing:
# df_resultados = classificar_perguntas_multiplos_modelos(df, limite=3)

# =========================================================
# 10. Save main CSV
# =========================================================
pasta_saida = "/home/marcelo-west/Atividade1"

arquivo_saida_resultados = os.path.join(
    pasta_saida,
    "resultados_classificacao_questions_135_151.csv"
)

df_resultados.to_csv(
    arquivo_saida_resultados,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print(f"\n✅ Main output file generated successfully: {arquivo_saida_resultados}")
print(df_resultados.head())

# =========================================================
# 11. Save agreement analysis CSV
# =========================================================
df_analise = gerar_analise_concordancia(df_resultados)

arquivo_saida_analise = os.path.join(
    pasta_saida,
    "analise_concordancia_questions_135_151.csv"
)

df_analise.to_csv(
    arquivo_saida_analise,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print(f"\n✅ Agreement analysis file generated successfully: {arquivo_saida_analise}")
print(df_analise)

# =========================================================
# 12. Generate charts
# =========================================================
gerar_grafico_barras(df_resultados, pasta_saida)
gerar_grafico_pizza(df_resultados, pasta_saida, "Classification_GPT", "GPT")
gerar_grafico_pizza(df_resultados, pasta_saida, "Classification_Gemini", "Gemini")
gerar_grafico_pizza(df_resultados, pasta_saida, "Classification_Perplexity", "Perplexity")
gerar_heatmap_concordancia(df_resultados, pasta_saida)

print("\n✅ Charts generated successfully!")
print(os.path.join(pasta_saida, "grafico_barras_distribuicao_questions_135_151.png"))
print(os.path.join(pasta_saida, "grafico_pizza_gpt_questions_135_151.png"))
print(os.path.join(pasta_saida, "grafico_pizza_gemini_questions_135_151.png"))
print(os.path.join(pasta_saida, "grafico_pizza_perplexity_questions_135_151.png"))
print(os.path.join(pasta_saida, "heatmap_concordancia_questions_135_151.png"))