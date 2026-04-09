import os
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

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
# 2. Função para normalizar a classificação
# =========================================================
def extrair_classificacao(resposta):
    if not resposta:
        return "ERRO"

    resposta_limpa = str(resposta).strip().lower()

    mapa = {
        "triagem": "Triagem",
        "generalista": "Generalista",
        "especialista": "Especialista",
        "expert": "Expert"
    }

    for chave, valor in mapa.items():
        if chave in resposta_limpa:
            return valor

    return "ERRO"

# =========================================================
# 3. Prompt padrão para classificação
# =========================================================
def montar_prompt_classificacao(pergunta):
    return f"""
Classifique a pergunta abaixo em apenas uma das quatro categorias:

1. Triagem -> para questões que qualquer leigo ou sistema básico resolve.
2. Generalista -> para questões que exigem conhecimento técnico base, mas não especialização profunda.
3. Especialista -> para questões que requerem conhecimento técnico avançado, protocolos específicos ou histórico de casos complexos.
4. Expert -> para questões de casos sem precedentes claros ou que exigem maior tempo de experiência e pesquisa.

Responda APENAS com uma destas palavras:
Triagem
Generalista
Especialista
Expert

Pergunta:
{pergunta}
"""

# =========================================================
# 4. Funções de consulta aos modelos
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
# 5. Carregamento do CSV
# =========================================================
caminho_arquivo = "/home/marcelo-west/Atividade1/dataset_usmle_questoes_de_217_a_243_Marcelo_West.csv"

df = pd.read_csv(
    caminho_arquivo,
    sep=";",
    encoding="utf-8-sig"
)

df = df[["Identificador", "Pergunta"]].copy()

# =========================================================
# 6. Classificação com múltiplos modelos
# =========================================================
def classificar_perguntas_multiplos_modelos(dataframe, limite=None):
    resultados = []

    if limite is not None:
        dataframe = dataframe.head(limite)

    for contador, (_, linha) in enumerate(dataframe.iterrows(), start=1):
        identificador = linha["Identificador"]
        pergunta = linha["Pergunta"]

        print(f"{contador} - Processando Identificador: {identificador}")

        try:
            classificacao_gpt = get_openai_classificacao(pergunta)
        except Exception as e:
            print(f"Erro GPT no Identificador {identificador}: {e}")
            classificacao_gpt = "ERRO"

        try:
            classificacao_gemini = get_gemini_classificacao(pergunta)
        except Exception as e:
            print(f"Erro Gemini no Identificador {identificador}: {e}")
            classificacao_gemini = "ERRO"

        try:
            classificacao_perplexity = get_perplexity_classificacao(pergunta)
        except Exception as e:
            print(f"Erro Perplexity no Identificador {identificador}: {e}")
            classificacao_perplexity = "ERRO"

        resultados.append({
            "Identificacao": identificador,
            "Pergunta": pergunta,
            "Classificacao_Gpt": classificacao_gpt,
            "Classificacao_Gemini": classificacao_gemini,
            "Classificacao_Perplexity": classificacao_perplexity
        })

    return pd.DataFrame(resultados)

# =========================================================
# 7. Função para gerar análise de concordância e distribuição
# =========================================================
def gerar_analise_concordancia(df_resultados):
    total_questoes = len(df_resultados)

    concordancia_total = (
        (df_resultados["Classificacao_Gpt"] == df_resultados["Classificacao_Gemini"]) &
        (df_resultados["Classificacao_Gpt"] == df_resultados["Classificacao_Perplexity"])
    ).sum()

    divergencia = total_questoes - concordancia_total

    linhas_analise = [
        {
            "tipo_analise": "resumo_geral",
            "modelo": "Todos",
            "categoria": "Total_Questoes",
            "quantidade": total_questoes,
            "percentual": 100.0
        },
        {
            "tipo_analise": "resumo_geral",
            "modelo": "Todos",
            "categoria": "Concordancia_3_Modelos",
            "quantidade": int(concordancia_total),
            "percentual": round((concordancia_total / total_questoes) * 100, 2) if total_questoes > 0 else 0.0
        },
        {
            "tipo_analise": "resumo_geral",
            "modelo": "Todos",
            "categoria": "Divergencia",
            "quantidade": int(divergencia),
            "percentual": round((divergencia / total_questoes) * 100, 2) if total_questoes > 0 else 0.0
        }
    ]

    categorias = ["Triagem", "Generalista", "Especialista", "Expert", "ERRO"]

    modelos = {
        "GPT": "Classificacao_Gpt",
        "Gemini": "Classificacao_Gemini",
        "Perplexity": "Classificacao_Perplexity"
    }

    for nome_modelo, coluna in modelos.items():
        contagem = df_resultados[coluna].value_counts()

        for categoria in categorias:
            qtd = int(contagem.get(categoria, 0))
            percentual = round((qtd / total_questoes) * 100, 2) if total_questoes > 0 else 0.0

            linhas_analise.append({
                "tipo_analise": "distribuicao_percentual",
                "modelo": nome_modelo,
                "categoria": categoria,
                "quantidade": qtd,
                "percentual": percentual
            })

    return pd.DataFrame(linhas_analise)

# =========================================================
# 8. Função para gerar gráficos
# =========================================================
def gerar_grafico_barras(df_resultados, pasta_saida):
    categorias = ["Triagem", "Generalista", "Especialista", "Expert", "ERRO"]

    modelos = {
        "GPT": df_resultados["Classificacao_Gpt"],
        "Gemini": df_resultados["Classificacao_Gemini"],
        "Perplexity": df_resultados["Classificacao_Perplexity"]
    }

    df_plot = pd.DataFrame(index=categorias)

    for nome_modelo, serie in modelos.items():
        contagem = serie.value_counts()
        df_plot[nome_modelo] = [contagem.get(cat, 0) for cat in categorias]

    ax = df_plot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Distribuição das classificações por modelo")
    ax.set_xlabel("Categoria")
    ax.set_ylabel("Quantidade de questões")
    plt.xticks(rotation=45)
    plt.tight_layout()

    caminho = os.path.join(pasta_saida, "grafico_barras_distribuicao.png")
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
    plt.title(f"Distribuição percentual das classificações - {nome_modelo}")
    plt.tight_layout()

    nome_arquivo = f"grafico_pizza_{nome_modelo.lower()}.png"
    caminho = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho, dpi=300)
    plt.close()

def gerar_heatmap_concordancia(df_resultados, pasta_saida):
    modelos = ["GPT", "Gemini", "Perplexity"]
    colunas = {
        "GPT": "Classificacao_Gpt",
        "Gemini": "Classificacao_Gemini",
        "Perplexity": "Classificacao_Perplexity"
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
    ax.set_title("Heatmap de concordância entre modelos (%)")

    for i in range(len(modelos)):
        for j in range(len(modelos)):
            ax.text(j, i, f"{matriz.values[i, j]:.1f}%",
                    ha="center", va="center")

    fig.colorbar(cax)
    plt.tight_layout()

    caminho = os.path.join(pasta_saida, "heatmap_concordancia.png")
    plt.savefig(caminho, dpi=300)
    plt.close()

# =========================================================
# 9. Executar classificação
# =========================================================
df_resultados = classificar_perguntas_multiplos_modelos(df, limite=None)

# Para teste:
# df_resultados = classificar_perguntas_multiplos_modelos(df, limite=3)

# =========================================================
# 10. Salvar arquivo principal
# =========================================================
pasta_saida = "/home/marcelo-west/Atividade1"

arquivo_saida_resultados = os.path.join(pasta_saida, "resultados_classificacao_modelos.csv")

df_resultados.to_csv(
    arquivo_saida_resultados,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ Arquivo principal gerado com sucesso: {arquivo_saida_resultados}")
print(df_resultados.head())

# =========================================================
# 11. Gerar e salvar análise de concordância
# =========================================================
df_analise = gerar_analise_concordancia(df_resultados)

arquivo_saida_analise = os.path.join(pasta_saida, "analise_concordancia_modelos.csv")

df_analise.to_csv(
    arquivo_saida_analise,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ Arquivo de análise gerado com sucesso: {arquivo_saida_analise}")
print(df_analise)

# =========================================================
# 12. Gerar gráficos
# =========================================================
gerar_grafico_barras(df_resultados, pasta_saida)
gerar_grafico_pizza(df_resultados, pasta_saida, "Classificacao_Gpt", "GPT")
gerar_grafico_pizza(df_resultados, pasta_saida, "Classificacao_Gemini", "Gemini")
gerar_grafico_pizza(df_resultados, pasta_saida, "Classificacao_Perplexity", "Perplexity")
gerar_heatmap_concordancia(df_resultados, pasta_saida)

print("\n✅ Gráficos gerados com sucesso!")
print(os.path.join(pasta_saida, "grafico_barras_distribuicao.png"))
print(os.path.join(pasta_saida, "grafico_pizza_gpt.png"))
print(os.path.join(pasta_saida, "grafico_pizza_gemini.png"))
print(os.path.join(pasta_saida, "grafico_pizza_perplexity.png"))
print(os.path.join(pasta_saida, "heatmap_concordancia.png"))