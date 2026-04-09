import json
import pandas as pd

arquivo_entrada = "/home/marcelo-west/Atividade1/questions_w_answers.jsonl"
arquivo_saida = "questions_w_answers_expandido.csv"

registros = []

with open(arquivo_entrada, "r", encoding="utf-8") as f:
    for linha in f:
        item = json.loads(linha)
        registro = {}

        for chave, valor in item.items():
            if valor is None:
                registro[chave] = ""

            elif isinstance(valor, list):
                # Cria colunas separadas para cada item da lista
                if len(valor) == 0:
                    registro[chave] = ""
                else:
                    for i, elemento in enumerate(valor, start=1):
                        registro[f"{chave}_{i}"] = str(elemento)

            elif isinstance(valor, dict):
                # Expande dicionários em subcolunas
                for subchave, subvalor in valor.items():
                    registro[f"{chave}_{subchave}"] = str(subvalor)

            else:
                registro[chave] = valor

        registros.append(registro)

df = pd.DataFrame(registros)

colunas_prioritarias = ["Question", "Free_form_answer"]
colunas_existentes_prioritarias = [c for c in colunas_prioritarias if c in df.columns]
outras_colunas = [c for c in df.columns if c not in colunas_existentes_prioritarias]

df = df[colunas_existentes_prioritarias + outras_colunas]

df.to_csv(
    arquivo_saida,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print(f"Arquivo CSV expandido gerado com sucesso: {arquivo_saida}")
print(f"Total de registros: {len(df)}")
print(f"Total de colunas: {len(df.columns)}")
print("\nColunas encontradas:")
print(df.columns.tolist())