import re
import pandas as pd
import pdfplumber
import os

def limpar_profundo(txt):
    if not txt: return ""
    # Normaliza espaços, quebras de linha e caracteres especiais
    return " ".join(txt.split()).strip()

def extrair_usmle_completo_v5(caminho_pdf):
    if not os.path.exists(caminho_pdf):
        print(f"Erro: Arquivo {caminho_pdf} não encontrado.")
        return None

    texto_completo = ""
    print("Lendo PDF e consolidando texto...")
    with pdfplumber.open(caminho_pdf) as pdf:
        for pagina in pdf.pages:
            content = pagina.extract_text()
            if content:
                texto_completo += content + "\n\n"

    # 1. Separar blocos por Identificador (Question X.X ... Step X)
    padrao_id = r'(Question\s+\d+\.\d+.*?Step\s+\d+)'
    partes = re.split(padrao_id, texto_completo)
    
    data = []
    # O split retorna [lixo, ID1, Corpo1, ID2, Corpo2...]
    for i in range(1, len(partes), 2):
        try:
            identificador = limpar_profundo(partes[i])
            corpo = partes[i+1]

            # 2. Extração de Gabaritos e Metadados
            res_correta = re.search(r'Correct Response:\s*([A-E])', corpo)
            res_open = re.search(r'OpenEvidence Response:\s*([A-E])', corpo)
            link_open = re.search(r'OpenEvidence Link:\s*(https?://[^\s]+)', corpo)
            
            # 3. Localizar e extrair Explicação e Referências
            # A explicação começa em "OpenEvidence Explanation" e vai até "References"
            # As referências vão de "References" até o fim do bloco ou próxima questão
            expl_match = re.search(r'OpenEvidence Explanation(.*?)(?=References|$)', corpo, re.DOTALL)
            ref_match = re.search(r'References(.*?)$', corpo, re.DOTALL)

            # 4. Processar Enunciado e Opções
            match_a = re.search(r'\(\s*A\s*\)', corpo)
            if not match_a: continue
            pos_a = match_a.start()
            
            questao_texto = limpar_profundo(corpo[:pos_a])

            def obter_opt(l, prox):
                p = fr'\({l}\)(.*?)(?=\({prox}\)|Correct Response:)'
                m = re.search(p, corpo, re.DOTALL)
                return f"({l}) {limpar_profundo(m.group(1))}" if m else ""

            o1, o2, o3, o4 = obter_opt('A','B'), obter_opt('B','C'), obter_opt('C','D'), obter_opt('D','E')
            m_e = re.search(r'\(E\)(.*?)(?=Correct Response:)', corpo, re.DOTALL)
            o5 = f"(E) {limpar_profundo(m_e.group(1))}" if m_e else ""

            data.append({
                'Identificador': identificador,
                'Questao': questao_texto,
                'opcoesrespostas': f"{o1} {o2} {o3} {o4} {o5}",
                'Resposta Correta': res_correta.group(1) if res_correta else "",
                'Resposta OpenEvidence': res_open.group(1) if res_open else "",
                'Link OpenEvidence': link_open.group(1) if link_open else "",
                'Explicacao OpenEvidence': limpar_profundo(expl_match.group(1)) if expl_match else "",
                'Referências': limpar_profundo(ref_match.group(1)) if ref_match else ""
            })
        except Exception:
            continue

    return pd.DataFrame(data)

# --- Execução ---
path_in = "/home/marcelo-west/USMLE/usmle_report.pdf"
path_out = "/home/marcelo-west/USMLE/usmle_325_questoes_full.csv"

df = extrair_usmle_completo_v5(path_in)

if df is not None:
    # Exportação com ponto e vírgula e aspas para proteção total no LibreOffice
    df.to_csv(path_out, index=False, sep=';', encoding='utf-8-sig', quoting=1)
    print(f"Sucesso! Planilha gerada com {len(df)} questões e todas as colunas de fundamentação.")