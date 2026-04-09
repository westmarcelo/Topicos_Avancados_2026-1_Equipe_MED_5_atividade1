# Topicos_Avancados_2026-1_Equipe_MED_5

# 1 Atividade 1 - Curadoria de Datasets e Inferência Básica com LLMs

# 2 Objetivo

Este trabalho tem como objetivo principal estabelecer um protocolo de curadoria de dados para a avaliação de
Modelos de Linguagem de Grande Escala (LLMS), integrando conjuntos de informações com estruturas comple-
mentares. A metodologia fundamenta-se na análise de dois datasets do domı́nio médico: o primeiro, composto
por questões abertas, visa mensurar a capacidade de sı́ntese e geração de texto livre; o segundo, baseado em itens
de múltipla escolha, foca na validação da precisão técnica e no raciocı́nio dedutivo especializado sob condições
de resposta estruturada.

Para além da organização dos dados, a investigação propõe uma etapa de imersão analı́tica voltada à inferência
básica, servindo como uma fase exploratória essencial para compreender o comportamento dos modelos em
domı́nios de conhecimento especı́ficos. Este contato inicial com os processos inferenciais permite identificar
as nuances da arquitetura dos LLMs e os desafios técnicos associados ao processamento de linguagem natural,
fornecendo a base empı́rica necessária para o desenvolvimento de protocolos de avaliação mais complexos e para
a otimização de sistemas de suporte à decisão. A Figura 7 apresenta as Etapas da Execução da Atividade 1.



# 3 Domı́nio Médico

A utilização de modelos de linguagem de grande escala no domı́nio médico justifica-se pela sua capacidade
de processar e sintetizar vastos volumes de dados clı́nicos não estruturados, proporcionando uma avaliação
padronizada e escalável da competência profissional (Kung et al., 2023), (Reading Turchioe et al., 2022).
Ao contrário das métricas de avaliação tradicionais, que podem ser limitadas pela subjetividade humana ou
pela latência no processamento, os LLMs permitem a realização de benchmarks rigorosos contra padrões-ouro,
como o USMLE, demonstrando uma acurácia que muitas vezes atinge o limiar de aprovação exigido de médicos
especialistas (Kassab et al., 2023), (Kung et al., 2023).

Além disso, a aplicação de modelos especializados, como a abordagem desta atividade, não apenas pode
otimizar o fluxo de trabalho hospitalar através da automação de classificações diagnósticas a partir de laudos
textuais, mas também pode estabelecer uma infraestrutura robusta para a educação médica assistida por IA e para
o desenvolvimento de sistemas de suporte à decisão de alta precisão (Pedrosa et al., 2021), (Vishwanath et al.,
2025), (Wang et al., 2025).

# 4 Datasets, Códigos e Modelos de Linguagem

# 4.1 Datasets
Foram utilizados 2 Datasets da Área Médica, cada Dataset de um repositório distinto.
Dataset M1 - Questões Abertas: 135 a 151 - Itaymanes K-QA: O Dataset M1 (K-QA) é um recurso
especializado para tarefas de Perguntas e Respostas (QA) no domı́nio médico, focado em fornecer respostas
fundamentadas em evidências clı́nicas de alta qualidade. Ele foi desenvolvido a partir de diretrizes de prática
clı́nica e fontes médicas confiáveis, visando mitigar alucinações em modelos de linguagem. O diferencial deste
conjunto de dados é o foco em informações acionáveis e tecnicamente precisas, sendo frequentemente utilizado
para avaliar a capacidade de LLMs em lidar com terminologia médica complexa e raciocı́nio clı́nico em cenários
onde a precisão é crı́tica (Manes et al., 2024).

Este dataset é estruturado para suportar tarefas de recuperação de informação e geração de texto, contendo
pares de perguntas e respostas que refletem dúvidas reais de profissionais de saúde e estudantes de medicina.
Ele serve como um importante benchmark para modelos que buscam especialização em domı́nios técnicos,
permitindo testar se o modelo consegue não apenas responder corretamente, mas também alinhar sua resposta
aos protocolos médicos estabelecidos (Manes et al., 2024).

Na preparação do Dataset M1 para a execução da atividade, o mesmo foi convertido do formato original
.JSONL para o formato .CSV através do script Converte_jsonl_para_csv.py

Clique aqui para visualizar o Dataset M1 Original:

# Dataset M2 - Múltipla Escolha com Gabarito: 217 a 243 - USMLE: 

O Dataset M2 (USMLE - United States Medical Licensing Examination) diz respeito ao exame de licenciamento médico, que consiste em três etapas, sendo este um requisito para os profissionais da medicina que pretendem atuar nos Estados Unidos.
A avaliação verifica a habilidade de um médico em aplicar conhecimentos, conceitos e princı́pios, além de
evidenciar competências focadas no paciente, as quais são relevantes tanto na saúde quanto na doença e formam
a base para um atendimento seguro e eficaz. No âmbito da Inteligência Artificial e do Processamento de
Linguagem Natural, pesquisadores empregam as questões desse exame para desenvolver datasets de referência
(benchmarks), como o MedQA, que desafiam os modelos a solucionar problemas de raciocı́nio clı́nico de elevada
complexidade (Kassab et al., 2023).

Os datasets derivados do USMLE são reconhecidos como a referência máxima para a avaliação da competência médica em Modelos de Linguagem de Grande Escala (LLMs), uma vez que as questões requerem não somente a memorização de informações, mas também a integração de dados diagnósticos, a compreensão da fisiopatologia e a habilidade de tomada de decisão clı́nica. Devido à sua natureza exigente e à constante atualização, a aplicação de seus dados para o treinamento e a avaliação de modelos possibilita a medição precisa do avanço da inteligência artificial em direção a habilidades profissionais no campo da saúde, funcionando como
um indicador de desempenho em contextos nos quais a precisão e o raciocı́nio clı́nico se mostram fundamentais
(Kassab et al., 2023).

Na preparação do Dataset M2 para a execução da atividade, o mesmo foi convertido do formato original .PDF para o formato .CSV através do script Converte_pdf_para_csv.py

Clique aqui para visualizar o Dataset M2 Original:
