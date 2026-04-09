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

# 4.2 Códigos

Foram desenvolvidos 6 scripts em Python para alcançar o objetivo desta atividade.

# 4.2.1 Classificação:

Os scripts ”Classificacao Questoes Abertas.py” e ”Classificacao Questoes Multipla Escolha.py” classificaram as questões em 4 categorias:

• Triagem: Questões que qualquer leigo ou sistema básico resolve.
• Generalista: Exige conhecimento técnico base, mas não especialização profunda.
• Especialista: Requer conhecimento técnico avançado, protocolos especı́ficos ou histórico de casos complexos.
• Expert: Casos sem precedentes claros ou que exigem tempo maior de experiência e pesquisa.

# 4.2.2 Inferência:

Foram utilizados 3 Modelos de Linguagem de Grande Escala (LLMS), cada modelo de um provedor
distinto para a execução da inferência.

# OpenAI: gpt-4o-mini: 

O GPT-4o-mini é um modelo de linguagem de pequeno porte da OpenAI, projetado para oferecer um equilı́brio otimizado entre alto desempenho e eficiência de custo em tarefas de inteligência artificial. Lançado como um sucessor mais capaz e econômico ao GPT-3.5 Turbo, ele suporta multimodalidade nativa (processando texto, áudio, imagens e vı́deo) e possui uma janela de contexto de
128 mil tokens. Sua arquitetura é especialmente eficaz para aplicações que exigem baixa latência, como assistentes em tempo real e automação de fluxos de trabalho, superando modelos de sua categoria em benchmarks de raciocı́nio acadêmico e codificação, ao mesmo tempo em que se mantém acessı́vel para desenvolvedores que operam em larga escala ou em ambientes com recursos mais limitados.

# Google: gemini-2.5-flash: 

O Gemini 2.5 Flash é um modelo de linguagem da Google, desenvolvido para oferecer velocidade extrema e eficiência sem comprometer a capacidade de raciocı́nio complexo. Como um modelo de porte médio otimizado para multimodalidade nativa, ele se destaca pelo processamento de grandes volumes de dados (long-context) com baixa latência, sendo ideal para aplicações que exigem
respostas rápidas em texto, código, imagens e vı́deo. Sua arquitetura foi refinada para ser altamente eficaz em tarefas de sumarização, extração de dados e conversação fluida, posicionando-se como uma ferramenta robusta para desenvolvedores que buscam escalar soluções de IA com um custo-benefı́cio superior, especialmente em fluxos de trabalho que demandam alta frequência de chamadas de API e processamento em tempo real.

# Perplexity: sonar-pro: 
O Sonar Pro é um modelo de linguagem avançado desenvolvido pela Perplexity, focado em transformar a experiência de busca convencional em uma interface de resposta direta, precisa e atualizada em tempo real. Diferente de modelos estáticos, ele é integrado a um motor de busca que realiza a varredura ativa da web para fundamentar cada resposta com fontes citadas, mitigando significativamente o risco de alucinações. O modelo é otimizado para lidar com consultas complexas que exigem a sı́ntese de múltiplas informações, oferecendo um equilı́brio entre profundidade analı́tica e agilidade, o que o torna uma ferramenta poderosa para pesquisa acadêmica, análise de mercado e verificação de fatos em ambientes onde a validade da informação é a prioridade máxima.

Para efetuar as inferências foram utilizados os scripts ”Inferencia_Questoes_Abertas.py” e ”Inferencia_Questoes_Multipla_Escolha.py”.

Nesta etapa, foram submetidas 17 questões abertas do Dataset M1 (135 a 151 - Itaymanes K-QA) e submetidas 27 questões de Múltipla Escolha do Dataset M2 (217 a 243 - USMLE), sendo gerados 4 arquivos: ”resultadosM1_modelos.csv”, ”resultadosM2_modelos.csv”, ”metricasM1_modelos.csv” e ”metricasM2_modelos.csv”. Os dois primerios arquivos registraram: Identificador, Pergunta, Resposta Correta, Resposta GPT, Resposta Gemini e Resposta Perplexity e os dois últimos arquivos registraram os valores das métricas
accuracy e f1 macro alcançadas.

# 5 Resultados

A Tabela 1 apresenta as questões do Dataset M1 classificadas pelas categorias das perguntas, conforme os modelos de linguagem: OpenAI: gpt-4o-mini, Google: gemini-2.5-flash e Perplexity: sonar-pro.

# 5.1 Classificação das questões do Dataset M1

<img width="1146" height="620" alt="image" src="https://github.com/user-attachments/assets/1f4fe4ac-0fa4-48d9-b812-40a3f2f8ccd1" />

# 5.2 Classificação das questões do Dataset M2

A Tabela 2 apresenta as questões do Dataset M2 classificadas pelas categorias das perguntas, conforme os modelos de linguagem: OpenAI: gpt-4o-mini, Google: gemini-2.5-flash e Perplexity: sonar-pro.

<img width="870" height="971" alt="image" src="https://github.com/user-attachments/assets/02e093d6-5995-43ea-8886-115f1411ab91" />

A Tabela 3 apresenta a concordância e distribuição percentual das classificações, conforme os modelos de linguagem: OpenAI: gpt-4o-mini, Google: gemini-2.5-flash e Perplexity: sonar-pro.

<img width="1141" height="520" alt="image" src="https://github.com/user-attachments/assets/c3391d02-2a3e-4014-b57e-dcb35657d9d2" />

A Tabela 4 apresenta a concordância e distribuição percentual das classificações, conforme os modelos de linguagem: OpenAI: gpt-4o-mini, Google: gemini-2.5-flash e Perplexity: sonar-pro.

<img width="1141" height="520" alt="image" src="https://github.com/user-attachments/assets/90891e6c-9c46-40e9-a3a8-a9501117d1a2" />
