# Topicos_Avancados_2026-1_Equipe_MED_5

# 1 Atividade 1 - Curadoria de Datasets e Inferência Básica com LLMs

# 2 Objetivo

Este trabalho tem como objetivo principal estabelecer um protocolo de curadoria de dados para a avaliação de Modelos de Linguagem de Grande Escala (LLMS), integrando conjuntos de informações com estruturas complementares. A metodologia fundamenta-se na análise de dois datasets do domı́nio médico: o primeiro, composto por questões abertas, visa mensurar a capacidade de sı́ntese e geração de texto livre; o segundo, baseado em itens de múltipla escolha, foca na validação da precisão técnica e no raciocı́nio dedutivo especializado sob condições de resposta estruturada.

Para além da organização dos dados, a investigação propõe uma etapa de imersão analı́tica voltada à inferência básica, servindo como uma fase exploratória essencial para compreender o comportamento dos modelos em domı́nios de conhecimento especı́ficos. Este contato inicial com os processos inferenciais permite identificar as nuances da arquitetura dos LLMs e os desafios técnicos associados ao processamento de linguagem natural, fornecendo a base empı́rica necessária para o desenvolvimento de protocolos de avaliação mais complexos e para a otimização de sistemas de suporte à decisão. A Figura 7 apresenta as Etapas da Execução da Atividade 1.

<img width="516" height="996" alt="image" src="https://github.com/user-attachments/assets/a4803567-f323-47f1-854c-baf9ce2693d9" />

# 3 Domı́nio Médico

A utilização de modelos de linguagem de grande escala no domı́nio médico justifica-se pela sua capacidade de processar e sintetizar vastos volumes de dados clı́nicos não estruturados, proporcionando uma avaliação padronizada e escalável da competência profissional (Kung et al., 2023), (Reading Turchioe et al., 2022).

Ao contrário das métricas de avaliação tradicionais, que podem ser limitadas pela subjetividade humana ou pela latência no processamento, os LLMs permitem a realização de benchmarks rigorosos contra padrões-ouro, como o USMLE, demonstrando uma acurácia que muitas vezes atinge o limiar de aprovação exigido de médicos especialistas (Kassab et al., 2023), (Kung et al., 2023).

Além disso, a aplicação de modelos especializados, como a abordagem desta atividade, não apenas pode otimizar o fluxo de trabalho hospitalar através da automação de classificações diagnósticas a partir de laudos textuais, mas também pode estabelecer uma infraestrutura robusta para a educação médica assistida por IA e para o desenvolvimento de sistemas de suporte à decisão de alta precisão (Pedrosa et al., 2021), (Vishwanath et al.,2025), (Wang et al., 2025).

# 4 Datasets, Códigos e Modelos de Linguagem

# 4.1 Datasets
Foram utilizados 2 Datasets da Área Médica, cada Dataset de um repositório distinto. 

# Dataset M1 - Questões Abertas: 135 a 151 - Itaymanes K-QA: 
O Dataset M1 (K-QA) é um recurso especializado para tarefas de Perguntas e Respostas (QA) no domı́nio médico, focado em fornecer respostas fundamentadas em evidências clı́nicas de alta qualidade. Ele foi desenvolvido a partir de diretrizes de prática
clı́nica e fontes médicas confiáveis, visando mitigar alucinações em modelos de linguagem. O diferencial deste conjunto de dados é o foco em informações acionáveis e tecnicamente precisas, sendo frequentemente utilizado para avaliar a capacidade de LLMs em lidar com terminologia médica complexa e raciocı́nio clı́nico em cenários onde a precisão é crı́tica (Manes et al., 2024).

Este dataset é estruturado para suportar tarefas de recuperação de informação e geração de texto, contendo pares de perguntas e respostas que refletem dúvidas reais de profissionais de saúde e estudantes de medicina. Ele serve como um importante benchmark para modelos que buscam especialização em domı́nios técnicos, permitindo testar se o modelo consegue não apenas responder corretamente, mas também alinhar sua resposta aos protocolos médicos estabelecidos (Manes et al., 2024).

Na preparação do Dataset M1 para a execução da atividade, o mesmo foi convertido do formato original .JSONL para o formato .CSV através do script Converte_jsonl_para_csv.py

Clique aqui para visualizar o Dataset M1 Original:

# Dataset M2 - Múltipla Escolha com Gabarito: 217 a 243 - USMLE: 

O Dataset M2 (USMLE - United States Medical Licensing Examination) diz respeito ao exame de licenciamento médico, que consiste em três etapas, sendo este um requisito para os profissionais da medicina que pretendem atuar nos Estados Unidos.

A avaliação verifica a habilidade de um médico em aplicar conhecimentos, conceitos e princı́pios, além de evidenciar competências focadas no paciente, as quais são relevantes tanto na saúde quanto na doença e formam a base para um atendimento seguro e eficaz. No âmbito da Inteligência Artificial e do Processamento de Linguagem Natural, pesquisadores empregam as questões desse exame para desenvolver datasets de referência (benchmarks), como o MedQA, que desafiam os modelos a solucionar problemas de raciocı́nio clı́nico de elevada complexidade (Kassab et al., 2023).

Os datasets derivados do USMLE são reconhecidos como a referência máxima para a avaliação da competência médica em Modelos de Linguagem de Grande Escala (LLMs), uma vez que as questões requerem não somente a memorização de informações, mas também a integração de dados diagnósticos, a compreensão da fisiopatologia e a habilidade de tomada de decisão clı́nica. Devido à sua natureza exigente e à constante atualização, a aplicação de seus dados para o treinamento e a avaliação de modelos possibilita a medição precisa do avanço da inteligência artificial em direção a habilidades profissionais no campo da saúde, funcionando como um indicador de desempenho em contextos nos quais a precisão e o raciocı́nio clı́nico se mostram fundamentais (Kassab et al., 2023).

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

# 5.3 Inferência das questões do Dataset M1

Os resultados obtidos indicam um desempenho relativamente próximo entre os três modelos, porém com algumas diferenças relevantes quando analisadas as métricas BLEU e ROUGE-L. O modelo Perplexity apresentou o maior valor de BLEU (0,0286), sugerindo uma maior sobreposição de n-gramas entre as respostas geradas e o padrão ouro, o que pode indicar maior aderência lexical em termos de escolha de palavras.

Em contrapartida, o modelo GPT obteve o melhor desempenho em ROUGE-L (0,1839), métrica que considera a maior subsequência comum entre os textos, refletindo melhor alinhamento estrutural e semântico das respostas. O modelo Gemini, por sua vez, apresentou os menores valores em ambas as métricas (BLEU = 0,0112, ROUGE-L = 0,1559), indicando menor similaridade global com as respostas de referência.

De forma geral, observa-se que todos os valores de BLEU são baixos, o que é esperado em tarefas de respostas abertas, nas quais diferentes formulações corretas podem ser semanticamente equivalentes, mas lexicalmente distintas. Nesse contexto, a métrica ROUGE-L tende a ser mais informativa, pois captura similaridade estrutural mais ampla. 

Assim, os resultados sugerem que o GPT apresenta melhor capacidade de gerar respostas semanticamente alinhadas com o padrão ouro, enquanto o Perplexity demonstra maior proximidade lexical pontual. O desempenho inferior do Gemini pode estar relacionado a variações na forma de resposta ou menor aderência ao estilo esperado pelo dataset. Esses achados reforçam a importância de utilizar múltiplas métricas na avaliação de modelos de linguagem em tarefas abertas, especialmente em domı́nios complexos como o médico.

# 5.4 Inferência das questões do Dataset M2

Os resultados obtidos a partir da avaliação dos modelos de linguagem — GPT, Gemini e Perplexity — demonstram diferenças relevantes de desempenho no contexto de resolução de questões de múltipla escolha na área médica. As métricas utilizadas para análise foram accuracy (acurácia) e F1-score macro, permitindo uma avaliação tanto da taxa global de acertos quanto do equilı́brio entre precisão e revocação entre as classes.

Observa-se que o modelo Gemini apresentou o melhor desempenho geral, alcançando uma acurácia de 0,9629 e um F1-score macro de 0,9701, indicando não apenas alta taxa de acerto, mas também consistência na classificação entre diferentes alternativas. Esse resultado sugere que o modelo possui maior capacidade de generalização e melhor interpretação semântica das questões e das opções de resposta, o que é particularmente relevante em domı́nios complexos como o médico.

O modelo GPT também apresentou desempenho elevado, com acurácia de 0,9259 e F1-score macro de 0,9374. Embora ligeiramente inferior ao Gemini, seus resultados indicam robustez e boa capacidade de tomada de decisão em tarefas de múltipla escolha. A pequena diferença em relação ao Gemini pode estar associada a nuances na compreensão contextual ou à forma como o modelo interpreta instruções restritivas (por exemplo, responder apenas com a letra da alternativa).

Por outro lado, o modelo Perplexity apresentou desempenho inferior em comparação aos demais, com acurácia de 0,8148 e F1-score macro de 0,8269. Embora ainda apresente resultados razoáveis, a diferença significativa em relação aos outros modelos pode ser explicada por fatores como maior variabilidade nas respostas, menor aderência ao formato solicitado ou limitações na interpretação das opções em tarefas estruturadas. É possı́vel que o modelo esteja mais otimizado para respostas abertas e baseadas em recuperação de informação (web-grounded), o que pode impactar negativamente seu desempenho em tarefas objetivas e restritas.

A análise conjunta das métricas revela que, embora todos os modelos sejam capazes de resolver questões de múltipla escolha com desempenho satisfatório, há variações importantes na precisão e consistência das respostas. O uso do F1-score macro foi particularmente relevante, pois evidencia o comportamento dos modelos em relação à distribuição das classes, evitando vieses decorrentes de possı́veis desbalanceamentos. 

Em termos práticos, os resultados indicam que o modelo Gemini se destaca como a melhor opção para aplicações que exigem alta precisão em tarefas de seleção de alternativas, seguido de perto pelo GPT. O modelo Perplexity, embora útil em outros contextos, pode demandar ajustes adicionais de prompt ou pós-processamento para alcançar desempenho comparável nesse tipo de tarefa.
Por fim, destaca-se que os resultados obtidos estão diretamente relacionados ao conjunto de dados utilizado e ao formato do prompt empregado. Estudos futuros podem explorar variações de prompt engineering, aumento do conjunto de dados e avaliação em diferentes domı́nios para validar a generalização dos resultados observados.

# 6 Conclusão

Os resultados evidenciam que o desempenho dos modelos varia de acordo com a natureza da tarefa. No Dataset M1 (respostas abertas), o GPT apresentou melhor alinhamento semântico (ROUGE-L), enquanto o Perplexity demonstrou maior aderência lexical (BLEU), e o Gemini teve desempenho inferior em ambas as métricas. Isso reforça que tarefas abertas exigem avaliação baseada em similaridade textual e interpretação semântica, não apenas correspondência exata.

No Dataset M2 (múltipla escolha), o cenário se inverte: o Gemini apresentou o melhor desempenho geral (maior acurácia e F1-score), seguido pelo GPT, enquanto o Perplexity obteve resultados inferiores. Esse comportamento indica maior capacidade do Gemini em tarefas estruturadas e objetivas, nas quais a precisão da escolha é determinante. 

De forma geral, conclui-se que não há um modelo universalmente superior, mas sim modelos mais adequados a diferentes tipos de tarefa. O GPT se destaca em geração semântica, o Gemini em classificação estruturada e o Perplexity em proximidade lexical, evidenciando a importância de selecionar o modelo conforme o contexto da aplicação.

# REFERENCES
Kassab, J., Massad, C., Kapadia, V., Hajjar, A. H. E., Dahdah, J. E., Helou, M. C. E., Haroun, E., Ramchand, J., and
Harb, S. C. (2023). Abstract 16722: Performance evaluation of chatgpt 4.0 on cardiovascular clinical cases from
the usmle step 2ck and step 3 of the national board of medical examiners. Circulation, 148(Suppl 1):A16722–
A16722.
Kung, T. H., Cheatham, M., Medenilla, A., Sillos, C., De Leon, L., Elepaño, C., Madriaga, M., Aggabao, R., Diaz-
Candido, G., Maningo, J., and Tseng, V. (2023). Performance of chatgpt on usmle: Potential for ai-assisted
medical education using large language models. PLOS Digital Health, 2(2):1–12.
Manes, I., Ronn, N., Cohen, D., Ber, R. I., Horowitz-Kugler, Z., and Stanovsky, G. (2024). K-qa: A real-world medical
qa benchmark.
Pedrosa, J., Oliveira, D., Meira Jr, W., and Ribeiro, A. (2021). Automated classification of cardiology diagnoses based
on textual medical reports. Journal of Information and Data Management, 12.
Reading Turchioe, M., Volodarskiy, A., Pathak, J., Wright, D. N., Tcheng, J. E., and Slotwiner, D. (2022). Systematic
review of current natural language processing methods and applications in cardiology. Heart, 108(12):909–916.
Vishwanath, K., Stryker, J., Alyakin, A., Alber, D., and Oermann, E. (2025). Medmobile: a mobile-sized language
model with clinical capabilities. BMJ Digital Health AI, 1:e000068.
Wang, Z., Wu, J., Teitge, B., Holodinsky, J., and Drew, S. (2025). Small language models for emergency departments
decision support: A benchmark study.

# APPENDIX

Gráficos da Classificação do Dataset M1

<img width="1100" height="705" alt="image" src="https://github.com/user-attachments/assets/99ade40b-9de8-4091-b193-79219bb6c746" />

<img width="658" height="880" alt="image" src="https://github.com/user-attachments/assets/5eea6da4-9e7b-4e17-abd7-23ef1562407f" />


Gráficos da Classificação do Dataset M2

<img width="1039" height="643" alt="image" src="https://github.com/user-attachments/assets/ef5f8e79-7612-471b-bf78-ae32b793b2d9" />

<img width="811" height="859" alt="image" src="https://github.com/user-attachments/assets/e9266e31-685a-4917-b801-1ddc8f70087c" />

<img width="818" height="887" alt="image" src="https://github.com/user-attachments/assets/6e0be2e3-89ea-4da9-b87d-e1d4648a1330" />

<img width="783" height="670" alt="image" src="https://github.com/user-attachments/assets/de6fa658-eed4-451e-b4fd-8c2b81ab5b3a" />







