from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path 
import fitz
import pandas as pd
import torch
import re

# 1. Carrega o modelo
print("Carregando o modelo SentenceTransformer...")
modelo = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

# 2. Definindo os requisitos 
requisitos = {
      "R1": "O documento identifica o nome da empresa autuada, sua razão social ou CNPJ.",
    "R2": "O texto descreve qual foi a infração cometida, a obrigação que foi descumprida ou a conduta irregular da empresa.",
    "R3": "O documento cita a base normativa, artigos de lei, resoluções da Anatel ou regras violadas.", 
    "R4": "O texto menciona o período da infração, como datas ou duração da conduta irregular.",
    "R5": "O documento indica se houve notificação prévia registrada, como um aviso formal ou comunicação anterior à empresa sobre a infração.", 
    "R6": "O texto apresenta evidências ou provas que sustentam a acusação, como registros de chamadas, relatórios técnicos ou outras formas de documentação.", 
    "R7": "O documento menciona as penalidades ou sanções aplicáveis, como multas, suspensão de serviços ou outras medidas punitivas previstas na legislação."
}

embeddings_requisitos = modelo.encode(list(requisitos.values()), convert_to_tensor=True)
nomes_requisitos = list(requisitos.keys())

# 3. Função para isolar a fase de instrução do documento
def isolar_fase_instrucao(texto):
    # Procura o bloco entre o início da Instrução e o início da Conclusão
    padrao = re.search(
        r'(?i)(?:3\.\s*ANÁLISE|DA ANÁLISE|DA INSTRUÇÃO|FUNDAMENTAÇÃO)'  # Início
        r'(.*?)'                                                         # Miolo (Fase de Instrução)
        r'(?:4\.\s*CONCLUSÃO|CONCLUSÃO|DA DECISÃO|DISPOSITIVO|$)',       # Fim (Corte)
        texto, 
        re.DOTALL
    )
    if padrao:
        return padrao.group(1).strip() # Retorna apenas o miolo
    return "" # Se não achar a estrutura, retorna vazio para não pontuar lixo

# 4. Configuração e Leitura do Arquivo Único
ARQUIVO_PADO = Path(r"C:\Users\Euler\Documents\Projeto Anatel\frw_semantic_frame\pdf_pados\53500.007555_2026-55\doc_15117751.pdf")
doc = fitz.open(ARQUIVO_PADO)
texto_limpo = "".join([pagina.get_text() for pagina in doc]).replace('\n', ' ')

# APLICANDO O FILTRO DA FASE DE INSTRUÇÃO AQUI
texto_instrucao = isolar_fase_instrucao(texto_limpo)

if not texto_instrucao:
    print("\nNenhuma fase de instrução foi localizada neste documento.")
else:
    # 5. Processamento e Embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    text_chunks = text_splitter.split_text(texto_instrucao)

    print(f"\nGerando embeddings para {len(text_chunks)} trechos de instrução do documento...")
    chunks = modelo.encode(text_chunks, convert_to_tensor=True)

    THRESHOLD = 0.70 
    resultados_documento = [] # Nome da lista corrigido

    # Matriz de similaridade
    cosine_scores = util.cos_sim(chunks, embeddings_requisitos)

    # 6. Avaliação de Conformidade
    for req_idx, requisito in enumerate(nomes_requisitos):
        scores_requisito = cosine_scores[:, req_idx]
                
        melhor_score = torch.max(scores_requisito).item()
        melhor_chunk_idx = torch.argmax(scores_requisito).item()
                
        conforme = "Sim" if melhor_score >= THRESHOLD else "Não"
                
        resultados_documento.append({
            "Documento": ARQUIVO_PADO.name,
            "Requisito": requisito,
            "Score Maximo": round(melhor_score, 4),
            "Status": conforme,
            "Trecho Mais Relevante": text_chunks[melhor_chunk_idx][:150] + "..." 
        })

    # 7. Salvar Resultados
    if resultados_documento:
        df_resultados = pd.DataFrame(resultados_documento)
        print("\n--- Resumo das Pontuações (Apenas Instrução) ---")
        print(df_resultados[['Requisito', 'Status', 'Score Maximo']])

        saida_csv = Path(r"C:\Users\Euler\Documents\Projeto Anatel\frw_semantic_frame\requisitos_instrucao.csv")
        saida_csv.parent.mkdir(parents=True, exist_ok=True)
        df_resultados.to_csv(saida_csv, index=False, encoding='utf-8-sig')
        print(f"\nSalvo com sucesso em: {saida_csv}") 
