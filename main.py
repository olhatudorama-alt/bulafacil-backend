"""
BulaFácil — Backend (Gemini gratuito + Fuzzy Search + Admin)
=============================================================
Servidor FastAPI usando Google Gemini API (gratuito).

Hospede no Railway.app — configure a variável:
  GEMINI_API_KEY = sua_chave_aqui
  ADMIN_TOKEN = senha_secreta_para_admin  (opcional, mas recomendado)

Arquivos necessários:
  - main.py (este arquivo)
  - medicamentos.json (dicionário de sinônimos)
"""

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import google.generativeai as genai
import os
import json
import re
from rapidfuzz import process, fuzz

app = FastAPI(title="BulaFácil API", version="1.2.0")

# Permite chamadas do app Flutter e do site HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura o Gemini com a chave da variável de ambiente
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# ── Carrega dicionário de sinônimos ─────────────────────────────────
def carregar_sinonimos():
    try:
        with open("medicamentos.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

SINONIMOS = carregar_sinonimos()
# Cria lista de chaves para busca fuzzy
CHAVES_SINONIMOS = list(SINONIMOS.keys())

def limpar_texto(texto: str) -> str:
    """Remove pontuação, acentos e normaliza para busca."""
    texto = texto.lower().strip()
    texto = re.sub(r'[^\w\s]', ' ', texto)  # Remove pontuação
    texto = re.sub(r'\s+', ' ', texto).strip()  # Remove espaços extras
    return texto

def normalizar_medicamento_fuzzy(nome: str, threshold: int = 85) -> tuple[str, bool]:
    """
    Converte nome de marca para genérico usando:
    1. Busca exata no dicionário
    2. Busca fuzzy (tolerante a erros de digitação)
    
    Retorna: (nome_normalizado, foi_encontrado)
    """
    nome_limpo = limpar_texto(nome)
    
    # 1º: Busca exata (case-insensitive)
    if nome_limpo in SINONIMOS:
        return SINONIMOS[nome_limpo], True
    
    # 2º: Busca fuzzy com rapidfuzz
    resultado = process.extractOne(
        nome_limpo,
        CHAVES_SINONIMOS,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold
    )
    
    if resultado:
        chave_encontrada, score, _ = resultado
        return SINONIMOS[chave_encontrada], True
    
    # Não encontrou: retorna original
    return nome, False


# ── Modelos de dados ────────────────────────────────────────────────
class ConsultaRequest(BaseModel):
    medicamentos: List[str]
    perfil: Optional[str] = "adulto"
    verificar_interacoes: Optional[bool] = True
    verificar_uso: Optional[bool] = True
    verificar_horarios: Optional[bool] = True
    verificar_perfil: Optional[bool] = True

class NovoMedicamentoRequest(BaseModel):
    nome_marca: str
    nome_generico: str
    token_admin: str


# ── Rota de status ──────────────────────────────────────────────────
@app.get("/")
def status():
    return {
        "status": "BulaFácil backend rodando", 
        "versao": "1.2.0", 
        "motor": "Gemini + Fuzzy Search",
        "base_medicamentos": f"{len(SINONIMOS)} sinônimos carregados",
        "fuzzy_threshold": 85
    }


# ── Rota: Listar medicamentos (para debug/admin) ───────────────────
@app.get("/medicamentos")
def listar_medicamentos(q: Optional[str] = None):
    """Lista medicamentos da base. Suporta busca parcial."""
    if not q:
        return {"total": len(SINONIMOS), "medicamentos": list(SINONIMOS.keys())[:50]}
    
    q_limpa = limpar_texto(q)
    resultados = [
        {"marca": k, "generico": v} 
        for k, v in SINONIMOS.items() 
        if q_limpa in limpar_texto(k) or q_limpa in limpar_texto(v)
    ][:20]
    
    return {"busca": q, "encontrados": len(resultados), "resultados": resultados}


# ── Rota: Adicionar medicamento (painel admin simples) ─────────────
@app.post("/admin/adicionar-medicamento")
async def adicionar_medicamento(req: NovoMedicamentoRequest):
    """
    Adiciona novo sinônimo ao dicionário.
    
    🔐 Protegido por token simples (configure ADMIN_TOKEN no Railway).
    ⚠️ A alteração é temporária (em memória). Para persistir, edite medicamentos.json.
    """
    token_esperado = os.environ.get("ADMIN_TOKEN")
    
    if token_esperado and req.token_admin != token_esperado:
        raise HTTPException(status_code=403, detail="Token admin inválido")
    
    marca = limpar_texto(req.nome_marca)
    generico = req.nome_generico.strip()
    
    if not marca or not generico:
        raise HTTPException(status_code=400, detail="Preencha nome da marca e genérico")
    
    # Adiciona ao dicionário em memória
    SINONIMOS[marca] = generico
    CHAVES_SINONIMOS.append(marca)
    
    return {
        "sucesso": True,
        "mensagem": f"'{req.nome_marca}' → '{generico}' adicionado com sucesso",
        "total_agora": len(SINONIMOS),
        "observacao": "Alteração válida apenas até o próximo reinício. Para persistir, edite medicamentos.json."
    }


# ── Rota principal de consulta ──────────────────────────────────────
@app.post("/consultar")
async def consultar(req: ConsultaRequest):
    if not req.medicamentos:
        raise HTTPException(status_code=400, detail="Informe ao menos um medicamento")

    # Limpa e filtra lista
    meds_raw = [m.strip() for m in req.medicamentos if m.strip()]
    if not meds_raw:
        raise HTTPException(status_code=400, detail="Nenhum medicamento válido informado")

    # Normaliza nomes com fuzzy matching
    meds_normalizados = []
    meds_nao_reconhecidos = []
    mapeamento = {}  # Para mostrar ao usuário: "AAS" → "ácido acetilsalicílico"
    
    for med in meds_raw:
        normalizado, encontrado = normalizar_medicamento_fuzzy(med)
        meds_normalizados.append(normalizado)
        mapeamento[med] = normalizado
        if not encontrado:
            meds_nao_reconhecidos.append(med)

    opcoes = []
    if req.verificar_interacoes: opcoes.append("Verificar interações entre medicamentos")
    if req.verificar_uso:        opcoes.append("Explicar para que serve cada remédio")
    if req.verificar_horarios:   opcoes.append("Orientações de horário e como tomar")
    if req.verificar_perfil:     opcoes.append("Segurança para o perfil informado")

    # Prompt otimizado
    prompt = f"""Você é um farmacêutico clínico experiente. Analise os seguintes medicamentos para um paciente com perfil: {req.perfil}.

Medicamentos (normalizados): {', '.join(meds_normalizados)}
Nomes originais informados: {', '.join(meds_raw)}
Mapeamento aplicado: {json.dumps(mapeamento, ensure_ascii=False)}

Tarefas: {', '.join(opcoes)}

INSTRUÇÕES:
1. Se um medicamento não for reconhecido, liste em "medicamentos_nao_reconhecidos" e sugira o genérico provável.
2. Use linguagem simples, em português do Brasil, acessível para leigos.
3. Para interações, priorize as clinicamente relevantes.
4. Seja conservador: em dúvida, classifique como "atenção" e recomende consulta profissional.

Responda SOMENTE com JSON puro e válido, sem markdown, sem blocos de código:
{{
  "nivel_risco": "seguro",
  "resumo_risco": "frase curta explicando o nível geral",
  "medicamentos_nao_reconhecidos": ["nome1", "nome2"],
  "sugestao_busca": "dica para o usuário encontrar o medicamento",
  "mapeamento_aplicado": {json.dumps(mapeamento, ensure_ascii=False)},
  "interacoes": [
    {{
      "medicamentos": ["med1", "med2"],
      "gravidade": "leve",
      "descricao": "explicação clara em português simples",
      "recomendacao": "o que fazer"
    }}
  ],
  "sobre_cada_um": [
    {{
      "nome": "nome do medicamento",
      "para_que_serve": "explicação simples",
      "classe": "classe farmacológica",
      "efeitos_comuns": ["efeito1", "efeito2"]
    }}
  ],
  "como_tomar": [
    {{
      "nome": "nome do medicamento",
      "orientacoes": "horário, com/sem comida, intervalo"
    }}
  ],
  "seguranca_perfil": {{
    "perfil": "{req.perfil}",
    "avaliacoes": [
      {{
        "nome": "nome do medicamento",
        "status": "seguro",
        "observacao": "explicação específica para este perfil"
      }}
    ],
    "alerta_geral": "alerta geral para este perfil"
  }},
  "aviso_legal": "Esta ferramenta é informativa e não substitui orientação médica ou farmacêutica. Em caso de dúvida, consulte um profissional de saúde."
}}

Valores possíveis:
- nivel_risco: "seguro", "atencao" ou "perigo"
- gravidade: "leve", "moderada" ou "grave"
- status: "seguro", "cautela", "contraindicado" ou "sem dados"
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text
        
        # Limpeza robusta
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        
        # Garante aviso legal
        if "aviso_legal" not in result:
            result["aviso_legal"] = "Esta ferramenta é informativa e não substitui orientação médica ou farmacêutica. Em caso de dúvida, consulte um profissional de saúde."
        
        return result

    except json.JSONDecodeError as e:
        return {
            "nivel_risco": "sem_dados",
            "resumo_risco": "Não foi possível processar a consulta no momento.",
            "medicamentos_nao_reconhecidos": meds_nao_reconhecidos,
            "sugestao_busca": "Tente usar o nome genérico ou verifique a grafia. Ex: 'ácido acetilsalicílico' em vez de 'AAS'.",
            "mapeamento_aplicado": mapeamento,
            "interacoes": [],
            "sobre_cada_um": [],
            "como_tomar": [],
            "seguranca_perfil": {
                "perfil": req.perfil,
                "avaliacoes": [],
                "alerta_geral": "Consulte um farmacêutico para orientação personalizada."
            },
            "aviso_legal": "Esta ferramenta é informativa e não substitui orientação médica ou farmacêutica. Em caso de dúvida, consulte um profissional de saúde."
        }
    except Exception as e:
        print(f"ERRO BulaFácil: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar sua consulta. Tente novamente em instantes.")