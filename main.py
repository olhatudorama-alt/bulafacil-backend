"""
BulaFácil — Backend (Gemini gratuito)
=======================================
Servidor FastAPI usando Google Gemini API (gratuito).

Hospede no Railway.app — configure a variável:
  GEMINI_API_KEY = sua_chave_aqui
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
import json

app = FastAPI(title="BulaFácil API", version="1.0.0")

# Permite chamadas do app Flutter e do site HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura o Gemini com a chave da variável de ambiente
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # Rápido e gratuito


# ── Modelos de dados ────────────────────────────────────────────────
class ConsultaRequest(BaseModel):
    medicamentos: List[str]
    perfil: Optional[str] = "adulto"
    verificar_interacoes: Optional[bool] = True
    verificar_uso: Optional[bool] = True
    verificar_horarios: Optional[bool] = True
    verificar_perfil: Optional[bool] = True


# ── Rota de status ──────────────────────────────────────────────────
@app.get("/")
def status():
    return {"status": "BulaFácil backend rodando", "versao": "1.0.0", "motor": "Gemini"}


# ── Rota principal de consulta ──────────────────────────────────────
@app.post("/consultar")
async def consultar(req: ConsultaRequest):
    if not req.medicamentos:
        raise HTTPException(status_code=400, detail="Informe ao menos um medicamento")

    meds = [m.strip() for m in req.medicamentos if m.strip()]
    if not meds:
        raise HTTPException(status_code=400, detail="Nenhum medicamento válido informado")

    opcoes = []
    if req.verificar_interacoes: opcoes.append("Verificar interações entre medicamentos")
    if req.verificar_uso:        opcoes.append("Explicar para que serve cada remédio")
    if req.verificar_horarios:   opcoes.append("Orientações de horário e como tomar")
    if req.verificar_perfil:     opcoes.append("Segurança para o perfil informado")

    prompt = f"""Você é um farmacêutico clínico experiente. Analise os seguintes medicamentos para um paciente com perfil: {req.perfil}.

Medicamentos: {', '.join(meds)}

Tarefas: {', '.join(opcoes)}

IMPORTANTE: Responda SOMENTE com JSON puro e válido, sem nenhum texto antes ou depois, sem markdown, sem blocos de código:
{{
  "nivel_risco": "seguro",
  "resumo_risco": "frase curta explicando o nível geral",
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
  }}
}}

Valores possíveis:
- nivel_risco: "seguro", "atencao" ou "perigo"
- gravidade: "leve", "moderada" ou "grave"
- status: "seguro", "cautela", "contraindicado" ou "sem dados"
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text
        # Remove possíveis marcações de código que o Gemini às vezes inclui
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erro ao processar resposta da IA")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
