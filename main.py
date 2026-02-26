"""
BulaFácil — Backend
====================
Servidor FastAPI que recebe consultas de medicamentos
e retorna análises usando a API da Anthropic.

Hospede no Railway.app — configure a variável:
  ANTHROPIC_API_KEY = sua_chave_aqui
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import anthropic
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

# Cliente Anthropic — lê a chave da variável de ambiente do Railway
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


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
    return {"status": "BulaFácil backend rodando", "versao": "1.0.0"}


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

Responda SOMENTE em JSON válido, sem texto fora do JSON:
{{
  "nivel_risco": "seguro" | "atencao" | "perigo",
  "resumo_risco": "frase curta explicando o nível geral",
  "interacoes": [
    {{
      "medicamentos": ["med1", "med2"],
      "gravidade": "leve" | "moderada" | "grave",
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
      "orientacoes": "horário, com/sem comida, intervalo, duração típica"
    }}
  ],
  "seguranca_perfil": {{
    "perfil": "{req.perfil}",
    "avaliacoes": [
      {{
        "nome": "nome do medicamento",
        "status": "seguro" | "cautela" | "contraindicado" | "sem dados",
        "observacao": "explicação específica para este perfil"
      }}
    ],
    "alerta_geral": "alerta geral para este perfil com estes medicamentos"
  }}
}}"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Haiku: mais barato, rápido e suficiente
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erro ao processar resposta da IA")
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=500, detail="Chave de API inválida — configure ANTHROPIC_API_KEY no Railway")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
