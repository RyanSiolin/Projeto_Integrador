"""
ETL para integração de dados do FinControl
Similar ao script do projeto principal
"""

import os
import json
import csv
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRANSACTIONS_JSON = os.path.join(OUTPUT_DIR, "transacoes.json")
TRANSACTIONS_CSV = os.path.join(OUTPUT_DIR, "transacoes.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "resumo.json")


def extrair_dados_localstorage(json_file="localstorage_backup.json"):
    """
    Extrai dados de um arquivo JSON (simula backup do localStorage do navegador)
    """
    backup_path = os.path.join(SCRIPT_DIR, json_file)
    if os.path.exists(backup_path):
        with open(backup_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def transformar_dados(dados):
    """
    Limpa e transforma os dados
    """
    if not dados:
        return []

    dados_limpos = []
    for item in dados:
        item_limpo = {
            "id": item.get("id"),
            "tipo": item.get("tipo"),
            "descricao": item.get("descricao", "").strip(),
            "valor": float(item.get("valor", 0)),
            "data": item.get("data"),
            "categoria": item.get("categoria"),
            "observacoes": item.get("observacoes", "").strip()
        }
        dados_limpos.append(item_limpo)

    return dados_limpos


def calcular_resumo(dados):
    """
    Calcula métricas de resumo
    """
    total_receitas = sum(t["valor"] for t in dados if t["tipo"] == "receita")
    total_despesas = sum(t["valor"] for t in dados if t["tipo"] == "despesa")
    saldo = total_receitas - total_despesas

    # Agrupar por categoria
    gastos_por_categoria = {}
    for t in dados:
        if t["tipo"] == "despesa":
            cat = t["categoria"]
            gastos_por_categoria[cat] = gastos_por_categoria.get(cat, 0) + t["valor"]

    return {
        "total_receitas": total_receitas,
        "total_despesas": total_despesas,
        "saldo_atual": saldo,
        "gastos_por_categoria": gastos_por_categoria,
        "total_transacoes": len(dados),
        "data_atualizacao": datetime.now().isoformat()
    }


def carregar_dados(dados, resumo):
    """
    Salva os dados processados em JSON e CSV
    """
    with open(TRANSACTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    if dados:
        with open(TRANSACTIONS_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dados[0].keys())
            writer.writeheader()
            writer.writerows(dados)


def pipeline_etl():
    """
    Executa o pipeline completo ETL
    """
    print("=" * 60)
    print("Pipeline ETL - FinControl Dashboard")
    print("=" * 60)

    print("\n[1/4] Extraindo dados...")
    dados_brutos = extrair_dados_localstorage()

    if not dados_brutos:
        print("⚠️ Nenhum dado encontrado. Usando dados de exemplo.")
        dados_brutos = [
            {"id": 1, "tipo": "receita", "descricao": "Salário", "valor": 4200, "data": "2025-04-25", "categoria": "Emprego", "observacoes": ""},
            {"id": 2, "tipo": "receita", "descricao": "Freela Design", "valor": 650, "data": "2025-04-22", "categoria": "Renda Extra", "observacoes": ""},
            {"id": 3, "tipo": "despesa", "descricao": "Aluguel", "valor": 900, "data": "2025-04-24", "categoria": "Moradia", "observacoes": ""}
        ]

    print(f"✅ {len(dados_brutos)} registros extraídos")

    print("\n[2/4] Limpando e transformando dados...")
    dados_transformados = transformar_dados(dados_brutos)
    print("✅ Dados transformados com sucesso")

    print("\n[3/4] Calculando métricas...")
    resumo = calcular_resumo(dados_transformados)
    print("✅ Métricas calculadas")

    print("\n[4/4] Salvando dados processados...")
    carregar_dados(dados_transformados, resumo)
    print(f"✅ Dados salvos em: {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("Resumo:")
    print(f"  Total de transações: {resumo['total_transacoes']}")
    print(f"  Total receitas: R$ {resumo['total_receitas']:.2f}")
    print(f"  Total despesas: R$ {resumo['total_despesas']:.2f}")
    print(f"  Saldo atual: R$ {resumo['saldo_atual']:.2f}")
    print("=" * 60)
    print("\n✅ Pipeline concluído com sucesso!")


if __name__ == "__main__":
    pipeline_etl()
