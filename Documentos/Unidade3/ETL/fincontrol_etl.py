"""
============================================================
  Módulos:
    01  extrair_dados()        — Extração & Ingestão
    02  tratar_limpar_dados()  — Inspeção & Limpeza
    03  transformar_dados()    — Transformação & Features
    04  estruturar_e_exportar()— Modelagem, Validação & Carga
============================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import date

# ─────────────────────────────────────────────────────────
#  CONFIGURAÇÕES DE CAMINHO
#  Ajuste os caminhos abaixo para o seu ambiente
# ─────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PASTA_OUTPUT = os.path.join(SCRIPT_DIR, "database")
os.makedirs(PASTA_OUTPUT, exist_ok=True)

ARQUIVO_FORMS = os.path.join(SCRIPT_DIR, "respostas_pesquisa.csv")
ARQUIVO_BCB   = os.path.join(SCRIPT_DIR, "planilha_202506.csv")

print(f"📁 Diretório de saída: {PASTA_OUTPUT}\n")


# ╔══════════════════════════════════════════════════════╗
# ║  MÓDULO 01 — EXTRAÇÃO & INGESTÃO                    ║
# ╚══════════════════════════════════════════════════════╝

def extrair_dados_form(nome_arquivo: str) -> pd.DataFrame:
    """
    Carrega o CSV do Google Forms (respostas da pesquisa acadêmica).
    Tenta múltiplos encodings para garantir leitura correta.
    Retorna DataFrame bruto ou DataFrame vazio em caso de erro.
    """
    print("=" * 60)
    print("MÓDULO 01 — EXTRAÇÃO & INGESTÃO")
    print("=" * 60)

    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            df_bruto = pd.read_csv(nome_arquivo, sep=",", encoding=enc)
            print(f"✅ [Forms] Lido com sep=',' enc='{enc}'")
            print(f"   → {len(df_bruto)} respostas encontradas no arquivo.")
            print(f"   → {df_bruto.shape[1]} colunas detectadas.\n")
            return df_bruto
        except FileNotFoundError:
            print(f"\n❌ ERRO DE EXTRAÇÃO: Arquivo '{nome_arquivo}' não encontrado.")
            print("   Verifique se o caminho está correto (maiúsculas/minúsculas).")
            return pd.DataFrame()
        except Exception:
            continue

    print("❌ ERRO DE LEITURA: Nenhum encoding funcionou para o arquivo Forms.")
    return pd.DataFrame()


def extrair_dados_bcb(nome_arquivo: str) -> pd.DataFrame:
    """
    Carrega o CSV do Banco Central do Brasil (dados de crédito).
    O arquivo usa ';' como separador e ',' como decimal.
    Retorna DataFrame bruto ou DataFrame vazio em caso de erro.
    """
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            df_bruto = pd.read_csv(
                nome_arquivo,
                sep=";",
                decimal=",",
                encoding=enc,
                low_memory=False,
            )
            print(f"✅ [BCB] Lido com sep=';' enc='{enc}'")
            print(f"   → {len(df_bruto):,} registros encontrados.")
            print(f"   → {df_bruto.shape[1]} colunas detectadas.\n")
            return df_bruto
        except FileNotFoundError:
            print(f"\n❌ ERRO DE EXTRAÇÃO: Arquivo '{nome_arquivo}' não encontrado.")
            return pd.DataFrame()
        except Exception:
            continue

    print("❌ ERRO DE LEITURA: Nenhum encoding funcionou para o arquivo BCB.")
    return pd.DataFrame()


# ╔══════════════════════════════════════════════════════╗
# ║  MÓDULO 02 — INSPEÇÃO & LIMPEZA                     ║
# ╚══════════════════════════════════════════════════════╝

def tratar_limpar_forms(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma df_bruto da pesquisa em df_limpo:
      - Renomeia colunas para snake_case
      - Remove duplicatas
      - Trata nulos de colunas condicionais
      - Normaliza strings (strip, title/lower)
      - Converte timestamp
    """
    print("=" * 60)
    print("MÓDULO 02 — INSPEÇÃO & LIMPEZA (Forms)")
    print("=" * 60)

    if df_bruto.empty:
        print("⚠️  DataFrame vazio — limpeza ignorada.\n")
        return df_bruto

    df = df_bruto.copy()

    # ── Normalizar nomes de colunas ──────────────────────
    df.columns = (
        df.columns
        .astype(str)
        .str.replace(r"\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.replace(r"\W+", "_", regex=True)
        .str.strip("_")
    )
    print(f"\n✅ Colunas normalizadas ({len(df.columns)} colunas)")

    # ── Renomeação explícita das colunas principais ──────
    mapa_colunas = {
        "carimbo_de_data_hora"                         : "timestamp",
        "qual_sua_faixa_etaria"                        : "faixa_etaria",
        "qual_sua_situacao_profissional"                : "situacao_profissional",
        "qual_sua_renda_mensal_aproximada"              : "renda_mensal",
        "voce_costuma_controlar_seus_gastos_mensais"    : "controla_gastos",
        "como_voce_controla_seus_gastos"                : "como_controla_gastos",
        "com_que_frequencia_voce_registra_seus_gastos"  : "freq_registro_gastos",
        "voce_costuma_ultrapassar_seu_orcamento_mensal" : "ultrapassa_orcamento",
        "voce_costuma_planejar_seus_gastos_antes_de_receber_seu_dinheiro": "planeja_gastos",
        "voce_possui_uma_reserva_de_emergencia"         : "tem_reserva_emergencia",
        "faz_algum_tipo_de_investimento"                : "faz_investimento",
        "se_sim_qual"                                   : "tipo_investimento",
        "voce_considera_que_possui_educacao_financeira_suficiente": "nota_educ_financeira",
        "voce_tem_interesse_em_aprender_mais_sobre_educacao_financeira": "interesse_aprender",
        "voce_ja_fez_algum_curso_ou_consumiu_conteudo_sobre_financas": "fez_curso_financas",
        "onde_voce_costuma_buscar_informacoes_sobre_financas_marque_todas_que_se_aplicam": "fontes_informacao",
        "voce_teve_algum_tipo_de_educacao_financeira_na_escola": "educ_financeira_escola",
        "sente_falta_de_um_conhecimento_financeiro_mais_amplo_no_dia_a_dia": "sente_falta_educ",
        "voce_acredita_que_ferramentas_digitais_podem_melhorar_sua_educacao_financeira": "acredita_ferramentas_digitais",
        "voce_entende_conceitos_basicos_de_financas_ex_juros_orcamento_divida_imposto_de_renda_inadimplencia": "entende_conceitos_basicos",
        "qual_desses_temas_voce_considera_mais_dificil_marque_todas_que_se_aplicam": "temas_dificeis",
        "o_que_mais_dificulta_seu_aprendizado_sobre_financas": "dificuldade_aprendizado",
        "voce_ja_utilizou_algum_aplicativo_de_controle_financeiro": "usou_app_financeiro",
        "o_que_voce_mais_gostaria_de_ver_em_um_sistema_de_controle_financeiro": "desejo_sistema",
        "qual_formato_voce_prefere_para_aprender_sobre_financas_marque_todas_que_se_aplicam": "formato_aprendizado",
        "qual_sua_maior_dificuldade_em_relacao_ao_dinheiro_marque_todas_que_se_aplicam": "maior_dificuldade_dinheiro",
        "voce_acha_que_um_dashboard_simples_ajudaria_no_controle_financeiro": "dashboard_ajudaria",
        "que_tipo_de_ajuda_voce_considera_mais_util": "tipo_ajuda_util",
        "atualmente_voce_se_encontra_em_situacao_de_inadimplencia_com_contas_ou_dividas_em_atraso": "esta_inadimplente",
        "quais_tipos_de_pendencias_financeira_voce_possui_atualmente_marque_todas_que_se_aplicam": "tipos_pendencias",
        "com_que_frequencia_voce_enfrenta_dificuldades_para_pagar_suas_contas_no_prazo": "freq_dificuldade_pagamento",
        "voce_teria_interesse_em_usar_um_app_site_de_controle_financeiro_que_alem_de_organizar_seus_gastos_tambem_ofere": "interesse_app_fincontrol",
    }
    # Aplica apenas as colunas que existem no df
    mapa_valido = {k: v for k, v in mapa_colunas.items() if k in df.columns}
    df = df.rename(columns=mapa_valido)
    print(f"✅ {len(mapa_valido)} colunas renomeadas para snake_case")

    # ── Duplicatas ───────────────────────────────────────
    antes = len(df)
    df = df.drop_duplicates()
    print(f"✅ Duplicatas removidas: {antes - len(df)} | Restantes: {len(df)}")

    # ── Nulos: colunas condicionais → 'Não se aplica' ───
    colunas_condicionais = [
        "tipo_investimento",
        "dificuldade_aprendizado",
        "desejo_sistema",
    ]
    # Colunas de experiência (nomes parcialmente normalizados)
    colunas_cond_parcial = [
        c for c in df.columns
        if "se_sim" in c or "experiencia" in c
    ]
    for col in colunas_condicionais + colunas_cond_parcial:
        if col in df.columns:
            df[col] = df[col].fillna("Não se aplica")

    nulos_restantes = df.isnull().sum().sum()
    print(f"✅ Nulos em colunas condicionais tratados | Nulos restantes: {nulos_restantes}")

    # ── Normalizar strings categóricas ──────────────────
    colunas_title = [
        "faixa_etaria", "situacao_profissional", "renda_mensal",
        "tem_reserva_emergencia", "faz_investimento", "esta_inadimplente",
        "usou_app_financeiro", "interesse_aprender",
    ]
    colunas_strip = [
        "fontes_informacao", "temas_dificeis", "maior_dificuldade_dinheiro",
        "tipos_pendencias", "formato_aprendizado",
    ]
    for col in colunas_title:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    for col in colunas_strip:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ── Converter timestamp ──────────────────────────────
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], dayfirst=True, errors="coerce"
        )
        print(f"✅ Coluna 'timestamp' convertida para datetime")

    print(f"\n📋 df_limpo_forms pronto: {df.shape[0]} linhas × {df.shape[1]} colunas\n")
    return df


def tratar_limpar_bcb(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma df_bruto do BCB em df_limpo:
      - Remove duplicatas
      - Converte colunas monetárias de string para float
      - Trata '<= 15' em numero_de_operacoes
      - Converte data_base para datetime
    """
    print("=" * 60)
    print("MÓDULO 02 — INSPEÇÃO & LIMPEZA (BCB)")
    print("=" * 60)

    if df_bruto.empty:
        print("⚠️  DataFrame vazio — limpeza ignorada.\n")
        return df_bruto

    df = df_bruto.copy()

    # ── Duplicatas ───────────────────────────────────────
    antes = len(df)
    df = df.drop_duplicates()
    print(f"✅ Duplicatas removidas: {antes - len(df)} | Restantes: {len(df):,}")

    # ── Colunas monetárias → float ───────────────────────
    colunas_monetarias = [
        "a_vencer_ate_90_dias",
        "a_vencer_de_91_ate_360_dias",
        "a_vencer_de_361_ate_1080_dias",
        "a_vencer_de_1081_ate_1800_dias",
        "a_vencer_de_1801_ate_5400_dias",
        "a_vencer_acima_de_5400_dias",
        "vencido_acima_de_15_dias",
        "carteira_ativa",
        "carteira_inadimplida_arrastada",
        "ativo_problematico",
    ]
    for col in colunas_monetarias:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    print(f"✅ {len(colunas_monetarias)} colunas monetárias convertidas para float")

    # ── numero_de_operacoes: '<= 15' → 0 ────────────────
    if "numero_de_operacoes" in df.columns:
        df["numero_de_operacoes"] = (
            df["numero_de_operacoes"]
            .astype(str)
            .str.strip()
            .replace({"<= 15": "0", "<= 15 ": "0"})
        )
        df["numero_de_operacoes"] = pd.to_numeric(
            df["numero_de_operacoes"], errors="coerce"
        ).fillna(0)
        print("✅ Coluna 'numero_de_operacoes' tratada (<= 15 → 0)")

    # ── data_base → datetime ─────────────────────────────
    if "data_base" in df.columns:
        df["data_base"] = pd.to_datetime(df["data_base"], errors="coerce")
        print("✅ Coluna 'data_base' convertida para datetime")

    # ── Nulos residuais ──────────────────────────────────
    nulos = df.isnull().sum()
    nulos_sig = nulos[nulos > 0]
    if not nulos_sig.empty:
        print(f"\n⚠️  Nulos residuais:\n{nulos_sig}")

    print(f"\n📋 df_limpo_bcb pronto: {df.shape[0]:,} linhas × {df.shape[1]} colunas\n")
    return df


# ╔══════════════════════════════════════════════════════╗
# ║  MÓDULO 03 — TRANSFORMAÇÃO & FEATURES               ║
# ╚══════════════════════════════════════════════════════╝

def transformar_dados(df_forms: pd.DataFrame, df_bcb: pd.DataFrame) -> dict:
    """
    Gera as tabelas modulares (Princípio da Responsabilidade Única):
      - tabela_perfil_pessoal    : perfil demográfico e financeiro
      - tabela_comportamento     : hábitos de controle financeiro
      - tabela_educacao          : nível e interesse em educação financeira
      - tabela_inadimplencia     : dados de inadimplência por respondente
      - tabela_credito_por_uf    : carteira de crédito agregada por estado (BCB)
      - tabela_perfil_x_credito  : cruzamento entre perfil e inadimplência BCB
    """
    print("=" * 60)
    print("MÓDULO 03 — TRANSFORMAÇÃO & FEATURES")
    print("=" * 60)
    tabelas = {}

    # ── Forms: Tabela 1 — Perfil Pessoal ────────────────
    colunas_perfil = [
        "timestamp", "faixa_etaria", "situacao_profissional",
        "renda_mensal", "tem_reserva_emergencia", "faz_investimento",
        "tipo_investimento",
    ]
    tabelas["tabela_perfil_pessoal"] = _selecionar_colunas(df_forms, colunas_perfil)
    print(f"✅ tabela_perfil_pessoal: {tabelas['tabela_perfil_pessoal'].shape}")

    # ── Forms: Tabela 2 — Comportamento Financeiro ──────
    colunas_comportamento = [
        "controla_gastos", "como_controla_gastos", "freq_registro_gastos",
        "ultrapassa_orcamento", "planeja_gastos", "maior_dificuldade_dinheiro",
        "freq_dificuldade_pagamento",
    ]
    tabelas["tabela_comportamento"] = _selecionar_colunas(df_forms, colunas_comportamento)
    print(f"✅ tabela_comportamento: {tabelas['tabela_comportamento'].shape}")

    # ── Forms: Tabela 3 — Educação Financeira ───────────
    colunas_educacao = [
        "nota_educ_financeira", "interesse_aprender", "fez_curso_financas",
        "fontes_informacao", "educ_financeira_escola", "sente_falta_educ",
        "acredita_ferramentas_digitais", "entende_conceitos_basicos",
        "temas_dificeis", "dificuldade_aprendizado", "formato_aprendizado",
        "dashboard_ajudaria", "tipo_ajuda_util", "interesse_app_fincontrol",
    ]
    tabelas["tabela_educacao"] = _selecionar_colunas(df_forms, colunas_educacao)
    print(f"✅ tabela_educacao: {tabelas['tabela_educacao'].shape}")

    # ── Forms: Tabela 4 — Inadimplência (respondentes) ──
    colunas_inadimplencia = [
        "faixa_etaria", "renda_mensal", "esta_inadimplente",
        "tipos_pendencias", "freq_dificuldade_pagamento",
    ]
    tabelas["tabela_inadimplencia_forms"] = _selecionar_colunas(df_forms, colunas_inadimplencia)
    print(f"✅ tabela_inadimplencia_forms: {tabelas['tabela_inadimplencia_forms'].shape}")

    # ── BCB: Feature — Carteira por UF ──────────────────
    if not df_bcb.empty:
        credito_uf = df_bcb.groupby("uf").agg(
            total_operacoes            = ("numero_de_operacoes", "sum"),
            carteira_ativa_total       = ("carteira_ativa", "sum"),
            inadimplida_total          = ("carteira_inadimplida_arrastada", "sum"),
            ativo_problematico_total   = ("ativo_problematico", "sum"),
            vencido_acima_15_total     = ("vencido_acima_de_15_dias", "sum"),
        ).reset_index()

        credito_uf["taxa_inadimplencia_pct"] = (
            credito_uf["inadimplida_total"]
            / credito_uf["carteira_ativa_total"].replace(0, np.nan)
            * 100
        ).round(2)

        credito_uf["taxa_ativo_problematico_pct"] = (
            credito_uf["ativo_problematico_total"]
            / credito_uf["carteira_ativa_total"].replace(0, np.nan)
            * 100
        ).round(2)

        tabelas["tabela_credito_por_uf"] = credito_uf
        print(f"✅ tabela_credito_por_uf: {credito_uf.shape}")

        # ── BCB: Feature — Carteira por Modalidade ──────
        credito_modalidade = df_bcb.groupby(["modalidade", "porte"]).agg(
            carteira_ativa_total     = ("carteira_ativa", "sum"),
            inadimplida_total        = ("carteira_inadimplida_arrastada", "sum"),
        ).reset_index()
        credito_modalidade["taxa_inadimplencia_pct"] = (
            credito_modalidade["inadimplida_total"]
            / credito_modalidade["carteira_ativa_total"].replace(0, np.nan)
            * 100
        ).round(2)
        tabelas["tabela_credito_por_modalidade"] = credito_modalidade
        print(f"✅ tabela_credito_por_modalidade: {credito_modalidade.shape}")

    # ── Feature Cruzada — Perfil survey x Inadimplência BCB ──
    if not df_forms.empty:
        perfil_renda = df_forms.groupby("renda_mensal").agg(
            total_respondentes       = ("faixa_etaria", "count"),
            pct_inadimplentes        = ("esta_inadimplente",
                                        lambda x: round((x.str.title() == "Sim").mean() * 100, 1)),
            pct_tem_reserva          = ("tem_reserva_emergencia",
                                        lambda x: round((x.str.title() == "Sim").mean() * 100, 1)),
            pct_faz_investimento     = ("faz_investimento",
                                        lambda x: round((x.str.title() == "Sim").mean() * 100, 1)),
            pct_usou_app             = ("usou_app_financeiro",
                                        lambda x: round((x.str.title() == "Sim").mean() * 100, 1)),
            media_nota_educ          = ("nota_educ_financeira", "mean"),
        ).reset_index()
        perfil_renda["media_nota_educ"] = perfil_renda["media_nota_educ"].round(2)
        tabelas["tabela_perfil_por_renda"] = perfil_renda
        print(f"✅ tabela_perfil_por_renda: {perfil_renda.shape}")

    print()
    return tabelas


def _selecionar_colunas(df: pd.DataFrame, lista: list) -> pd.DataFrame:
    """Helper: retorna apenas as colunas existentes no df."""
    existentes = [c for c in lista if c in df.columns]
    faltantes  = [c for c in lista if c not in df.columns]
    if faltantes:
        print(f"   ⚠️  Colunas ausentes (ignoradas): {faltantes}")
    if not existentes:
        return pd.DataFrame()
    return df[existentes].copy()


# ╔══════════════════════════════════════════════════════╗
# ║  MÓDULO 04 — MODELAGEM, VALIDAÇÃO & CARGA           ║
# ╚══════════════════════════════════════════════════════╝

def estruturar_e_exportar(
    df_forms_limpo: pd.DataFrame,
    df_bcb_limpo: pd.DataFrame,
    tabelas: dict,
) -> None:
    """
    Valida os DataFrames finais e exporta todos os CSVs
    para a pasta 'database/', prontos para análise e BI.
    """
    print("=" * 60)
    print("MÓDULO 04 — VALIDAÇÃO & EXPORTAÇÃO")
    print("=" * 60)

    # ── Validações ───────────────────────────────────────
    assert not df_forms_limpo.empty,   "❌ FALHA: df_forms_limpo está vazio!"
    assert not df_bcb_limpo.empty,     "❌ FALHA: df_bcb_limpo está vazio!"
    assert len(tabelas) > 0,           "❌ FALHA: Nenhuma tabela gerada!"

    # Verifica colunas críticas
    for col in ["renda_mensal", "esta_inadimplente", "nota_educ_financeira"]:
        assert col in df_forms_limpo.columns, f"❌ Coluna '{col}' ausente no forms limpo!"
    for col in ["carteira_ativa", "uf", "data_base"]:
        assert col in df_bcb_limpo.columns, f"❌ Coluna '{col}' ausente no BCB limpo!"

    print("✅ Todas as validações aprovadas\n")

    # ── Exportar bases limpas ────────────────────────────
    arquivos_salvos = []

    caminho = os.path.join(PASTA_OUTPUT, "forms_limpo.csv")
    df_forms_limpo.to_csv(caminho, index=False, sep=";", encoding="utf-8-sig")
    arquivos_salvos.append(("forms_limpo.csv", len(df_forms_limpo)))

    caminho = os.path.join(PASTA_OUTPUT, f"bcb_limpo_{date.today()}.csv")
    df_bcb_limpo.to_csv(caminho, index=False, sep=";", encoding="utf-8-sig")
    arquivos_salvos.append((f"bcb_limpo_{date.today()}.csv", len(df_bcb_limpo)))

    # ── Exportar tabelas modulares ───────────────────────
    for nome, df_salvar in tabelas.items():
        if df_salvar is None or df_salvar.empty:
            print(f"   ⚠️  {nome}.csv (vazio) — não será salvo.")
            continue
        caminho = os.path.join(PASTA_OUTPUT, f"{nome}.csv")
        df_salvar.to_csv(caminho, index=False, sep=";", encoding="utf-8-sig")
        arquivos_salvos.append((f"{nome}.csv", len(df_salvar)))

    # ── Relatório final ──────────────────────────────────
    print("\n--- BASE DE DADOS ESTRUTURADA CONCLUÍDA ---\n")
    print(f"{'Arquivo':<45} {'Linhas':>10}")
    print("-" * 57)
    for nome, qtd in arquivos_salvos:
        print(f"{nome:<45} {qtd:>10,}")
    print("-" * 57)
    print(f"\n📂 Todos os arquivos salvos em: {PASTA_OUTPUT}")
    print("\n✅ Finalizado.")


# ╔══════════════════════════════════════════════════════╗
# ║  EXECUÇÃO PRINCIPAL                                 ║
# ╚══════════════════════════════════════════════════════╝

if __name__ == "__main__":

    # ── Módulo 01: Extração ──────────────────────────────
    df_bruto_forms = extrair_dados_form(ARQUIVO_FORMS)
    df_bruto_bcb   = extrair_dados_bcb(ARQUIVO_BCB)

    if df_bruto_forms.empty or df_bruto_bcb.empty:
        print("\n❌ FALHA NA EXTRAÇÃO. Verifique os arquivos e tente novamente.")
        sys.exit(1)

    # ── Módulo 02: Limpeza ───────────────────────────────
    df_forms_limpo = tratar_limpar_forms(df_bruto_forms)
    df_bcb_limpo   = tratar_limpar_bcb(df_bruto_bcb)

    # ── Módulo 03: Transformação ─────────────────────────
    tabelas = transformar_dados(df_forms_limpo, df_bcb_limpo)

    # ── Módulo 04: Exportação ────────────────────────────
    estruturar_e_exportar(df_forms_limpo, df_bcb_limpo, tabelas)
