"""
Servidor Flask para integração do dashboard FinControl
Fornece API para:
- Salvar e carregar dados do localStorage
- Executar pipeline ETL
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import csv
from datetime import datetime
from etl_dashboard import transformar_dados, calcular_resumo, carregar_dados

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "transacoes.json")
SUMMARY_FILE = os.path.join(DATA_DIR, "resumo.json")


@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')


@app.route('/api/transacoes', methods=['GET'])
def get_transacoes():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            transacoes = json.load(f)
        return jsonify({'success': True, 'data': transacoes})
    return jsonify({'success': True, 'data': []})


@app.route('/api/transacoes', methods=['POST'])
def save_transacoes():
    try:
        dados = request.json.get('dados', [])
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        return jsonify({'success': True, 'message': 'Dados salvos com sucesso!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/etl', methods=['POST'])
def run_etl():
    try:
        dados = request.json.get('dados', [])
        
        dados_transformados = transformar_dados(dados)
        resumo = calcular_resumo(dados_transformados)
        carregar_dados(dados_transformados, resumo)
        
        return jsonify({
            'success': True,
            'message': 'ETL executado com sucesso!',
            'resumo': resumo
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/resumo', methods=['GET'])
def get_resumo():
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
            resumo = json.load(f)
        return jsonify({'success': True, 'data': resumo})
    return jsonify({'success': True, 'data': None})


if __name__ == '__main__':
    print("=" * 60)
    print("FinControl Dashboard Server")
    print("=" * 60)
    print("\nAcesse o dashboard em: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
