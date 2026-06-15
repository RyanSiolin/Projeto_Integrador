# FinControl Dashboard - Guia de Uso

## Visão Geral

Este dashboard é uma aplicação web funcional para controle financeiro pessoal, com integração a um backend em Python usando Flask.

## Funcionalidades

1. **Dashboard Principal**: Visualização de saldo, receitas, despesas e gráficos
2. **Lançamentos**: Adicionar novas transações (receitas/despesas)
3. **Transações**: Listar e gerenciar todas as transações
4. **Relatórios**: Gráficos detalhados de evolução financeira
5. **Educação Financeira**: Conteúdos e dicas
6. **Configurações**: Exportar/importar dados e integração com backend

## Como Usar

### Opção 1: Aplicativo Standalone (sem backend)

1. Abra o arquivo `index.html` no navegador
2. O aplicativo usa `localStorage` para salvar os dados no navegador
3. Todas as funcionalidades estão disponíveis, exceto a integração com backend

### Opção 2: Com Integração Backend (Recomendado)

1. Instale as dependências Python:
```bash
cd dashboard
pip install -r requirements.txt
```

2. Execute o servidor Flask:
```bash
python server.py
```

3. Acesse o dashboard no navegador: `http://localhost:5000`

## Funcionalidade do ETL (Extract, Transform, Load)

O arquivo `etl_dashboard.py` trabalha em conjunto com o servidor Flask para:

1. **Extrair**: Receber dados do frontend (localStorage) via API
2. **Transformar**: Limpar e validar os dados
3. **Carregar**: Salvar os dados processados em arquivos JSON e CSV na pasta `data/`

## Fluxo de Integração

```
Frontend (Navegador)
    ↓
API REST Flask (/api/etl)
    ↓
Pipeline ETL Python
    ↓
Arquivos JSON/CSV na pasta data/
```

## Arquivos Gerados

- `data/transacoes.json`: Dados brutos das transações
- `data/resumo.json`: Resumo financeiro calculado
- `data/transacoes.csv`: Dados em formato CSV para planilhas

## Estrutura de Arquivos

```
dashboard/
├── index.html              # Página inicial (redireciona para dashboard)
├── dashboard.html          # Dashboard principal
├── lancamento.html         # Formulário de lançamentos
├── transacoes.html         # Lista de transações
├── relatorios.html         # Relatórios e gráficos
├── educativo.html          # Conteúdo educacional
├── configuracoes.html      # Configurações e exportação
├── style.css               # Estilos do dashboard
├── app.js                  # Lógica do frontend
├── server.py               # Servidor Flask (backend)
├── etl_dashboard.py        # Pipeline ETL
├── requirements.txt        # Dependências Python
└── data/                   # Pasta para dados processados
```

## Exportar e Importar Dados

### Exportar
1. Vá para a página **Configurações**
2. Clique em **Exportar JSON** ou **Exportar CSV**
3. O arquivo será baixado automaticamente

### Importar
1. Vá para a página **Configurações**
2. Selecione um arquivo JSON exportado anteriormente
3. Clique em **Importar**
4. Os dados serão restaurados

## Sincronizar com Backend

### Sincronização Automática (Padrão)
1. Certifique-se que o servidor Flask está rodando
2. Vá para a página **Configurações**
3. Verifique se a URL do servidor está correta (`http://localhost:5000`)
4. **Pronto!** A sincronização é automática:
   - Sempre que você adicionar ou excluir uma transação, os dados são enviados para o backend
   - Quando o dashboard abre, ele tenta carregar os dados do servidor
   - Se o servidor estiver offline, o aplicativo continua funcionando com dados locais

### Alterar URL do Servidor
1. Vá para a página **Configurações**
2. Altere o campo "URL do Servidor"
3. A nova URL é salva automaticamente no navegador

## Tecnologias Utilizadas

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Gráficos**: Chart.js
- **Backend**: Flask (Python)
- **Armazenamento**: localStorage (navegador) + arquivos JSON/CSV
