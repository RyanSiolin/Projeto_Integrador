const STORAGE_KEY = 'fincontrol_transacoes';
const META_POUPANCA_KEY = 'fincontrol_meta_poupanca';
const DEFAULT_META_POUPANCA = 500;
const BACKEND_URL_KEY = 'fincontrol_backend_url';
const DEFAULT_BACKEND_URL = 'http://localhost:5000';

const CATEGORIAS = [
  'Moradia', 'Alimentação', 'Transporte', 'Lazer',
  'Saúde', 'Educação', 'Emprego', 'Renda Extra', 'Outros'
];

const DADOS_INICIAIS = [
  { id: 1, tipo: 'receita', descricao: 'Salário', valor: 4200, data: '2025-04-25', categoria: 'Emprego', observacoes: '' },
  { id: 2, tipo: 'receita', descricao: 'Freela Design', valor: 650, data: '2025-04-22', categoria: 'Renda Extra', observacoes: '' },
  { id: 3, tipo: 'despesa', descricao: 'Aluguel', valor: 900, data: '2025-04-24', categoria: 'Moradia', observacoes: '' }
];

function obterBackendUrl() {
  return localStorage.getItem(BACKEND_URL_KEY) || DEFAULT_BACKEND_URL;
}

async function enviarParaBackend(endpoint, dados = null, method = 'POST') {
  try {
    const url = `${obterBackendUrl()}${endpoint}`;
    const options = {
      method: method,
      headers: {
        'Content-Type': 'application/json',
      },
    };
    if (dados) {
      options.body = JSON.stringify(dados);
    }
    const response = await fetch(url, options);
    return await response.json();
  } catch (error) {
    console.error('Erro na comunicação com o backend:', error);
    return { success: false, error: error.message };
  }
}

function inicializarDados() {
  if (!localStorage.getItem(STORAGE_KEY)) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(DADOS_INICIAIS));
  }
  if (!localStorage.getItem(META_POUPANCA_KEY)) {
    localStorage.setItem(META_POUPANCA_KEY, DEFAULT_META_POUPANCA.toString());
  }
}

function obterMetaPoupanca() {
  const valor = localStorage.getItem(META_POUPANCA_KEY);
  return valor ? parseFloat(valor) : DEFAULT_META_POUPANCA;
}

function salvarMetaPoupanca(valor) {
  localStorage.setItem(META_POUPANCA_KEY, valor.toString());
}

function obterTransacoes() {
  const dados = localStorage.getItem(STORAGE_KEY);
  return dados ? JSON.parse(dados) : [];
}

function salvarTransacoes(transacoes) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(transacoes));
}

function formatarMoeda(valor) {
  return valor.toLocaleString('pt-BR', {
    style: 'currency',
    currency: 'BRL'
  });
}

function formatarData(dataStr) {
  const data = new Date(dataStr + 'T00:00:00');
  return data.toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit' });
}

function obterMesAnoAtual() {
  const data = new Date();
  const meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'];
  return `${meses[data.getMonth()]} ${data.getFullYear()}`;
}

function calcularTotais() {
  const transacoes = obterTransacoes();
  const totalReceitas = transacoes
    .filter(t => t.tipo === 'receita')
    .reduce((sum, t) => sum + t.valor, 0);
  const totalDespesas = transacoes
    .filter(t => t.tipo === 'despesa')
    .reduce((sum, t) => sum + t.valor, 0);
  const saldoAtual = totalReceitas - totalDespesas;
  const poupanca = totalReceitas * 0.12;
  const meta = obterMetaPoupanca();
  const percentualPoupanca = Math.min(100, Math.round((poupanca / meta) * 100));

  return { totalReceitas, totalDespesas, saldoAtual, poupanca, percentualPoupanca, metaPoupanca: meta };
}

function obterUltimasTransacoes(limite = 5) {
  const transacoes = obterTransacoes();
  return transacoes
    .sort((a, b) => new Date(b.data) - new Date(a.data))
    .slice(0, limite);
}

async function sincronizarComServidor() {
  const transacoes = obterTransacoes();
  await enviarParaBackend('/api/transacoes', { dados: transacoes }, 'POST');
  await enviarParaBackend('/api/etl', { dados: transacoes }, 'POST');
}

function adicionarTransacao(transacao) {
  const transacoes = obterTransacoes();
  const novaTransacao = {
    id: Date.now(),
    ...transacao
  };
  transacoes.push(novaTransacao);
  salvarTransacoes(transacoes);
  sincronizarComServidor();
}

function excluirTransacao(id) {
  const transacoes = obterTransacoes();
  const transacoesAtualizadas = transacoes.filter(t => t.id !== id);
  salvarTransacoes(transacoesAtualizadas);
  sincronizarComServidor();
}

async function carregarDadosDoServidor() {
  try {
    const resposta = await fetch(`${obterBackendUrl()}/api/transacoes`);
    const resultado = await resposta.json();
    if (resultado.success && resultado.data && resultado.data.length > 0) {
      salvarTransacoes(resultado.data);
      return true;
    }
  } catch (error) {
    console.log('Servidor offline, usando dados locais');
  }
  return false;
}

function renderizarTabelaUltimasTransacoes() {
  const tbody = document.getElementById('tabelaUltimasTransacoes');
  if (!tbody) return;

  const transacoes = obterUltimasTransacoes();
  tbody.innerHTML = transacoes.map(t => `
    <tr>
      <td>${formatarData(t.data)}</td>
      <td>${t.descricao}</td>
      <td>${t.categoria}</td>
      <td><span class="badge ${t.tipo === 'receita' ? 'badge-success' : 'badge-danger'}">${t.tipo === 'receita' ? 'Entrada' : 'Saída'}</span></td>
      <td style="color: ${t.tipo === 'receita' ? '#10b981' : '#ef4444'};">
        ${t.tipo === 'receita' ? '+' : '-'}${formatarMoeda(t.valor)}
      </td>
    </tr>
  `).join('');
}

function renderizarDashboard() {
  const mesAnoEl = document.getElementById('mesAno');
  if (mesAnoEl) {
    mesAnoEl.textContent = obterMesAnoAtual();
  }

  const totais = calcularTotais();

  const saldoEl = document.getElementById('saldoAtual');
  if (saldoEl) {
    saldoEl.textContent = formatarMoeda(totais.saldoAtual);
    saldoEl.className = 'card-value ' + (totais.saldoAtual >= 0 ? 'positive' : 'negative');
  }

  const receitasEl = document.getElementById('totalReceitas');
  if (receitasEl) {
    receitasEl.textContent = formatarMoeda(totais.totalReceitas);
  }

  const despesasEl = document.getElementById('totalDespesas');
  if (despesasEl) {
    despesasEl.textContent = formatarMoeda(totais.totalDespesas);
  }

  const metaEl = document.getElementById('metaPoupanca');
  if (metaEl) {
    metaEl.textContent = formatarMoeda(totais.metaPoupanca);
  }

  const progressEl = document.getElementById('progressPoupanca');
  if (progressEl) {
    progressEl.style.width = totais.percentualPoupanca + '%';
  }

  const percentualEl = document.getElementById('percentualPoupanca');
  if (percentualEl) {
    percentualEl.textContent = totais.percentualPoupanca + '%';
  }

  renderizarTabelaUltimasTransacoes();
  renderizarGraficoBarras();
  renderizarGraficoPizza();
}

function renderizarGraficoBarras() {
  const ctx = document.getElementById('chartBar');
  if (!ctx) return;

  const transacoes = obterTransacoes();
  const meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'];
  const dadosMensais = {};
  
  transacoes.forEach(t => {
    const data = new Date(t.data + 'T00:00:00');
    const mesAno = `${meses[data.getMonth()]}/${data.getFullYear().toString().slice(-2)}`;
    
    if (!dadosMensais[mesAno]) {
      dadosMensais[mesAno] = { receitas: 0, despesas: 0 };
    }
    
    if (t.tipo === 'receita') {
      dadosMensais[mesAno].receitas += t.valor;
    } else {
      dadosMensais[mesAno].despesas += t.valor;
    }
  });
  
  const labels = Object.keys(dadosMensais);
  const dadosReceitas = labels.map(label => dadosMensais[label].receitas);
  const dadosDespesas = labels.map(label => dadosMensais[label].despesas);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Receitas',
          data: dadosReceitas,
          backgroundColor: '#10b981',
          borderRadius: 4
        },
        {
          label: 'Despesas',
          data: dadosDespesas,
          backgroundColor: '#ef4444',
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom'
        }
      }
    }
  });
}

function renderizarGraficoPizza() {
  const ctx = document.getElementById('chartPie');
  if (!ctx) return;

  const transacoes = obterTransacoes();
  const despesas = transacoes.filter(t => t.tipo === 'despesa');

  const valoresPorCategoria = {};
  despesas.forEach(d => {
    valoresPorCategoria[d.categoria] = (valoresPorCategoria[d.categoria] || 0) + d.valor;
  });

  const labels = Object.keys(valoresPorCategoria);
  const valores = Object.values(valoresPorCategoria);
  const cores = ['#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [
        {
          data: valores,
          backgroundColor: cores.slice(0, labels.length),
          borderWidth: 0
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom'
        }
      }
    }
  });
}

function carregarCategoriasNoSelect() {
  const selectCategoria = document.getElementById('categoria');
  if (selectCategoria) {
    CATEGORIAS.forEach(categoria => {
      const option = document.createElement('option');
      option.value = categoria;
      option.textContent = categoria;
      selectCategoria.appendChild(option);
    });
  }
}

function configurarFormularioLancamento() {
  carregarCategoriasNoSelect();
  
  const form = document.getElementById('formLancamento');
  const valorInput = document.getElementById('valor');
  const dataInput = document.getElementById('data');

  if (dataInput) {
    dataInput.valueAsDate = new Date();
  }

  if (valorInput) {
    valorInput.addEventListener('input', function(e) {
      let valor = e.target.value.replace(/\D/g, '');
      if (valor.length > 0) {
        valor = (parseInt(valor) / 100).toFixed(2);
        e.target.value = valor.replace('.', ',');
      }
    });
  }

  if (form) {
    form.addEventListener('submit', function(e) {
      e.preventDefault();

      const tipo = document.querySelector('input[name="tipo"]:checked').value;
      const descricao = document.getElementById('descricao').value;
      const valorStr = document.getElementById('valor').value.replace(',', '.');
      const valor = parseFloat(valorStr);
      const data = document.getElementById('data').value;
      const categoria = document.getElementById('categoria').value;
      const observacoes = document.getElementById('observacoes').value;

      adicionarTransacao({
        tipo,
        descricao,
        valor,
        data,
        categoria,
        observacoes
      });

      window.location.href = 'dashboard.html';
    });
  }
}

function renderizarTodasTransacoes(transacoesFiltradas = null) {
  const tbody = document.getElementById('tabelaTransacoes');
  if (!tbody) return;

  const transacoes = (transacoesFiltradas || obterTransacoes())
    .sort((a, b) => new Date(b.data) - new Date(a.data));

  tbody.innerHTML = transacoes.map(t => `
    <tr>
      <td>${formatarData(t.data)}</td>
      <td>${t.descricao}</td>
      <td>${t.categoria}</td>
      <td><span class="badge ${t.tipo === 'receita' ? 'badge-success' : 'badge-danger'}">${t.tipo === 'receita' ? 'Entrada' : 'Saída'}</span></td>
      <td style="color: ${t.tipo === 'receita' ? '#10b981' : '#ef4444'};">
        ${t.tipo === 'receita' ? '+' : '-'}${formatarMoeda(t.valor)}
      </td>
      <td class="actions">
        <button onclick="handleExcluir(${t.id})">🗑️</button>
      </td>
    </tr>
  `).join('');
}

function aplicarFiltros() {
  const filtroMes = document.getElementById('filtroMes');
  const filtroTipo = document.getElementById('filtroTipo');
  const filtroCategoria = document.getElementById('filtroCategoria');
  const filtroBusca = document.getElementById('filtroBusca');

  let transacoes = obterTransacoes();

  if (filtroMes && filtroMes.value !== 'todos') {
    const mesSelecionado = parseInt(filtroMes.value);
    transacoes = transacoes.filter(t => {
      const dataTransacao = new Date(t.data + 'T00:00:00');
      return dataTransacao.getMonth() + 1 === mesSelecionado;
    });
  }

  if (filtroTipo && filtroTipo.value !== 'todos') {
    transacoes = transacoes.filter(t => t.tipo === filtroTipo.value);
  }

  if (filtroCategoria && filtroCategoria.value !== 'todas') {
    transacoes = transacoes.filter(t => t.categoria === filtroCategoria.value);
  }

  if (filtroBusca && filtroBusca.value.trim() !== '') {
    const busca = filtroBusca.value.toLowerCase().trim();
    transacoes = transacoes.filter(t => 
      t.descricao.toLowerCase().includes(busca) || 
      t.categoria.toLowerCase().includes(busca)
    );
  }

  renderizarTodasTransacoes(transacoes);
}

function handleExcluir(id) {
  if (confirm('Tem certeza que deseja excluir esta transação?')) {
    excluirTransacao(id);
    renderizarTodasTransacoes();
  }
}

function configurarFiltros() {
  const filtroCategoria = document.getElementById('filtroCategoria');
  if (filtroCategoria) {
    CATEGORIAS.forEach(cat => {
      const option = document.createElement('option');
      option.value = cat;
      option.textContent = cat;
      filtroCategoria.appendChild(option);
    });
  }

  const filtroMes = document.getElementById('filtroMes');
  if (filtroMes) {
    const optionTodos = document.createElement('option');
    optionTodos.value = 'todos';
    optionTodos.textContent = 'Todos os meses';
    filtroMes.appendChild(optionTodos);
    
    const meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                   'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'];
    const dataAtual = new Date();
    meses.forEach((mes, idx) => {
      const option = document.createElement('option');
      option.value = idx + 1;
      option.textContent = `${mes} ${dataAtual.getFullYear()}`;
      filtroMes.appendChild(option);
    });
  }

  const btnFiltrar = document.getElementById('btnFiltrar');
  if (btnFiltrar) {
    btnFiltrar.addEventListener('click', aplicarFiltros);
  }

  const filtroBusca = document.getElementById('filtroBusca');
  if (filtroBusca) {
    filtroBusca.addEventListener('keyup', function(e) {
      if (e.key === 'Enter') {
        aplicarFiltros();
      }
    });
  }
}

function renderizarAlertas() {
  const container = document.getElementById('alertasContainer');
  if (!container) return;

  const alertas = [
    {
      tipo: 'warning',
      icon: '⚠️',
      texto: 'Você está próximo de atingir o limite de gastos com lazer este mês.'
    },
    {
      tipo: 'info',
      icon: '💡',
      texto: 'Lembre-se de reservar pelo menos 10% da sua receita para a poupança!'
    },
    {
      tipo: 'danger',
      icon: '🔔',
      texto: 'A data de vencimento do aluguel está chegando!'
    }
  ];

  container.innerHTML = alertas.map(a => `
    <div class="alert alert-${a.tipo}">
      <span class="alert-icon">${a.icon}</span>
      <div>${a.texto}</div>
    </div>
  `).join('');
}

function obterDadosMensais() {
  const transacoes = obterTransacoes();
  const meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'];
  const dadosMensais = {};
  
  transacoes.forEach(t => {
    const data = new Date(t.data + 'T00:00:00');
    const mesAno = `${meses[data.getMonth()]}/${data.getFullYear().toString().slice(-2)}`;
    
    if (!dadosMensais[mesAno]) {
      dadosMensais[mesAno] = { receitas: 0, despesas: 0, saldo: 0 };
    }
    
    if (t.tipo === 'receita') {
      dadosMensais[mesAno].receitas += t.valor;
    } else {
      dadosMensais[mesAno].despesas += t.valor;
    }
  });
  
  const labels = Object.keys(dadosMensais);
  const saldos = [];
  let saldoAcumulado = 0;
  
  labels.forEach(label => {
    saldoAcumulado += dadosMensais[label].receitas - dadosMensais[label].despesas;
    saldos.push(saldoAcumulado);
  });
  
  return { labels, saldos };
}

function obterDadosReceitasPorCategoria() {
  const transacoes = obterTransacoes();
  const receitas = transacoes.filter(t => t.tipo === 'receita');
  
  const valoresPorCategoria = {};
  receitas.forEach(r => {
    valoresPorCategoria[r.categoria] = (valoresPorCategoria[r.categoria] || 0) + r.valor;
  });
  
  return {
    labels: Object.keys(valoresPorCategoria),
    valores: Object.values(valoresPorCategoria)
  };
}

function renderizarGraficosRelatorios() {
  const ctxSaldo = document.getElementById('chartSaldo');
  if (ctxSaldo) {
    const dadosMensais = obterDadosMensais();
    new Chart(ctxSaldo, {
      type: 'line',
      data: {
        labels: dadosMensais.labels,
        datasets: [{
          label: 'Saldo',
          data: dadosMensais.saldos,
          borderColor: '#4f46e5',
          backgroundColor: 'rgba(79, 70, 229, 0.1)',
          fill: true,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'bottom' } }
      }
    });
  }

  const ctxReceitas = document.getElementById('chartReceitas');
  if (ctxReceitas) {
    const dadosReceitas = obterDadosReceitasPorCategoria();
    const cores = ['#10b981', '#4f46e5', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899'];
    
    new Chart(ctxReceitas, {
      type: 'pie',
      data: {
        labels: dadosReceitas.labels.length > 0 ? dadosReceitas.labels : ['Nenhuma receita'],
        datasets: [{
          data: dadosReceitas.valores.length > 0 ? dadosReceitas.valores : [1],
          backgroundColor: dadosReceitas.labels.length > 0 ? cores.slice(0, dadosReceitas.labels.length) : ['#cbd5e1'],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'bottom' } }
      }
    });
  }
}

function configurarPaginaConfiguracoes() {
  const btnExportarJSON = document.getElementById('btnExportarJSON');
  const btnExportarCSV = document.getElementById('btnExportarCSV');
  const btnImportar = document.getElementById('btnImportar');
  const inputImportar = document.getElementById('inputImportar');
  const backendUrl = document.getElementById('backendUrl');
  const btnResetarDados = document.getElementById('btnResetarDados');
  const metaPoupancaInput = document.getElementById('metaPoupancaInput');
  const btnSalvarMeta = document.getElementById('btnSalvarMeta');

  if (metaPoupancaInput) {
    metaPoupancaInput.value = obterMetaPoupanca();
  }

  if (btnSalvarMeta && metaPoupancaInput) {
    btnSalvarMeta.addEventListener('click', () => {
      const novaMeta = parseFloat(metaPoupancaInput.value);
      if (!isNaN(novaMeta) && novaMeta >= 0) {
        salvarMetaPoupanca(novaMeta);
        alert('Meta de poupança atualizada com sucesso!');
      } else {
        alert('Por favor, insira um valor válido para a meta.');
      }
    });
  }

  if (backendUrl) {
    backendUrl.value = obterBackendUrl();
    backendUrl.addEventListener('change', function() {
      localStorage.setItem(BACKEND_URL_KEY, this.value);
    });
  }

  if (btnExportarJSON) {
    btnExportarJSON.addEventListener('click', exportarDadosJSON);
  }

  if (btnExportarCSV) {
    btnExportarCSV.addEventListener('click', exportarDadosCSV);
  }

  if (btnImportar && inputImportar) {
    btnImportar.addEventListener('click', () => {
      if (inputImportar.files.length > 0) {
        importarDados(inputImportar.files[0]);
      }
    });
  }

  if (btnResetarDados) {
    btnResetarDados.addEventListener('click', () => {
      if (confirm('Tem certeza que deseja resetar todos os dados?')) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(DADOS_INICIAIS));
        sincronizarComServidor();
        alert('Dados resetados com sucesso!');
        window.location.href = 'dashboard.html';
      }
    });
  }
}

function exportarDadosJSON() {
  const dados = obterTransacoes();
  const blob = new Blob([JSON.stringify(dados, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `fincontrol_dados_${new Date().toISOString().split('T')[0]}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportarDadosCSV() {
  const dados = obterTransacoes();
  if (dados.length === 0) {
    alert('Não há dados para exportar');
    return;
  }

  const headers = Object.keys(dados[0]);
  const csvContent = [
    headers.join(','),
    ...dados.map(obj => headers.map(header => {
      const value = obj[header];
      if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
        return `"${value.replace(/"/g, '""')}"`;
      }
      return value;
    }).join(','))
  ].join('\n');

  const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `fincontrol_dados_${new Date().toISOString().split('T')[0]}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function importarDados(file) {
  const reader = new FileReader();
  reader.onload = async function(e) {
    try {
      const dados = JSON.parse(e.target.result);
      if (Array.isArray(dados)) {
        salvarTransacoes(dados);
        await sincronizarComServidor();
        alert('Dados importados com sucesso!');
        window.location.href = 'dashboard.html';
      } else {
        alert('Arquivo inválido. Por favor, importe um arquivo JSON válido.');
      }
    } catch (error) {
      alert('Erro ao importar arquivo: ' + error.message);
    }
  };
  reader.readAsText(file);
}

document.addEventListener('DOMContentLoaded', async function() {
  inicializarDados();
  
  await carregarDadosDoServidor();

  const pagina = window.location.pathname.split('/').pop();

  if (pagina === 'dashboard.html' || pagina === '') {
    renderizarDashboard();
  } else if (pagina === 'lancamento.html') {
    configurarFormularioLancamento();
  } else if (pagina === 'transacoes.html') {
    renderizarTodasTransacoes();
    configurarFiltros();
  } else if (pagina === 'relatorios.html') {
    renderizarGraficosRelatorios();
  } else if (pagina === 'configuracoes.html') {
    configurarPaginaConfiguracoes();
  }
});
