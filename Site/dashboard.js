// Dados e configuração dos gráficos do dashboard

// Cores dos rótulos conforme o tema (claro/escuro) — evita texto cinza ilegível no modo escuro
(function () {
  const dark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
  Chart.defaults.color = dark ? '#f1f5f9' : '#475569';
  Chart.defaults.borderColor = dark ? 'rgba(148,163,184,.25)' : 'rgba(0,0,0,.08)';
})();

// Faixa Etária
const ctx1 = document.getElementById('chartFaixaEtaria').getContext('2d');
new Chart(ctx1, {
  type: 'pie',
  data: {
    labels: ['18-24 Anos', '25-34 Anos', '35-44 Anos', '45+ Anos'],
    datasets: [{
      data: [59, 5, 2, 4],
      backgroundColor: ['#0f766e', '#334155', '#06b6d4', '#10b981'],
      borderWidth: 0
    }]
  },
  options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
});

// Controle de Gastos
const ctx2 = document.getElementById('chartControleGastos').getContext('2d');
new Chart(ctx2, {
  type: 'doughnut',
  data: {
    labels: ['Controla', 'Às vezes', 'Não controla'],
    datasets: [{
      data: [37, 17, 16],
      backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
      borderWidth: 0
    }]
  },
  options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
});

// Reserva de Emergência
const ctx3 = document.getElementById('chartReservaEmergencia').getContext('2d');
new Chart(ctx3, {
  type: 'bar',
  data: {
    labels: ['Possui', 'Está construindo', 'Não possui'],
    datasets: [{
      label: 'Participantes',
      data: [34, 20, 16],
      backgroundColor: ['#10b981', '#0f766e', '#64748b'],
      borderRadius: 8
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { y: { beginAtZero: true } }
  }
});

// Dificuldades Financeiras
const ctx4 = document.getElementById('chartDificuldades').getContext('2d');
new Chart(ctx4, {
  type: 'radar',
  data: {
    labels: ['Controle de gastos', 'Investimentos', 'Dívidas', 'Planejamento mensal', 'Juros compostos'],
    datasets: [{
      label: 'Nível de dificuldade',
      data: [4.2, 4.8, 4.5, 4.0, 4.7],
      borderColor: '#334155',
      backgroundColor: 'rgba(118, 75, 162, 0.2)',
      borderWidth: 2
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom' } },
    scales: { r: { beginAtZero: true, max: 5 } }
  }
});

// Taxa Selic
const ctx5 = document.getElementById('chartSelic').getContext('2d');
new Chart(ctx5, {
  type: 'line',
  data: {
    labels: ['1994', '1998', '2002', '2006', '2010', '2014', '2018', '2020', '2022', '2024'],
    datasets: [{
      label: 'Taxa Selic (% a.a.)',
      data: [50, 25, 26, 13, 11, 12, 6.5, 2, 13.75, 14.75],
      borderColor: '#0f766e',
      backgroundColor: 'rgba(15,118,110,0.12)',
      tension: 0.4,
      fill: true,
      pointRadius: 5,
      pointBackgroundColor: '#0f766e'
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom' } },
    scales: {
      y: { beginAtZero: true, title: { display: true, text: '% ao ano' } }
    }
  }
});

// Situação Profissional
const ctx6 = document.getElementById('chartSituacaoProfissional').getContext('2d');
new Chart(ctx6, {
  type: 'bar',
  data: {
    labels: ['Estudante', 'Estagiário', 'Autônomo', 'Empregado CLT', 'Desempregado', 'Empresário', 'Militar'],
    datasets: [{
      label: 'Participantes',
      data: [38, 15, 7, 6, 2, 1, 1],
      backgroundColor: ['#0f766e', '#334155', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
      borderRadius: 8
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { y: { beginAtZero: true } }
  }
});

// Tipos de Investimentos
const ctx7 = document.getElementById('chartTiposInvestimentos').getContext('2d');
new Chart(ctx7, {
  type: 'doughnut',
  data: {
    labels: ['Renda Fixa (CDB/Tesouro/LCI)', 'Ações', 'Criptomoedas', 'Poupança', 'Imóveis', 'Outros'],
    datasets: [{
      data: [20, 8, 5, 4, 2, 2],
      backgroundColor: ['#10b981', '#0f766e', '#f59e0b', '#06b6d4', '#ef4444', '#8b5cf6'],
      borderWidth: 0
    }]
  },
  options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
});

// Formato de Aprendizado Preferido
const ctx8 = document.getElementById('chartFormatoAprendizado').getContext('2d');
new Chart(ctx8, {
  type: 'pie',
  data: {
    labels: ['Vídeos', 'Aplicativos', 'Cursos Online', 'Redes Sociais', 'Artigos'],
    datasets: [{
      data: [55, 40, 25, 20, 10],
      backgroundColor: ['#0f766e', '#334155', '#10b981', '#f59e0b', '#06b6d4'],
      borderWidth: 0
    }]
  },
  options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
});

// Nota de Educação Financeira
const ctx9 = document.getElementById('chartNotaEducacao').getContext('2d');
new Chart(ctx9, {
  type: 'bar',
  data: {
    labels: ['Nota 1', 'Nota 2', 'Nota 3', 'Nota 4', 'Nota 5'],
    datasets: [{
      label: 'Participantes',
      data: [3, 12, 27, 24, 1],
      backgroundColor: ['#ef4444', '#f59e0b', '#0f766e', '#10b981', '#06b6d4'],
      borderRadius: 8
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { y: { beginAtZero: true } }
  }
});
