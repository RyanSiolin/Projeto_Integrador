/* FinControl - Acessibilidade (padrão eMAG) + modo escuro
   - Barra de acessibilidade: modo escuro, alto contraste, tamanho de fonte
   - Atalhos de teclado / links de acesso rápido (accesskey 1, 2, 4)
   - VLibras (tradução em Libras para usuários surdos)
   - Preferências persistidas em localStorage
*/
(function () {
  const html = document.documentElement;
  const LS = { dark: 'acc_dark', hc: 'acc_hc', fs: 'acc_fs' };
  const steps = ['87.5%', '100%', '112.5%', '125%', '137.5%'];

  // ---- aplica preferências salvas ----
  if (localStorage.getItem(LS.dark) === '1') html.setAttribute('data-bs-theme', 'dark');
  if (localStorage.getItem(LS.hc) === '1') html.classList.add('alto-contraste');
  const savedFs = localStorage.getItem(LS.fs);
  if (savedFs) html.style.fontSize = savedFs;

  function build() {
    // ---- links de acesso rápido (skip links) ----
    const skip = document.createElement('nav');
    skip.className = 'skip-links';
    skip.setAttribute('aria-label', 'Atalhos de acessibilidade');
    skip.innerHTML =
      '<a href="#conteudo" accesskey="1">Ir para o conteúdo [1]</a>' +
      '<a href="#menu" accesskey="2">Ir para o menu [2]</a>' +
      '<a href="#rodape" accesskey="4">Ir para o rodapé [4]</a>';
    document.body.insertBefore(skip, document.body.firstChild);

    // ---- barra de acessibilidade ----
    const bar = document.createElement('div');
    bar.className = 'barra-acessibilidade';
    bar.setAttribute('role', 'region');
    bar.setAttribute('aria-label', 'Barra de acessibilidade');
    bar.innerHTML =
      '<div class="container d-flex flex-wrap gap-2 align-items-center justify-content-end py-1">' +
      '<button type="button" id="accDark" class="btn-acc"><i class="bi bi-moon-stars" aria-hidden="true"></i> <span>Modo escuro</span></button>' +
      '<button type="button" id="accHc" class="btn-acc"><i class="bi bi-circle-half" aria-hidden="true"></i> Alto contraste</button>' +
      '<span class="btn-acc-group" role="group" aria-label="Tamanho da fonte">' +
      '<button type="button" id="accMinus" class="btn-acc" aria-label="Diminuir fonte">A-</button>' +
      '<button type="button" id="accReset" class="btn-acc" aria-label="Tamanho de fonte padrão">A</button>' +
      '<button type="button" id="accPlus" class="btn-acc" aria-label="Aumentar fonte">A+</button>' +
      '</span></div>';
    document.body.insertBefore(bar, skip.nextSibling);

    // ---- fonte ----
    const curIdx = () => { const i = steps.indexOf(html.style.fontSize || '100%'); return i < 0 ? 1 : i; };
    const setFs = v => { html.style.fontSize = v; localStorage.setItem(LS.fs, v); };
    document.getElementById('accPlus').onclick = () => setFs(steps[Math.min(curIdx() + 1, steps.length - 1)]);
    document.getElementById('accMinus').onclick = () => setFs(steps[Math.max(curIdx() - 1, 0)]);
    document.getElementById('accReset').onclick = () => setFs('100%');

    // ---- modo escuro ----
    const dBtn = document.getElementById('accDark');
    function syncDark() {
      const on = html.getAttribute('data-bs-theme') === 'dark';
      dBtn.querySelector('span').textContent = on ? 'Modo claro' : 'Modo escuro';
      dBtn.querySelector('i').className = (on ? 'bi bi-sun' : 'bi bi-moon-stars');
      dBtn.setAttribute('aria-pressed', on);
    }
    dBtn.onclick = () => {
      const on = html.getAttribute('data-bs-theme') === 'dark';
      localStorage.setItem(LS.dark, on ? '0' : '1');
      // recarrega para os gráficos (canvas) redesenharem com as cores do tema
      location.reload();
    };
    syncDark();

    // ---- alto contraste ----
    const hcBtn = document.getElementById('accHc');
    hcBtn.onclick = () => {
      const on = html.classList.toggle('alto-contraste');
      localStorage.setItem(LS.hc, on ? '1' : '0');
      hcBtn.setAttribute('aria-pressed', on);
    };
    hcBtn.setAttribute('aria-pressed', html.classList.contains('alto-contraste'));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
  else build();
})();
