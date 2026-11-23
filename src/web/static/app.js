const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const colors = ['#31d78a', '#4d95ff', '#a275f7', '#e7b857', '#ef7474', '#57cbd1'];
const sampleAssets = [
  { ticker: 'SPY', weight: 35 }, { ticker: 'QQQ', weight: 25 },
  { ticker: 'TLT', weight: 20 }, { ticker: 'GLD', weight: 20 },
];
let portfolioState = { values: [], risk: null, assets: [], period: 'ALL', sort: { key: 'asset', ascending: true } };
let selectedPpoRunId = null;

const money = (value) => Number.isFinite(Number(value)) ? Number(value).toLocaleString('en-IN', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }) : '--';
const percent = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : '--';
const number = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(2) : '--';
const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character]));

function showView(view) {
  $$('.view').forEach((element) => element.classList.toggle('active', element.id === view));
  $$('#navigation button').forEach((button) => button.classList.toggle('active', button.dataset.view === view));
  history.replaceState(null, '', `#${view}`);
}

$$('#navigation button').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view)));
showView(location.hash.slice(1) || 'overview');

$$('[data-tabs] button').forEach((button) => button.addEventListener('click', () => {
  const tab = button.dataset.tab;
  $$('[data-tabs] button').forEach((item) => item.classList.toggle('active', item === button));
  $$('[data-tab-panel]').forEach((panel) => panel.classList.toggle('active', panel.dataset.tabPanel === tab));
}));

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || 'The requested data is unavailable.');
  return data;
}

async function refreshStorage() {
  try {
    const stats = await request('/api/dashboard/stats');
    $('#storage-json').textContent = JSON.stringify(stats, null, 2);
  } catch (error) {
    $('#storage-json').textContent = `Storage statistics unavailable: ${error.message}`;
  }
}

function parseCsv(text) {
  const rows = []; let row = []; let field = ''; let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]; const next = text[index + 1];
    if (character === '"' && quoted && next === '"') { field += '"'; index += 1; }
    else if (character === '"') quoted = !quoted;
    else if (character === ',' && !quoted) { row.push(field.trim()); field = ''; }
    else if ((character === '\n' || character === '\r') && !quoted) {
      if (character === '\r' && next === '\n') index += 1;
      row.push(field.trim()); if (row.some((value) => value)) rows.push(row); row = []; field = '';
    } else field += character;
  }
  row.push(field.trim()); if (row.some((value) => value)) rows.push(row);
  return rows;
}

function findColumn(headers, candidates) {
  const normalized = headers.map((header) => header.toLowerCase().replace(/[^a-z0-9]/g, ''));
  const position = normalized.findIndex((header) => candidates.includes(header));
  return position < 0 ? -1 : position;
}

function renderCsvAnalysis(filename, headers, rows) {
  const result = $('#csv-analysis-result');
  const summary = $('#csv-analysis-summary');
  const dateIndex = findColumn(headers, ['date', 'datetime', 'timestamp', 'time']);
  const valueIndex = findColumn(headers, ['adjustedclose', 'adjclose', 'close', 'price', 'value', 'nav']);
  const numericColumns = headers.filter((_, index) => rows.some((row) => Number.isFinite(Number(String(row[index] || '').replace(/,/g, '')))));
  summary.innerHTML = [
    ['File', filename], ['Rows detected', rows.length], ['Columns', headers.length],
    ['Available fields', headers.join(', ')], ['Numeric columns', numericColumns.join(', ') || 'None detected'],
  ].map(([label, value]) => `<div class="result-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join('');
  const previewHeaders = headers.slice(0, 8); const previewRows = rows.slice(0, 8);
  $('#csv-analysis-table thead').innerHTML = `<tr>${previewHeaders.map((header) => `<th>${escapeHtml(header)}</th>`).join('')}</tr>`;
  $('#csv-analysis-table tbody').innerHTML = previewRows.map((row) => `<tr>${previewHeaders.map((_, index) => `<td>${escapeHtml(row[index] || '')}</td>`).join('')}</tr>`).join('');
  if (dateIndex >= 0 && valueIndex >= 0) {
    const series = rows.map((row) => ({ date: row[dateIndex], value: Number(String(row[valueIndex] || '').replace(/,/g, '')) }))
      .filter((point) => !Number.isNaN(new Date(point.date).getTime()) && Number.isFinite(point.value))
      .sort((left, right) => new Date(left.date) - new Date(right.date));
    drawChart('#csv-analysis-chart', series);
  } else $('#csv-analysis-chart').innerHTML = '<div class="chart-empty">No chart created. Add a Date column and a Close, Adjusted Close, Price, Value, or NAV column to view a series.</div>';
  result.hidden = false;
}

$('#csv-analysis-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = $('#csv-analysis-file').files[0]; const message = $('#csv-analysis-message'); const button = $('#csv-analysis-button');
  if (!file) { message.className = 'form-message error'; message.textContent = 'Choose a CSV file first.'; return; }
  if (file.size > 10 * 1024 * 1024) { message.className = 'form-message error'; message.textContent = 'CSV files must be 10 MB or smaller.'; return; }
  button.disabled = true; message.className = 'form-message'; message.textContent = 'Analysing your CSV locally...';
  try {
    const parsed = parseCsv(await file.text());
    if (parsed.length < 2) throw new Error('The CSV needs a header row and at least one data row.');
    const headers = parsed[0]; const rows = parsed.slice(1).filter((row) => row.length === headers.length);
    if (!headers.length || !rows.length) throw new Error('No usable rows were found in this CSV.');
    renderCsvAnalysis(file.name, headers, rows); message.className = 'form-message success'; message.textContent = 'CSV analysis completed. The file was not sent to the server or used for training.';
  } catch (error) { message.className = 'form-message error'; message.textContent = error.message; }
  finally { button.disabled = false; }
});

function currentAssets() {
  return $$('.allocation-row').map((row) => ({
    ticker: row.querySelector('[data-ticker]').value.trim().toUpperCase(),
    weight: Number(row.querySelector('[data-weight]').value) || 0,
  })).filter((asset) => asset.ticker);
}

function renderAllocation() {
  const assets = currentAssets();
  const total = assets.reduce((sum, item) => sum + item.weight, 0);
  let cursor = 0;
  const segments = assets.map((asset, index) => {
    const start = cursor;
    cursor += total ? (asset.weight / total) * 360 : 0;
    return `${colors[index % colors.length]} ${start}deg ${cursor}deg`;
  });
  $('#allocation-chart').style.background = segments.length ? `conic-gradient(${segments.join(',')})` : '#21372a';
  $('#allocation-total').textContent = `${total.toFixed(0)}%`;
  $('#allocation-legend').innerHTML = assets.length ? assets.map((asset, index) => `<div class="legend-row"><span><i class="legend-dot" style="background:${colors[index % colors.length]}"></i>${escapeHtml(asset.ticker)}</span><b>${asset.weight.toFixed(1)}%</b></div>`).join('') : '<span class="empty-inline">No target weights entered.</span>';
}

function addAsset(asset = { ticker: '', weight: '' }) {
  const row = document.createElement('div');
  row.className = 'allocation-row';
  row.innerHTML = `<input data-ticker aria-label="Asset ticker" maxlength="10" placeholder="Ticker" value="${escapeHtml(asset.ticker)}"><input data-weight aria-label="Asset target weight percentage" type="number" min="0" max="100" step="0.1" placeholder="Weight %" value="${asset.weight}"><button class="remove-button" type="button" aria-label="Remove asset">x</button>`;
  row.querySelectorAll('input').forEach((input) => input.addEventListener('input', renderAllocation));
  row.querySelector('button').addEventListener('click', () => { row.remove(); renderAllocation(); });
  $('#allocation-rows').append(row);
}

sampleAssets.forEach(addAsset);
renderAllocation();
$('#add-asset').addEventListener('click', () => addAsset());

function portfolioPayload() {
  const assets = currentAssets();
  const total = assets.reduce((sum, item) => sum + item.weight, 0);
  if (!assets.length || Math.abs(total - 100) > 0.01) throw new Error('Add one or more assets and make target weights total exactly 100%.');
  const start = $('#analysis-start').value;
  const end = $('#analysis-end').value;
  if (!start || !end || start >= end) throw new Error('Choose a valid start date before the end date.');
  return {
    tickers: assets.map((asset) => asset.ticker),
    weights: Object.fromEntries(assets.map((asset) => [asset.ticker, asset.weight / 100])),
    start_date: start,
    end_date: end,
    initial_investment: Number($('#analysis-capital').value),
  };
}

function valueRows(rawValues) {
  return rawValues.map((item) => ({ date: String(item.datadate ?? item.index ?? item.date ?? ''), value: Number(item.value ?? item.portfolio_values ?? 0) })).filter((item) => item.date && Number.isFinite(item.value));
}

function filterValues(values) {
  if (portfolioState.period === 'ALL' || values.length < 2) return values;
  const days = { '1M': 31, '3M': 92, '6M': 184, '1Y': 366 }[portfolioState.period];
  const last = new Date(values[values.length - 1].date).getTime();
  return values.filter((item) => new Date(item.date).getTime() >= last - days * 86400000);
}

function drawChart(containerSelector, values) {
  const container = $(containerSelector);
  const data = filterValues(values);
  if (data.length < 2) { container.innerHTML = '<div class="chart-empty">No compatible portfolio value series is available for this period.</div>'; return; }
  const width = 900, height = 270, pad = { top: 18, right: 20, bottom: 32, left: 64 };
  const min = Math.min(...data.map((item) => item.value));
  const max = Math.max(...data.map((item) => item.value));
  const range = max - min || 1;
  const point = (item, index) => ({ x: pad.left + index * ((width - pad.left - pad.right) / (data.length - 1)), y: pad.top + (max - item.value) * ((height - pad.top - pad.bottom) / range) });
  const points = data.map(point);
  const polyline = points.map((item) => `${item.x.toFixed(2)},${item.y.toFixed(2)}`).join(' ');
  const area = `${pad.left},${height - pad.bottom} ${polyline} ${points[points.length - 1].x},${height - pad.bottom}`;
  const ticks = [0, .5, 1].map((ratio) => { const y = pad.top + ratio * (height - pad.top - pad.bottom); const value = max - ratio * range; return `<g><line x1="${pad.left}" x2="${width - pad.right}" y1="${y}" y2="${y}" stroke="#20352a" stroke-width="1"/><text x="4" y="${y + 4}" fill="#92a397" font-size="10">${money(value)}</text></g>`; }).join('');
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Portfolio value over time"><defs><linearGradient id="portfolioFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#31d78a" stop-opacity=".26"/><stop offset="100%" stop-color="#31d78a" stop-opacity="0"/></linearGradient></defs>${ticks}<polygon points="${area}" fill="url(#portfolioFill)"/><polyline points="${polyline}" fill="none" stroke="#31d78a" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/><line id="chart-guide" x1="0" x2="0" y1="${pad.top}" y2="${height - pad.bottom}" stroke="#8ddfb0" opacity="0"/></svg>`;
  const svg = container.querySelector('svg'); const guide = svg.querySelector('#chart-guide'); const tooltip = $('#chart-tooltip');
  svg.addEventListener('mousemove', (event) => { const box = svg.getBoundingClientRect(); const x = (event.clientX - box.left) * width / box.width; const index = Math.max(0, Math.min(data.length - 1, Math.round((x - pad.left) / (width - pad.left - pad.right) * (data.length - 1)))); const selected = points[index]; guide.setAttribute('x1', selected.x); guide.setAttribute('x2', selected.x); guide.setAttribute('opacity', '1'); tooltip.hidden = false; tooltip.style.left = `${event.clientX + 12}px`; tooltip.style.top = `${event.clientY + 12}px`; tooltip.innerHTML = `${escapeHtml(data[index].date)}<strong>${money(data[index].value)}</strong>`; });
  svg.addEventListener('mouseleave', () => { guide.setAttribute('opacity', '0'); tooltip.hidden = true; });
}

function renderRisk(risk) {
  const values = [['Volatility', percent(risk?.annualized_volatility)], ['Sharpe ratio', number(risk?.sharpe_ratio)], ['Max drawdown', percent(risk?.maximum_drawdown)], ['CVaR', percent(risk?.cvar)]];
  $('#risk-summary').innerHTML = values.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join('');
}

function renderPerformanceMetrics(performance) {
  const values = [['Current value', money(performance.current_value)], ['Total return', percent(performance.total_return)], ['Annualized return', percent(performance.annualized_return)]];
  $('#performance-metrics').innerHTML = values.map(([label, value]) => `<article><p>${label}</p><strong>${value}</strong></article>`).join('');
}

function renderAssets() {
  const assets = [...portfolioState.assets].sort((left, right) => { const key = portfolioState.sort.key; const a = key === 'asset' ? String(left.asset) : Number(left[key] ?? 0); const b = key === 'asset' ? String(right.asset) : Number(right[key] ?? 0); const result = a > b ? 1 : a < b ? -1 : 0; return portfolioState.sort.ascending ? result : -result; });
  const body = $('#asset-table tbody');
  body.innerHTML = assets.length ? assets.map((asset) => `<tr><td>${escapeHtml(asset.asset)}</td><td>${percent(asset.weight)}</td><td>${percent(asset.total_return)}</td><td>${percent(asset.annualized_volatility)}</td></tr>`).join('') : '<tr><td colspan="4" class="empty-cell">No asset analytics loaded.</td></tr>';
}

async function loadPortfolio() {
  const message = $('#form-message');
  try {
    const payload = portfolioPayload();
    message.className = 'form-message'; message.textContent = 'Loading stored historical data and calculating portfolio analytics...';
    const options = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) };
    const [performance, risk, assets] = await Promise.all([request('/portfolio/performance', options), request('/portfolio/risk', options), request('/portfolio/assets', options)]);
    portfolioState.values = valueRows(performance.portfolio_values || []); portfolioState.risk = risk;
    portfolioState.assets = (assets.metrics || []).map((asset) => ({ asset: asset.tic ?? asset.index ?? asset.asset ?? '--', weight: payload.weights[asset.tic ?? asset.index ?? asset.asset] ?? 0, total_return: asset.total_return, annualized_volatility: asset.annualized_volatility }));
    drawChart('#analysis-chart', portfolioState.values); drawChart('#performance-chart', portfolioState.values); renderRisk(risk); renderPerformanceMetrics(performance); renderAssets();
    $('#performance-caption').textContent = `Stored historical portfolio values from ${payload.start_date} to ${payload.end_date}.`;
    $('#chart-range').textContent = `${portfolioState.values.length} observations`;
    $('#export-portfolio').disabled = false;
    message.textContent = 'Portfolio analytics loaded from existing backend data.';
    showView('analysis');
  } catch (error) { message.className = 'form-message error'; message.textContent = error.message; }
}

$('#allocation-form').addEventListener('submit', (event) => { event.preventDefault(); loadPortfolio(); });
$$('#period-filters button').forEach((button) => button.addEventListener('click', () => { portfolioState.period = button.dataset.period; $$('#period-filters button').forEach((item) => item.classList.toggle('active', item === button)); drawChart('#performance-chart', portfolioState.values); drawChart('#analysis-chart', portfolioState.values); }));
$$('#asset-table th').forEach((header) => header.addEventListener('click', () => { const key = header.dataset.sort; portfolioState.sort.ascending = portfolioState.sort.key === key ? !portfolioState.sort.ascending : true; portfolioState.sort.key = key; renderAssets(); }));

$('#export-portfolio').addEventListener('click', () => { const rows = [['date', 'portfolio_value'], ...portfolioState.values.map((item) => [item.date, item.value])]; const content = rows.map((row) => row.join(',')).join('\n'); const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([content], { type: 'text/csv' })); link.download = 'portfolio-performance.csv'; link.click(); URL.revokeObjectURL(link.href); });

async function loadPpoValidation() {
  const message = $('#ppo-validation-message'); const button = $('#load-ppo-validation');
  button.disabled = true; message.className = 'form-message'; message.textContent = 'Loading deterministic PPO validation and benchmark results...';
  try {
    const url = selectedPpoRunId ? `/api/ppo/runs/${encodeURIComponent(selectedPpoRunId)}/validation` : '/api/ppo/validation';
    const data = await request(url); const strategies = data.evaluation?.strategies;
    if (!strategies?.['Synthetic PPO']) throw new Error('The saved evaluation did not include the expected PPO metrics.');
    const ppo = strategies['Synthetic PPO'];
    $('#ppo-nav').textContent = money(ppo.final_portfolio_value); $('#equal-nav').textContent = money(strategies['Equal-weight']?.final_portfolio_value); $('#buyhold-nav').textContent = money(strategies['SPY buy-and-hold']?.final_portfolio_value); $('#ppo-cost').textContent = money(ppo.total_transaction_costs);
    $('#ppo-comparison').innerHTML = Object.entries(strategies).map(([name, metrics]) => `<div class="result-row"><span>${escapeHtml(name)}</span><strong>${money(metrics.final_portfolio_value)} | ${percent(metrics.cumulative_return)} | Sharpe ${number(metrics.sharpe_ratio)}</strong></div>`).join('');
    $('#ppo-weights').innerHTML = Object.entries(ppo.ending_weights || {}).sort((a, b) => b[1] - a[1]).map(([ticker, weight]) => `<div class="result-row"><span>${escapeHtml(ticker)}</span><strong>${percent(weight)}</strong></div>`).join('') || '<span class="empty-inline">No final allocation was returned.</span>';
    $('#ppo-scenario').textContent = `Scenario ${data.scenario_id} | seed ${data.scenario_seed} | ${data.steps} decisions`;
    $('#ppo-validation-results').hidden = false;
    $('#overview-nav').textContent = money(ppo.final_portfolio_value); $('#overview-return').textContent = percent(ppo.cumulative_return); $('#overview-sharpe').textContent = number(ppo.sharpe_ratio); $('#overview-drawdown').textContent = percent(ppo.maximum_drawdown);
    $('#overview-source').textContent = `Synthetic evaluation: ${data.scenario_id}`;
    $('#agent-summary').innerHTML = `<div class="summary-row"><span>PPO vs equal-weight NAV</span><b>${money(ppo.final_portfolio_value)} / ${money(strategies['Equal-weight']?.final_portfolio_value)}</b></div><div class="summary-row"><span>Transaction costs</span><b>${money(ppo.total_transaction_costs)}</b></div><div class="summary-row"><span>Evaluation decisions</span><b>${data.steps}</b></div>`;
    $('#rebalancing-summary').innerHTML = `<div class="summary-row"><span>One-way turnover</span><b>${number(ppo.portfolio_turnover)}</b></div><div class="summary-row"><span>Final allocation</span><b>${Object.keys(ppo.ending_weights || {}).length} assets</b></div>`;
    message.textContent = 'Saved PPO validation results loaded.';
  } catch (error) { message.className = 'form-message error'; message.textContent = error.message; } finally { button.disabled = false; }
}

$('#load-ppo-validation').addEventListener('click', loadPpoValidation);
$('#load-overview').addEventListener('click', loadPpoValidation);

async function pollTraining(runId) {
  const job = await request(`/api/ppo/runs/${encodeURIComponent(runId)}`);
  if (job.state === 'running') { $('#ppo-validation-message').textContent = 'Synthetic PPO training is running. Keep the server open; results load automatically after completion.'; window.setTimeout(() => pollTraining(runId).catch(trainingError), 2000); return; }
  $('#train-synthetic-ppo').disabled = false; $('#train-synthetic-ppo').textContent = 'Train synthetic PPO';
  if (job.state === 'completed') { selectedPpoRunId = runId; $('#ppo-validation-message').textContent = 'Training completed. Loading validation results...'; loadPpoValidation(); return; }
  throw new Error(job.log_tail || 'Synthetic PPO training did not complete.');
}
function trainingError(error) { $('#train-synthetic-ppo').disabled = false; $('#train-synthetic-ppo').textContent = 'Train synthetic PPO'; $('#ppo-validation-message').className = 'form-message error'; $('#ppo-validation-message').textContent = error.message || 'Synthetic PPO training failed.'; }
$('#train-synthetic-ppo').addEventListener('click', async () => { const button = $('#train-synthetic-ppo'); button.disabled = true; button.textContent = 'Starting...'; $('#ppo-validation-message').textContent = 'Creating an isolated synthetic-only PPO experiment...'; try { const job = await request('/api/ppo/runs', { method: 'POST' }); selectedPpoRunId = job.run_id; button.textContent = 'Training...'; await pollTraining(job.run_id); } catch (error) { trainingError(error); } });

$('#synthetic-form').addEventListener('submit', async (event) => { event.preventDefault(); const message = $('#synthetic-message'); message.className = 'form-message'; message.textContent = 'Generating scenario...'; try { const seed = encodeURIComponent($('#synthetic-seed').value); const sessions = encodeURIComponent($('#synthetic-sessions').value); const data = await request(`/api/synthetic/preview?seed=${seed}&sessions=${sessions}`); $('#synthetic-result').className = 'result-list'; $('#synthetic-result').innerHTML = [['Seed', data.seed], ['Sessions', data.sessions], ['Bull / bear / sideways', `${data.regime_counts.bull} / ${data.regime_counts.bear} / ${data.regime_counts.sideways}`], ['Benchmark growth', `${data.benchmark_growth}x`], ['VIX start / end', `${data.vix_start} / ${data.vix_end}`], ['Assets', data.assets.join(', ')]].map(([label, value]) => `<div class="result-row"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`).join(''); message.textContent = 'Synthetic scenario generated in memory. No model was trained.'; } catch (error) { message.className = 'form-message error'; message.textContent = error.message; } });

async function refreshDashboard() {
  $('#api-status').textContent = 'Refreshing'; $('#api-status').classList.remove('online');
  try { await refreshStorage(); const status = await request('/api/status'); $('#api-status').textContent = `API online - ${status.mode}`; $('#api-status').classList.add('online'); } catch (error) { $('#api-status').textContent = 'API unavailable'; }
}
$('#refresh-dashboard').addEventListener('click', refreshDashboard); $('#refresh-storage').addEventListener('click', refreshStorage); refreshDashboard();
