const sample=[{ticker:'SPY',weight:35},{ticker:'QQQ',weight:25},{ticker:'TLT',weight:20},{ticker:'GLD',weight:20}],colors=['#e16372','#93b9ff','#e3bd73','#8ed195','#c899e8'];const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
function showView(name){$$('.view').forEach(v=>v.classList.toggle('active',v.id===name));$$('#navigation button').forEach(b=>b.classList.toggle('active',b.dataset.view===name));history.replaceState(null,'',`#${name}`)}$$('#navigation button').forEach(b=>b.onclick=()=>showView(b.dataset.view));showView(location.hash.slice(1)||'overview');
$$('[data-tabs] button').forEach(button=>button.onclick=()=>{const tab=button.dataset.tab;$$('[data-tabs] button').forEach(b=>b.classList.toggle('active',b===button));$$('[data-tab-panel]').forEach(p=>p.classList.toggle('active',p.dataset.tabPanel===tab))});
async function loadStats(){try{const d=await (await fetch('/api/dashboard/stats')).json();$('#price-records').textContent=d.price_records??0;$('#stored-files').textContent=d.total_files??0;$('#storage-used').textContent=`${Number(d.total_size_mb||0).toFixed(1)} MB`;$('#storage-json').textContent=JSON.stringify(d,null,2)}catch{$('#storage-json').textContent='Storage statistics are unavailable.'}}loadStats();
function addRow(a={ticker:'',weight:''}){const r=document.createElement('div');r.className='allocation-row';r.innerHTML=`<input aria-label="Ticker" maxlength="10" value="${a.ticker}" placeholder="Ticker"><input aria-label="Weight percentage" type="number" min="0" max="100" step="0.1" value="${a.weight}" placeholder="Weight %"><button class="remove-button" type="button">×</button>`;r.querySelectorAll('input').forEach(i=>i.oninput=renderAllocation);r.querySelector('button').onclick=()=>{r.remove();renderAllocation()};$('#allocation-rows').append(r)}
function assets(){return $$('.allocation-row').map(r=>({ticker:r.querySelector('input').value.trim().toUpperCase(),weight:Number(r.querySelectorAll('input')[1].value)||0})).filter(a=>a.ticker)}function renderAllocation(){const a=assets(),total=a.reduce((s,x)=>s+x.weight,0);let at=0;const stops=a.map((x,i)=>{const from=at;at+=total?x.weight/total*360:0;return `${colors[i%colors.length]} ${from}deg ${at}deg`});$('#allocation-chart').style.background=stops.length?`conic-gradient(${stops.join(',')})`:'#333';$('#allocation-legend').innerHTML=a.map((x,i)=>`<div class="legend-item"><span><i class="legend-dot" style="background:${colors[i%colors.length]}"></i>${x.ticker}</span><b>${x.weight.toFixed(1)}%</b></div>`).join('')}sample.forEach(addRow);renderAllocation();$('#add-asset').onclick=()=>addRow();
$('#allocation-form').onsubmit=async e=>{e.preventDefault();const a=assets(),total=a.reduce((s,x)=>s+x.weight,0),m=$('#form-message');if(!a.length||Math.abs(total-100)>.01){m.className='form-message error';m.textContent='Weights must total exactly 100%.';return}try{const r=await fetch('/portfolio/allocation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tickers:a.map(x=>x.ticker),weights:Object.fromEntries(a.map(x=>[x.ticker,x.weight/100])),start_date:'2023-01-01',end_date:'2024-01-01'})}),d=await r.json();if(!r.ok)throw Error(d.detail);m.className='form-message';m.textContent=`Validated ${d.total_weight.toFixed(2)} total portfolio weight. No trades were placed.`}catch(err){m.className='form-message error';m.textContent=err.message}};
$('#synthetic-form').onsubmit=async e=>{e.preventDefault();const m=$('#synthetic-message'),seed=$('#synthetic-seed').value,sessions=$('#synthetic-sessions').value;m.textContent='Generating reproducible scenario…';try{const r=await fetch(`/api/synthetic/preview?seed=${seed}&sessions=${sessions}`),d=await r.json();if(!r.ok)throw Error(d.detail);$('#synthetic-result').className='result-list';$('#synthetic-result').innerHTML=`<div><span>Seed</span><strong>${d.seed}</strong></div><div><span>Sessions</span><strong>${d.sessions}</strong></div><div><span>Bull / bear / sideways</span><strong>${d.regime_counts.bull} / ${d.regime_counts.bear} / ${d.regime_counts.sideways}</strong></div><div><span>Benchmark growth</span><strong>${d.benchmark_growth}×</strong></div><div><span>VIX start → end</span><strong>${d.vix_start} → ${d.vix_end}</strong></div><div><span>Asset universe</span><strong>${d.assets.join(', ')}</strong></div>`;m.textContent='Scenario generated in memory; no model was trained.'}catch(err){m.className='form-message error';m.textContent=`Could not generate scenario: ${err.message}`}};
fetch('/api/status').then(r=>r.json()).then(d=>{$('#api-status').textContent=`API online · ${d.mode} mode`}).catch(()=>{$('#api-status').textContent='API unavailable';$('#api-status').style.color='#ff9b9b'});


// ==========================================
// PPO VALIDATION RESULTS
// ==========================================

const ppoButton = document.querySelector('#load-ppo-validation');

if (ppoButton) {
    const ppoMessage = document.querySelector('#ppo-validation-message');
    const ppoResults = document.querySelector('#ppo-validation-results');

    const formatNumber = value =>
        Number(value).toLocaleString('en-IN', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });

    const formatPercent = value =>
        `${(Number(value) * 100).toFixed(2)}%`;

    const formatRatio = value =>
        Number(value).toFixed(3);

    function addPpoResultRow(container, label, value) {
        const row = document.createElement('div');
        const name = document.createElement('span');
        const result = document.createElement('strong');

        row.className = 'result-row';
        name.textContent = label;
        result.textContent = value;

        row.append(name, result);
        container.appendChild(row);
    }

    ppoButton.addEventListener('click', async () => {
        ppoButton.disabled = true;
        ppoButton.textContent = 'Loading...';

        ppoMessage.className = 'form-message';
        ppoMessage.textContent =
            'Loading the saved PPO model and evaluating its validation scenario...';

        try {
            const response = await fetch('/api/ppo/validation');
            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail || 'Could not load PPO validation results.'
                );
            }

            const evaluation = data.evaluation;
            const strategies = evaluation?.strategies;
            if (
                data.source_type !== 'synthetic' ||
                data.synthetic !== true ||
                !evaluation ||
                evaluation.evaluation_type !== 'synthetic-only' ||
                !strategies
            ) {
                throw new Error('Unexpected validation response.');
            }

            const requiredStrategies = [
                'Synthetic PPO',
                'SPY buy-and-hold',
                'Equal-weight',
                'Monthly rebalancing',
                'Threshold rebalancing'
            ];
            if (requiredStrategies.some(name => !strategies[name])) {
                throw new Error('One or more evaluation strategies are missing.');
            }

            const ppo = strategies['Synthetic PPO'];
            document.querySelector('#ppo-nav').textContent =
                formatNumber(ppo.final_portfolio_value);

            document.querySelector('#equal-nav').textContent =
                formatNumber(strategies['Equal-weight'].final_portfolio_value);

            document.querySelector('#buyhold-nav').textContent =
                formatNumber(strategies['SPY buy-and-hold'].final_portfolio_value);

            document.querySelector('#monthly-nav').textContent =
                formatNumber(strategies['Monthly rebalancing'].final_portfolio_value);

            document.querySelector('#threshold-nav').textContent =
                formatNumber(strategies['Threshold rebalancing'].final_portfolio_value);

            document.querySelector('#ppo-cost').textContent =
                formatNumber(ppo.total_transaction_costs);

            const comparison =
                document.querySelector('#ppo-comparison');

            comparison.replaceChildren();

            const metricRows = [
                ['Final portfolio value', 'final_portfolio_value', formatNumber],
                ['Cumulative return', 'cumulative_return', formatPercent],
                ['Annualized return', 'annualized_return', formatPercent],
                ['Annualized volatility', 'annualized_volatility', formatPercent],
                ['Sharpe ratio', 'sharpe_ratio', formatRatio],
                ['Sortino ratio', 'sortino_ratio', formatRatio],
                ['Maximum drawdown', 'maximum_drawdown', formatPercent],
                ['Transaction costs', 'total_transaction_costs', formatNumber],
                ['One-way turnover', 'portfolio_turnover', formatRatio]
            ];

            Object.entries(strategies).forEach(([name, metrics]) => {
                metricRows.forEach(([label, key, format]) => {
                    addPpoResultRow(
                        comparison,
                        `${name} · ${label}`,
                        format(metrics[key])
                    );
                });
            });

            const weights =
                document.querySelector('#ppo-weights');

            weights.replaceChildren();

            Object.entries(ppo.ending_weights)
                .sort((a, b) => b[1] - a[1])
                .forEach(([ticker, weight]) => {
                    addPpoResultRow(
                        weights,
                        ticker,
                        `${(Number(weight) * 100).toFixed(2)}%`
                    );
                });

            document.querySelector('#ppo-scenario').textContent =
                `Scenario: ${data.scenario_id} | ` +
                `Seed: ${data.scenario_seed} | ` +
                `Decisions: ${data.steps} | ` +
                `Sessions: ${evaluation.period.evaluation_sessions} | ` +
                `Initial capital: ${formatNumber(evaluation.cost_assumptions.initial_capital)}`;

            ppoResults.hidden = false;

            ppoMessage.className = 'form-message';
            ppoMessage.textContent =
                'PPO and benchmark evaluation completed successfully.';

        } catch (error) {
            ppoMessage.className = 'form-message error';
            ppoMessage.textContent = error.message;
            ppoResults.hidden = true;

        } finally {
            ppoButton.disabled = false;
            ppoButton.textContent = 'Load validation results';
        }
    });
}
