# Phase 5.9 — Repository Data Origin & License Audit

**Project:** RL-Powered Investment Portfolio Rebalancer  
**Scope:** Existing daily files in [data/fmp_daily/](../data/fmp_daily/), their Git/repository context, and evidence of data rights  
**Budget constraint:** ₹0  
**Decision:** **NO-GO** for PPO training, backtesting presented as rights-cleared research, or dashboard display until both data rights and financial-data quality are established.

## Executive summary

The Git remote points to `https://github.com/AI4Finance-Foundation/FinRL-Trading.git`. The local checkout is on `master` at `4409abe925c904e570be78ebfb5e77ac3491dff8`; its history is not shallow. The required CSVs in [data/fmp_daily/](../data/fmp_daily/) are ignored by Git and do not occur in the local commit history or the locally available `v1.0.0` tag. The tracked archive [data/finrl_trading.7z](../data/finrl_trading.7z) contains only `finrl_trading.db`, not the price CSVs. Therefore, a standard checkout of the recorded Git tree does not account for those eight CSVs.

The repository does contain a Yahoo Finance download workflow: its README describes automatic download, and `deploy.sh` calls `yfinance.download` with `auto_adjust=False`, then writes raw OHLCV fields to `data/fmp_daily/`. The script includes the required symbols. This establishes that the repository can generate similarly named files from Yahoo, but it does **not** establish that the existing CSVs came from that workflow. In particular, the checked-in script writes no `adj_close` column, whereas the existing files have one. The local Yahoo price-comparison results are a clue, not proof of origin. FMP code and directory naming likewise do not prove these specific files came from FMP.

The repository’s Apache-2.0 license covers software, not automatically the underlying market data. No data-specific license, provider attribution, acquisition record, or permission for local retention, university research, RL training, faculty demonstration, public dashboard display, derived-output publication, or redistribution was found. The data origin, usage rights, and adjustment quality are separate unresolved questions.

## 1. Git origin and history

| Item | Local evidence |
|---|---|
| Remote URL | `https://github.com/AI4Finance-Foundation/FinRL-Trading.git` |
| Current branch | `master` |
| Current commit | `4409abe925c904e570be78ebfb5e77ac3491dff8` |
| History | Local history is not shallow; earliest commit visible locally is dated 2020-05-12. |
| Local tag | `v1.0.0` |
| GitHub release notes | Not present in local Git metadata; no external GitHub access was performed. |
| Submodules | No `.gitmodules` or submodule entries were found. |
| LFS | The `.gitattributes` LFS rule applies to `data/finrl_trading.db`; no required price CSV is LFS-tracked. |

The repository root [LICENSE](../LICENSE) is Apache License 2.0. Local Git history can establish the code and tracked paths, but it cannot establish the acquisition history of files that Git excludes.

## 2. Dataset inclusion and download workflow

The root [.gitignore](../.gitignore) rule `/data/*` excludes local data beneath `data/`, including [data/fmp_daily/](../data/fmp_daily/). The eight price CSVs are untracked/ignored and no matching price-file paths appear in the local commit history or available tag tree. This means they were not delivered as tracked files by an ordinary checkout of the recorded tree. It does not show who or what placed them in the working directory.

The repository does contain a relevant workflow:

* [README.md](../README.md) describes `data/` as runtime data storage, says the `{SYMBOL}_daily.csv` files belong under [data/fmp_daily/](../data/fmp_daily/), and describes [deploy.sh](../deploy.sh) as downloading data. It identifies Yahoo Finance as the free/default option and FMP as an optional/recommended paid option in its examples.
* [deploy.sh](../deploy.sh), introduced in commit `8bd28d097b1831b9febee965375a3c21471872d2` (2026-03-24), defaults to [data/fmp_daily/](../data/fmp_daily/), calls `yf.download(..., auto_adjust=False)`, and writes date, raw open/high/low/close, and volume. Its configured symbols include `^GSPC`, `^VIX`, `SPY`, `QQQ`, plus configured symbols.
* The existing CSVs additionally contain `adj_close`. The current `deploy.sh` output selection does not write that field. Thus the script provides a plausible Yahoo acquisition route but does not, as currently checked in, fully explain the existing files’ schema.
* [src/data/data_fetcher.py](../src/data/data_fetcher.py) contains FMP historical-price endpoint code that maps `adjClose` to `adj_close`. This proves that an FMP-capable implementation exists in the repository, not that it generated the ignored CSVs.
* The tracked archive [data/finrl_trading.7z](../data/finrl_trading.7z) was listed read-only; it contains `finrl_trading.db` only, not the required CSVs.

The owner-provided statement that the CSVs were already present when the repository was cloned and were not personally downloaded from FMP or Yahoo is recorded as owner testimony; it is not independently verified by Git. Possibilities such as a pre-populated working directory, a prior local script run, or files copied from elsewhere remain unproven.

### What is established about origin

* **Established:** These files are present in the current local working tree and excluded from ordinary Git tracking.
* **Established:** The repository provides a Yahoo/yfinance script capable of writing raw OHLCV files to the same folder.
* **Established:** FMP fetching code also exists.
* **Not established:** Which provider, person, script, package, or date produced any of the eight existing files.
* **Not established:** Whether their content was later transformed, combined, or copied from another location.

The existing Yahoo comparison artifact reports exact raw-close matches and very small adjusted-close differences for the six ETF files over the compared period. That numerical similarity is not provider attribution: multiple sources may share upstream data, values may have been copied, and comparison does not document acquisition. No equivalent benchmark origin proof was identified. See [the Phase 5.8 report](./phase_5_8_corporate_action_evidence_recovery.md) and the existing [price comparison artifact](../data/yahoo_corporate_actions/price_comparison.json).

## 3. Eight-symbol provenance matrix

The date ranges and row counts below are from the prior Phase 5.6 repository inventory. Each file is a per-symbol CSV with date and price columns, including raw OHLCV and `adj_close`; no embedded provider, retrieval timestamp, endpoint, query parameters, provider-version, or license metadata was identified.

| Symbol | Local file | Rows and date coverage | Git tracking | Provider proven? | Provenance confidence | Rights evidence |
|---|---|---:|---|---|---|---|
| SPY | [`SPY_daily.csv`](../data/fmp_daily/SPY_daily.csv) | 2,479; 2017-01-03–2026-11-11 | Ignored; absent from history | No | Low; Yahoo comparison is a clue only | None found |
| QQQ | [`QQQ_daily.csv`](../data/fmp_daily/QQQ_daily.csv) | 2,479; 2017-01-03–2026-11-11 | Ignored; absent from history | No | Low; Yahoo comparison is a clue only | None found |
| IEF | [`IEF_daily.csv`](../data/fmp_daily/IEF_daily.csv) | 2,479; 2017-01-03–2026-11-11 | Ignored; absent from history | No | Low; Yahoo comparison is a clue only | None found |
| GLD | [`GLD_daily.csv`](../data/fmp_daily/GLD_daily.csv) | 2,479; 2017-01-03–2026-11-11 | Ignored; absent from history | No | Low; Yahoo comparison is a clue only | None found |
| XLU | [`XLU_daily.csv`](../data/fmp_daily/XLU_daily.csv) | 2,479; 2017-01-03–2026-11-11 | Ignored; absent from history | No | Low; Yahoo comparison is a clue only | None found |
| XLV | [`XLV_daily.csv`](../data/fmp_daily/XLV_daily.csv) | 2,479; 2017-01-03–2026-11-11 | Ignored; absent from history | No | Low; Yahoo comparison is a clue only | None found |
| ^GSPC | [`^GSPC_daily.csv`](../data/fmp_daily/%5EGSPC_daily.csv) | 2,479; 2017-01-03–2026-11-11 | Ignored; absent from history | No | Low; Yahoo workflow supports possibility, not proof | None found |
| ^VIX | [`^VIX_daily.csv`](../data/fmp_daily/%5EVIX_daily.csv) | 2,482; through 2026-11-12; earliest date 2017-01-03 | Ignored; absent from history | No | Low; Yahoo workflow supports possibility, not proof | None found |

The `fmp_daily` directory name is not provenance evidence. Likewise, the presence of a symbol in a downloader’s configuration does not prove that the local file was produced by it.

## 4. Licensing and permissions

| Permission/use | Evidence found for these price files | Status |
|---|---|---|
| Software use | Root project license is Apache-2.0 | Applies to licensed software; not sufficient for data rights |
| Market-data license | No data-specific license or provider terms attached to the files | Unverified |
| Local retention | No acquisition receipt or terms permitting storage found | Unverified |
| Non-commercial university research | No explicit data permission found | Unverified |
| RL/model training | No explicit permission found | Unverified |
| Faculty review/demo | No explicit permission found | Unverified |
| Public dashboard/display | No explicit permission found | Unverified |
| Publication of derived outputs | No explicit permission found | Unverified |
| Raw-data redistribution | No explicit permission found | Unverified; do not redistribute |
| Attribution obligation | No source-specific attribution record found | Unknown |

The README’s educational/research framing describes the software’s intended context; it is not a market-data grant. If the files came from Yahoo Finance, the repository does not include evidence of permission for automated collection or the contemplated display/use. If they came from FMP or another provider, the provider’s applicable license and acquisition plan are also not recorded. Neither hypothesis can be used to claim permission.

## 5. Financial adjustment quality (separate from origin and rights)

The existing files contain raw OHLCV and an `adj_close` field, not adjusted OHLC. The repository’s current Yahoo download script uses `auto_adjust=False` and writes raw OHLCV only. FMP code can map a provider field named `adjClose` to `adj_close`, but neither code path establishes which method produced the local field or how its history was revised.

Earlier audits found 106 issuer-backed dividend records for SPY, XLU, and XLV; coverage for capital-gain distributions, special distributions, return of capital, and splits remains incomplete or unverified. Phase 5.8’s factor/date comparisons are corroboration only, not proof that all events are present or that a consistent total-return methodology was applied. Do not treat adjusted close as adjusted OHLC or reconstruct adjusted OHLC by scaling raw OHLC without independently verified methodology. Do not silently mix raw and adjusted prices or double-count corporate actions.

Historical revision policy, survivorship/look-ahead effects, and whether benchmark and ETF series were acquired under compatible terms are not established by the repository artifacts.

## 6. Final feasibility decision

### **NO-GO**

Do not use the existing files as rights-cleared inputs for the planned historical PPO training, backtest, faculty demonstration, or public dashboard at this time. The blockers are independent:

1. **Software license:** Apache-2.0 is present for the project software.
2. **Data license:** No permission for retention, academic training, demonstrations, display, or derived publication is evidenced.
3. **Provenance:** Git and local documentation do not establish the original source or acquisition event for the ignored CSVs.
4. **Financial quality:** Adjusted OHLC and complete, reconciled corporate-action methodology are not established.

This is a no-go for project use pending evidence, not a finding that a particular provider definitely supplied the files or that a particular provider’s terms were violated.

## 7. Recommended next action

1. Preserve the CSVs unchanged and do not distribute them or use them in training/display until resolved.
2. Request from whoever supplied or initially populated the clone the original acquisition records, source URL/provider, retrieval date, exact endpoint/query settings, license/plan at acquisition, and any transformation steps. Request checksums or original response snapshots if available.
3. If repository history is insufficient, obtain evidence from the original packaging/clone event (for example, an archive or clone bundle with a verifiable manifest and file checksums). The ignored files cannot be recovered from this local Git history.
4. Obtain written permission from the proven provider covering local retention, university research, RL model training, faculty demonstrations, dashboard audience/display, derived outputs, and any redistribution/attribution requirements.
5. Separately validate adjusted-price and corporate-action methodology for all six ETFs and confirm benchmark coverage before any research run.

The local remote is `https://github.com/AI4Finance-Foundation/FinRL-Trading.git`. No external GitHub access was performed; GitHub release metadata is not included in the local clone. External repository evidence would not, by itself, prove the provenance of ignored local files unless it contains a checksum-matched source artifact or acquisition record.

## 8. Scope and safety record

This audit used local repository history, configuration, source, prior reports, and existing artifacts. No external market-data API was called, no data was downloaded or modified, no purchase, training, or trade was performed, and `.env` was not read. The only created file for this phase is this report.
