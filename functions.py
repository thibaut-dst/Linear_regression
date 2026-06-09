"""Functions extracted from Linear_regression.ipynb."""
from __future__ import annotations

def setup_environment():
    """Notebook cell 2."""
    global C, ElasticNet, KFold, Lasso, LinearRegression, Ridge, StandardScaler, cross_val_score, go, make_subplots, minimize, np, pd, sm, sms, stats, warnings
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats
    from scipy.optimize import minimize
    import statsmodels.api as sm
    import statsmodels.stats.api as sms
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    import warnings
    warnings.filterwarnings('ignore')
    
    np.random.seed(42)
    
    # Consistent color palette
    C = dict(data='#636EFA', fit='#EF553B', true='#00CC96', alt='#AB63FA', neutral='gray')

def capm_demo():
    """Notebook cell 6."""
    global eps, fig, market_excess, n, stock_excess, true_alpha, true_beta
    # ─── Synthetic CAPM data (used throughout sections 0-2) ──────────────────────
    n = 200
    
    # Monthly market excess returns: annualised μ=7%, σ=15%
    market_excess = np.random.normal(0.07/12, 0.15/np.sqrt(12), n)
    
    true_alpha = 0.002   # 20 bps / month alpha
    true_beta  = 1.30    # aggressive stock (amplifies market moves)
    eps        = np.random.normal(0, 0.04, n)   # idiosyncratic vol
    
    stock_excess = true_alpha + true_beta * market_excess + eps
    
    print(f"Market — mean: {market_excess.mean():.4f}/mo, std: {market_excess.std():.4f}")
    print(f"Stock  — mean: {stock_excess.mean():.4f}/mo, std: {stock_excess.std():.4f}")
    print(f"Correlation  : {np.corrcoef(market_excess, stock_excess)[0,1]:.4f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=market_excess, y=stock_excess, mode='markers',
        marker=dict(color=C['data'], opacity=0.5, size=5), name='Monthly returns'))
    fig.update_layout(
        title='CAPM: Stock Excess Return vs Market Excess Return',
        xaxis_title='Market Excess Return (rₘ − rᶠ)',
        yaxis_title='Stock Excess Return (rᵢ − rᶠ)',
        width=700, height=450)
    fig.show()

def ols_minimization_demo():
    """Notebook cell 12."""
    global COL_BAD, COL_OLS, b0_bad, b0_opt, b1_bad, b1_opt, fig, i, n_il, rss_bad, rss_opt, x_il, x_line, xb_il, y_il, yb_il, yhat_bad, yhat_opt
    # ─── Illustrative chart: what OLS minimises ───────────────────────────────────
    # Small toy dataset for visual clarity
    np.random.seed(7)
    n_il = 14
    x_il = np.linspace(1, 9, n_il)
    y_il = 2 + 1.3*x_il + np.random.normal(0, 1.2, n_il)
    
    # OLS optimal line
    xb_il = x_il.mean(); yb_il = y_il.mean()
    b1_opt = np.sum((x_il - xb_il) * (y_il - yb_il)) / np.sum((x_il - xb_il)**2)
    b0_opt = yb_il - b1_opt * xb_il
    
    # A plausible-looking but suboptimal line
    b0_bad, b1_bad = 0.5, 1.8
    
    yhat_opt = b0_opt + b1_opt * x_il
    yhat_bad = b0_bad + b1_bad * x_il
    rss_opt  = np.sum((y_il - yhat_opt)**2)
    rss_bad  = np.sum((y_il - yhat_bad)**2)
    
    x_line = np.linspace(0, 10, 200)
    
    # Colours
    COL_BAD = '#AB63FA'   # purple  — candidate line
    COL_OLS = '#FFA15A'   # orange  — OLS line
    
    fig = go.Figure()
    
    # ── Vertical bars: candidate line (purple) ────────────────────────────────────
    for i in range(n_il):
        fig.add_shape(type='line',
            x0=x_il[i], y0=y_il[i], x1=x_il[i], y1=yhat_bad[i],
            line=dict(color=COL_BAD, width=2))
    
    # ── Vertical bars: OLS line (orange) — drawn on top ──────────────────────────
    for i in range(n_il):
        fig.add_shape(type='line',
            x0=x_il[i], y0=y_il[i], x1=x_il[i], y1=yhat_opt[i],
            line=dict(color=COL_OLS, width=1.5))
    
    # ── Legend proxies for the bars (shapes have no legend entry by default) ──────
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color=COL_BAD, width=2),
        name=f'Residuals eᵢ — candidate line  (Σeᵢ² = {rss_bad:.1f})'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color=COL_OLS, width=2),
        name=f'Residuals eᵢ — OLS line  (Σeᵢ² = {rss_opt:.1f})  ← minimum'))
    
    # ── Fitted lines ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=x_line, y=b0_bad + b1_bad * x_line, mode='lines',
        line=dict(color=COL_BAD, width=2, dash='dash'), name='Candidate line'))
    fig.add_trace(go.Scatter(x=x_line, y=b0_opt + b1_opt * x_line, mode='lines',
        line=dict(color=COL_OLS, width=2, dash='dash'), name='OLS line'))
    
    # ── Data points ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=x_il, y=y_il, mode='markers',
        marker=dict(color='black', size=7, opacity=0.85), name='Data'))
    
    fig.update_layout(
        title=(f'OLS minimises Σeᵢ²  —  same chart, two lines, two sets of residuals<br>'
               f'Purple bars² sum to <b>{rss_bad:.1f}</b>  |  '
               f'Orange bars² sum to <b>{rss_opt:.1f}</b>  ← OLS achieves the minimum'),
        xaxis_title='x', yaxis_title='y',
        xaxis_range=[0, 10],
        height=490, width=820,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='lightgray', borderwidth=1))
    fig.show()
    
    print(f"Candidate line RSS = {rss_bad:.2f}")
    print(f"OLS line      RSS = {rss_opt:.2f}  (no other line can go lower)")

def ols_from_scratch():
    """Notebook cell 14."""
    global beta0_ols, beta1_ols, fig, i, lr, residuals, x, x_bar, x_line, y, y_bar, y_hat
    # ─── OLS from scratch ─────────────────────────────────────────────────────────
    x, y = market_excess, stock_excess
    
    x_bar, y_bar = x.mean(), y.mean()
    beta1_ols = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar)**2)
    beta0_ols = y_bar - beta1_ols * x_bar
    y_hat     = beta0_ols + beta1_ols * x
    residuals = y - y_hat
    
    print("=== OLS (from scratch) ===")
    print(f"  β₀ (alpha) = {beta0_ols:.6f}  [true: {true_alpha}]")
    print(f"  β₁ (beta)  = {beta1_ols:.6f}  [true: {true_beta}]")
    
    # ─── Validate against sklearn + plot ──────────────────────────────────────────
    lr = LinearRegression().fit(x.reshape(-1, 1), y)
    print("=== sklearn validation ===")
    print(f"  β₀ = {lr.intercept_:.6f}  β₁ = {lr.coef_[0]:.6f}  (should match above ✓)")
    
    x_line = np.linspace(x.min(), x.max(), 200)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
        marker=dict(color=C['data'], opacity=0.4, size=5), name='Monthly returns'))
    fig.add_trace(go.Scatter(x=x_line, y=beta0_ols + beta1_ols * x_line, mode='lines',
        line=dict(color=C['fit'], width=2.5),
        name=f'OLS: α={beta0_ols:.4f}, β={beta1_ols:.4f}'))
    fig.add_trace(go.Scatter(x=x_line, y=true_alpha + true_beta * x_line, mode='lines',
        line=dict(color=C['true'], width=2, dash='dash'),
        name=f'True line (from DGP): α={true_alpha}, β={true_beta}'))
    
    # Draw a sample of residuals
    for i in range(0, n, 25):
        fig.add_shape(type='line', x0=x[i], y0=y[i], x1=x[i], y1=y_hat[i],
            line=dict(color='gray', width=1, dash='dot'))
    
    fig.update_layout(
        title='OLS: minimising Residual Sum of Squares<br>Gray lines = residuals eᵢ = yᵢ − ŷᵢ',
        xaxis_title='Market Excess Return', yaxis_title='Stock Excess Return',
        width=800, height=500, legend=dict(x=0.02, y=0.98))
    fig.show()

def mle_demo():
    """Notebook cell 18."""
    global b0_mle, b1_grid, b1_mle, fig, ll_vals, log_s2_mle, neg_log_likelihood, res, rss_val, s2_mle, s2_mle_fm, s2_ols_fm
    # ─── MLE via numerical optimisation ───────────────────────────────────────────
    def neg_log_likelihood(params, x, y):
        b0, b1, log_s2 = params
        s2 = np.exp(log_s2)          # log-transform ensures s2 > 0
        nll = (len(y)/2)*np.log(2*np.pi*s2) + np.sum((y - b0 - b1*x)**2) / (2*s2)
        return nll
    
    res = minimize(neg_log_likelihood, x0=[0, 1, np.log(0.01)], args=(x, y), method='BFGS')
    b0_mle, b1_mle, log_s2_mle = res.x
    s2_mle = np.exp(log_s2_mle)
    
    rss_val    = np.sum(residuals**2)
    s2_mle_fm  = rss_val / n       # MLE formula
    s2_ols_fm  = rss_val / (n - 2) # OLS (unbiased)
    
    print("=== Coefficients (should be identical) ===")
    print(f"  OLS: β₀={beta0_ols:.6f}, β₁={beta1_ols:.6f}")
    print(f"  MLE: β₀={b0_mle:.6f}, β₁={b1_mle:.6f}")
    print(f"\n=== σ² estimates ===")
    print(f"  MLE  (biased, /n)  : {s2_mle_fm:.6f}")
    print(f"  OLS  (unbiased,/n-2): {s2_ols_fm:.6f}")
    print(f"  Ratio n/(n-2) = {n/(n-2):.4f}  (MLE underestimates σ²)")
    
    # ─── Log-likelihood profile for β₁ ───────────────────────────────────────────
    b1_grid = np.linspace(beta1_ols - 0.6, beta1_ols + 0.6, 300)
    ll_vals  = [-neg_log_likelihood([beta0_ols, b, log_s2_mle], x, y) for b in b1_grid]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b1_grid, y=ll_vals, mode='lines',
        line=dict(color=C['fit'], width=2), name='ℓ(β₁)'))
    fig.add_vline(x=beta1_ols, line=dict(color=C['fit'], dash='dash'),
        annotation_text=f'MLE=OLS: {beta1_ols:.4f}', annotation_position='top right')
    fig.add_vline(x=true_beta, line=dict(color=C['true'], dash='dot'),
        annotation_text=f'True β₁={true_beta}', annotation_position='top left')
    fig.update_layout(
        title='Log-likelihood profile ℓ(β₁)  — β₀ and σ² fixed at OLS estimates',
        xaxis_title='β₁', yaxis_title='Log-likelihood  ℓ',
        width=700, height=400)
    fig.show()

def sst_ssr_rss_demo():
    """Notebook cell 21."""
    global COL_RSS, COL_SSR, COL_SST, b0_d, b1_d, col, fig_d, i, n_d, x_d, xb_d, xl_d, y_d, yb_d, yhat_d
    # ─── SST / SSR / RSS — geometric decomposition ───────────────────────────────
    # Small toy dataset so bars are clearly visible
    np.random.seed(7)
    n_d  = 14
    x_d  = np.linspace(1, 9, n_d)
    y_d  = 2 + 1.3 * x_d + np.random.normal(0, 1.2, n_d)
    
    xb_d   = x_d.mean();  yb_d = y_d.mean()
    b1_d   = np.sum((x_d - xb_d) * (y_d - yb_d)) / np.sum((x_d - xb_d)**2)
    b0_d   = yb_d - b1_d * xb_d
    yhat_d = b0_d + b1_d * x_d
    xl_d   = np.linspace(0, 10, 200)
    
    COL_SST = '#636EFA'   # blue  — total bar
    COL_SSR = '#00CC96'   # green — explained part
    COL_RSS = '#EF553B'   # red   — residual part
    
    # No subplot_titles here — we add them as annotations below each panel instead
    fig_d = make_subplots(rows=1, cols=2)
    
    # ── Panel 1: SST bars (yᵢ to ȳ) ─────────────────────────────────────────────
    for i in range(n_d):
        fig_d.add_shape(type='line',
            x0=x_d[i], y0=y_d[i], x1=x_d[i], y1=yb_d,
            line=dict(color=COL_SST, width=3), row=1, col=1)
    
    fig_d.add_trace(go.Scatter(x=[0, 10], y=[yb_d, yb_d], mode='lines',
        line=dict(color='black', width=1.5, dash='dash'),
        name='ȳ  (mean)', showlegend=True), row=1, col=1)
    fig_d.add_trace(go.Scatter(x=x_d, y=y_d, mode='markers',
        marker=dict(color='black', size=7), name='data', showlegend=False), row=1, col=1)
    fig_d.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color=COL_SST, width=3),
        name='yᵢ − ȳ  (SST bar — total deviation)'), row=1, col=1)
    
    # ── Panel 2: SSR bars (ȳ to ŷᵢ) + RSS bars (ŷᵢ to yᵢ) ──────────────────────
    for i in range(n_d):
        # SSR: how far the model's prediction moves from the mean (explained part)
        fig_d.add_shape(type='line',
            x0=x_d[i], y0=yb_d, x1=x_d[i], y1=yhat_d[i],
            line=dict(color=COL_SSR, width=3), row=1, col=2)
        # RSS: how far the actual point sits from the model's prediction (unexplained)
        fig_d.add_shape(type='line',
            x0=x_d[i], y0=yhat_d[i], x1=x_d[i], y1=y_d[i],
            line=dict(color=COL_RSS, width=3), row=1, col=2)
    
    fig_d.add_trace(go.Scatter(x=[0, 10], y=[yb_d, yb_d], mode='lines',
        line=dict(color='black', width=1.5, dash='dash'),
        name='ȳ  (mean)', showlegend=False), row=1, col=2)
    fig_d.add_trace(go.Scatter(x=xl_d, y=b0_d + b1_d * xl_d, mode='lines',
        line=dict(color='gray', width=2),
        name='ŷ = β₀ + β₁x  (OLS line)'), row=1, col=2)
    fig_d.add_trace(go.Scatter(x=x_d, y=y_d, mode='markers',
        marker=dict(color='black', size=7), name='data', showlegend=False), row=1, col=2)
    fig_d.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color=COL_SSR, width=3),
        name='ŷᵢ − ȳ  (SSR bar — model explains this)'), row=1, col=2)
    fig_d.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color=COL_RSS, width=3),
        name='yᵢ − ŷᵢ  (RSS bar — residual, unexplained)'), row=1, col=2)
    
    for col in [1, 2]:
        fig_d.update_xaxes(title_text='x', range=[0, 10], row=1, col=col)
        fig_d.update_yaxes(title_text='y', row=1, col=col)
    
    fig_d.update_layout(
        height=510, width=1020,
        margin=dict(b=110),   # extra bottom space for the captions
        title=(
            'For every point: (yᵢ − ȳ) = (ŷᵢ − ȳ) + (yᵢ − ŷᵢ)  →  SST = SSR + RSS<br>'
            'R² = SSR/SST = share of the blue bar that is green'
        ),
        # Panel captions placed below each subplot
        annotations=[
            dict(
                text='SST = Σ(yᵢ − ȳ)²<br>total variation around the mean',
                x=0.225, y=-0.12, xref='paper', yref='paper',
                xanchor='center', yanchor='top',
                showarrow=False, font=dict(size=13)
            ),
            dict(
                text='SST = SSR + RSS<br>decomposition per point',
                x=0.775, y=-0.12, xref='paper', yref='paper',
                xanchor='center', yanchor='top',
                showarrow=False, font=dict(size=13)
            ),
        ]
    )
    fig_d.show()

def residual_diagnostics_demo():
    """Notebook cell 25."""
    global _, fig, intercept, osm, osr, r2, rss, sl_smooth, slope, sm_lowess, ssr, sst
    # ─── R² and residual diagnostics ──────────────────────────────────────────────
    sst = np.sum((y - y_bar)**2)
    ssr = np.sum((y_hat - y_bar)**2)
    rss = np.sum(residuals**2)
    r2  = 1 - rss / sst
    
    print(f"SST = {sst:.6f}")
    print(f"SSR = {ssr:.6f}")
    print(f"RSS = {rss:.6f}   (SSR + RSS = {ssr+rss:.6f} ✓)")
    print(f"\nR² = {r2:.4f}  → {r2*100:.1f}% of variance explained")
    print(f"\nResidual properties (must hold by construction):")
    print(f"  Σeᵢ   = {residuals.sum():.2e}  ≈ 0 ✓")
    print(f"  Σxᵢeᵢ = {np.dot(x, residuals):.2e}  ≈ 0 ✓")
    
    # ─── Diagnostic plots ─────────────────────────────────────────────────────────
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals)
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Residuals vs Fitted', 'Q-Q Plot (normality)',
        'Residuals vs Index (autocorrelation)', 'Scale-Location (homoscedasticity)'])
    
    # 1. Residuals vs Fitted
    fig.add_trace(go.Scatter(x=y_hat, y=residuals, mode='markers',
        marker=dict(color=C['data'], opacity=0.5, size=4), showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=1)
    
    # 2. Q-Q
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers',
        marker=dict(color=C['data'], opacity=0.5, size=4), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=osm, y=np.array(osm)*slope + intercept, mode='lines',
        line=dict(color='red', width=1.5), showlegend=False), row=1, col=2)
    
    # 3. Residuals vs Index
    fig.add_trace(go.Scatter(x=np.arange(n), y=residuals, mode='markers+lines',
        marker=dict(color=C['data'], opacity=0.4, size=3),
        line=dict(color=C['data'], width=0.8), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=2, col=1)
    
    # 4. Scale-Location
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    sl_smooth = sm_lowess(np.sqrt(np.abs(residuals)), y_hat, frac=0.5)
    fig.add_trace(go.Scatter(x=y_hat, y=np.sqrt(np.abs(residuals)), mode='markers',
        marker=dict(color=C['data'], opacity=0.5, size=4), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=sl_smooth[:, 0], y=sl_smooth[:, 1], mode='lines',
        line=dict(color='red', width=1.5), showlegend=False), row=2, col=2)
    
    fig.update_xaxes(title_text='Fitted ŷ', row=1, col=1)
    fig.update_yaxes(title_text='Residual e', row=1, col=1)
    fig.update_xaxes(title_text='Theoretical quantiles', row=1, col=2)
    fig.update_yaxes(title_text='Sample quantiles', row=1, col=2)
    fig.update_xaxes(title_text='Observation index', row=2, col=1)
    fig.update_yaxes(title_text='Residual e', row=2, col=1)
    fig.update_xaxes(title_text='Fitted ŷ', row=2, col=2)
    fig.update_yaxes(title_text='√|e|', row=2, col=2)
    
    fig.update_layout(title=f'Residual Diagnostics  (R² = {r2:.4f})',
        height=650, width=1000, showlegend=False)
    fig.show()

def fama_french_two_factor_demo():
    """Notebook cell 29."""
    global MKT_g, SMB_g, X_ff, Y_plane, beta_hat, eps_ff, est, fig_plane, i, lab, lr_ff, mkt, mkt_g, n_ff, ret_ff, smb, smb_g, tru, true_alpha_ff, true_betas, xs_r, yhat_ff, ys_r, zs_r
    # ─── Fama-French 2-factor data (MKT + SMB) ───────────────────────────────────
    # Two regressors so the fit lives on a plane we can plot in 3D
    np.random.seed(42)
    n_ff = 300
    
    mkt = np.random.normal(0.07/12, 0.15/np.sqrt(12), n_ff)  # market excess return
    smb = np.random.normal(0.02/12, 0.10/np.sqrt(12), n_ff)  # size factor (Small Minus Big)
    
    true_alpha_ff = 0.001
    true_betas    = np.array([1.2, 0.5])       # β_MKT, β_SMB
    eps_ff        = np.random.normal(0, 0.035, n_ff)
    
    ret_ff = true_alpha_ff + true_betas[0]*mkt + true_betas[1]*smb + eps_ff
    
    # Design matrix X: [1 | mkt | smb]  — shape (n, 3)
    X_ff = np.column_stack([np.ones(n_ff), mkt, smb])
    print(f"X shape: {X_ff.shape}  (n={n_ff}, k=3: intercept + 2 regressors)")
    
    # ─── OLS: β̂ = (X'X)⁻¹ X'y ────────────────────────────────────────────────────
    beta_hat = np.linalg.solve(X_ff.T @ X_ff, X_ff.T @ ret_ff)
    
    print("\n=== FF2 OLS Estimates ===")
    for lab, est, tru in zip(['alpha', 'beta_MKT', 'beta_SMB'], beta_hat, [true_alpha_ff]+list(true_betas)):
        print(f"  {lab:10}  est={est:+.5f}   true={tru:+.5f}")
    
    lr_ff = LinearRegression().fit(X_ff[:, 1:], ret_ff)
    print(f"\nsklearn: intercept={lr_ff.intercept_:.5f}  coefs={lr_ff.coef_}  ✓")
    
    # ─── 3D visualisation: data cloud + regression hyperplane ─────────────────────
    # With 2 predictors the fitted surface y = β₀ + β₁·mkt + β₂·smb is a flat plane
    mkt_g = np.linspace(mkt.min(), mkt.max(), 25)
    smb_g = np.linspace(smb.min(), smb.max(), 25)
    MKT_g, SMB_g = np.meshgrid(mkt_g, smb_g)
    Y_plane = beta_hat[0] + beta_hat[1]*MKT_g + beta_hat[2]*SMB_g
    
    # Pack residual lines into a single Scatter3d trace using None separators
    yhat_ff = X_ff @ beta_hat
    xs_r, ys_r, zs_r = [], [], []
    for i in range(0, n_ff, 4):   # every 4th point keeps the chart readable
        xs_r += [mkt[i], mkt[i], None]
        ys_r += [smb[i], smb[i], None]
        zs_r += [ret_ff[i], yhat_ff[i], None]
    
    fig_plane = go.Figure()
    
    fig_plane.add_trace(go.Surface(
        x=MKT_g, y=SMB_g, z=Y_plane,
        colorscale=[[0, 'rgba(99,110,250,0.25)'], [1, 'rgba(99,110,250,0.55)']],
        showscale=False, opacity=0.6,
        name='Regression plane  ŷ = β̂₀ + β̂₁·mkt + β̂₂·smb'))
    
    fig_plane.add_trace(go.Scatter3d(
        x=mkt, y=smb, z=ret_ff, mode='markers',
        marker=dict(size=2.5, color='black', opacity=0.55),
        name='Observations'))
    
    fig_plane.add_trace(go.Scatter3d(
        x=xs_r, y=ys_r, z=zs_r, mode='lines',
        line=dict(color='#EF553B', width=1.5),
        name='Residuals  eᵢ = yᵢ − ŷᵢ'))
    
    fig_plane.update_layout(
        scene=dict(
            xaxis_title='MKT (market excess return)',
            yaxis_title='SMB (size factor)',
            zaxis_title='Portfolio return',
            camera=dict(eye=dict(x=1.6, y=-1.8, z=0.8))
        ),
        title="OLS regression plane in ℝ³ — generalises the 2D fitted line to 2 regressors<br>"
              "β̂ = (X'X)⁻¹X'y minimises the total squared length of the red vertical bars",
        width=860, height=580, showlegend=True,
        legend=dict(x=0.01, y=0.95))
    fig_plane.show()

def leverage_cooks_distance_demo():
    """Notebook cell 32."""
    global X_sm_ff, cooks_d, fig, hi_lev, infl_idx, influence, k_ff, leverage, n_ff_, sm_ff, std_resid, threshold
    # ─── Hat matrix, leverage, Cook's distance ────────────────────────────────────
    X_sm_ff = sm.add_constant(np.column_stack([mkt, smb]))
    sm_ff   = sm.OLS(ret_ff, X_sm_ff).fit()
    influence = sm_ff.get_influence()
    
    leverage    = influence.hat_matrix_diag
    cooks_d     = influence.cooks_distance[0]
    std_resid   = influence.resid_studentized_internal
    
    k_ff  = 3   # number of parameters (intercept + mkt + smb)
    n_ff_ = len(ret_ff)
    
    print(f"Hat matrix diagonal sum  = {leverage.sum():.4f}  (should equal k={k_ff} ✓)")
    print(f"Leverage range: [{leverage.min():.4f}, {leverage.max():.4f}]")
    print(f"Mean leverage  = {leverage.mean():.4f}  (should equal k/n = {k_ff/n_ff_:.4f} ✓)")
    print(f"\nHigh-leverage observations (h > 2k/n = {2*k_ff/n_ff_:.4f}):")
    hi_lev = np.where(leverage > 2*k_ff/n_ff_)[0]
    print(f"  {len(hi_lev)} observations ({len(hi_lev)/n_ff_*100:.1f}%)")
    
    # ─── Influence plot: leverage vs studentised residuals ─────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=leverage, y=std_resid,
        mode='markers',
        marker=dict(size=6, color=cooks_d, colorscale='RdYlGn_r',
                    showscale=True, colorbar=dict(title="Cook's D"),
                    opacity=0.7),
        text=[f"i={i}<br>h={h:.3f}<br>D={d:.4f}" for i,h,d in zip(range(n_ff_), leverage, cooks_d)],
        hoverinfo='text', name='observations'))
    
    # Cook's D contour (D = 4/n)
    threshold = 4 / n_ff_
    fig.add_hline(y=2,  line=dict(color='orange', dash='dash'), annotation_text='|t|=2')
    fig.add_hline(y=-2, line=dict(color='orange', dash='dash'))
    fig.add_vline(x=2*k_ff/n_ff_, line=dict(color='purple', dash='dash'),
                  annotation_text=f'2k/n={2*k_ff/n_ff_:.3f}')
    
    fig.update_layout(
        title="Influence Plot: Leverage vs Studentised Residuals<br>Color = Cook's Distance",
        xaxis_title='Leverage  hᵢᵢ',
        yaxis_title='Studentised residual',
        width=800, height=500)
    fig.show()
    
    print(f"\nMost influential observations (Cook's D > {threshold:.4f}):")
    infl_idx = np.where(cooks_d > threshold)[0]
    print(f"  {len(infl_idx)} observations  ({len(infl_idx)/n_ff_*100:.1f}%)")

def adjusted_r2_demo():
    """Notebook cell 35."""
    global X_cur, Xbase, aics, ar2s, bics, col, extra, fig, hml, i, ks, noise, r, r2s, regression_stats, results
    # ─── Adjusted R² and information criteria ─────────────────────────────────────
    def regression_stats(X, y):
        n_, k_ = X.shape
        b = np.linalg.solve(X.T @ X, X.T @ y)
        yhat = X @ b
        rss_ = np.sum((y - yhat)**2)
        sst_ = np.sum((y - y.mean())**2)
        r2_  = 1 - rss_/sst_
        adj_r2 = 1 - (1 - r2_)*(n_ - 1)/(n_ - k_)
        s2_ = rss_ / (n_ - k_)
        ll  = -n_/2*np.log(2*np.pi*s2_) - rss_/(2*s2_)
        aic = 2*k_ - 2*ll
        bic = k_*np.log(n_) - 2*ll
        return dict(k=k_, r2=r2_, adj_r2=adj_r2, aic=aic, bic=bic, rss=rss_)
    
    # Compare models of increasing complexity (adding random noise predictors)
    np.random.seed(99)
    noise = np.random.normal(0, 1, (n_ff, 5))
    results = []
    if 'hml' not in globals():
        # This section compares a base 3-factor design to noisier variants.
        hml = np.random.normal(0.03/12, 0.12/np.sqrt(12), n_ff)
    Xbase = np.column_stack([np.ones(n_ff), mkt, smb, hml])  # k=4
    
    for extra in range(6):
        if extra == 0:
            X_cur = Xbase
        else:
            X_cur = np.column_stack([Xbase, noise[:, :extra]])
        results.append(regression_stats(X_cur, ret_ff))
    
    ks    = [r['k']     for r in results]
    r2s   = [r['r2']    for r in results]
    ar2s  = [r['adj_r2']for r in results]
    aics  = [r['aic']   for r in results]
    bics  = [r['bic']   for r in results]
    
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['R² vs Adjusted R²  (adding noise predictors)',
                        'AIC and BIC  (adding noise predictors)'])
    
    fig.add_trace(go.Scatter(x=ks, y=r2s,  mode='lines+markers',
        line=dict(color=C['fit']), name='R²'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=ar2s, mode='lines+markers',
        line=dict(color=C['true']), name='Adj R²'), row=1, col=1)
    fig.add_vline(x=4, line=dict(color='gray', dash='dot'),
        annotation_text='True model (k=4)', row=1, col=1)
    
    fig.add_trace(go.Scatter(x=ks, y=aics, mode='lines+markers',
        line=dict(color=C['data']), name='AIC'), row=1, col=2)
    fig.add_trace(go.Scatter(x=ks, y=bics, mode='lines+markers',
        line=dict(color=C['alt']), name='BIC'), row=1, col=2)
    fig.add_vline(x=4, line=dict(color='gray', dash='dot'),
        annotation_text='True model (k=4)', row=1, col=2)
    
    for col in [1,2]:
        fig.update_xaxes(title_text='k (number of parameters)', row=1, col=col)
    fig.update_yaxes(title_text='Metric value', row=1, col=1)
    fig.update_yaxes(title_text='Information criterion', row=1, col=2)
    fig.update_layout(height=420, width=1000,
        title='R² always increases; Adj R², AIC, BIC correctly penalise noise predictors')
    fig.show()
    
    print("k  |  R²     | Adj R²  |  AIC    |  BIC")
    print("-" * 45)
    for i, r in enumerate(results):
        print(f"{r['k']}  | {r['r2']:.5f} | {r['adj_r2']:.5f} | {r['aic']:.2f} | {r['bic']:.2f}")

def multicollinearity_vif_demo():
    """Notebook cell 38."""
    global X_mc, Xb, Xc, _, bc, betas_clean, betas_collinear, bk, fig, flag, idx, j, n_mc, name, tru, v, vif, vifs, x1, x2, x2c, x3, y_b, y_c, y_mc
    # ─── Multicollinearity and VIF ────────────────────────────────────────────────
    np.random.seed(7)
    n_mc = 200
    
    # Create a collinear dataset
    x1 = np.random.normal(0, 1, n_mc)
    x2 = 0.97*x1 + np.random.normal(0, np.sqrt(1 - 0.97**2), n_mc)  # corr ≈ 0.97
    x3 = np.random.normal(0, 1, n_mc)   # independent
    y_mc = 1 + 2*x1 + 3*x2 + 1.5*x3 + np.random.normal(0, 0.5, n_mc)
    
    print(f"Correlation x1-x2: {np.corrcoef(x1, x2)[0,1]:.4f}  (nearly 1!)")
    print(f"Correlation x1-x3: {np.corrcoef(x1, x3)[0,1]:.4f}")
    
    # VIF: regress each predictor on the others
    def vif(X_df):
        vifs = {}
        cols = list(range(X_df.shape[1]))
        for j in cols:
            X_other = np.delete(X_df, j, axis=1)
            X_other = np.column_stack([np.ones(len(X_other)), X_other])
            b = np.linalg.solve(X_other.T @ X_other, X_other.T @ X_df[:, j])
            yhat_j = X_other @ b
            r2_j = 1 - np.sum((X_df[:,j] - yhat_j)**2)/np.sum((X_df[:,j] - X_df[:,j].mean())**2)
            vifs[f'x{j+1}'] = 1 / (1 - r2_j)
        return vifs
    
    X_mc = np.column_stack([x1, x2, x3])
    vifs = vif(X_mc)
    print("\n=== VIF scores ===")
    for name, v in vifs.items():
        flag = " ← SEVERE" if v > 10 else (" ← check" if v > 5 else "")
        print(f"  {name}: {v:.2f}{flag}")
    
    # Show coefficient instability under collinearity
    # Compare estimates across 100 bootstrap samples
    betas_collinear  = []
    betas_clean      = []
    for _ in range(200):
        idx = np.random.choice(n_mc, n_mc, replace=True)
        Xb  = np.column_stack([np.ones(n_mc), x1[idx], x2[idx], x3[idx]])
        y_b = y_mc[idx]
        betas_collinear.append(np.linalg.solve(Xb.T@Xb, Xb.T@y_b)[1:])  # drop intercept
    
        # Clean version: replace x2 with uncorrelated x2_clean
        x2c = np.random.normal(0, 1, n_mc)
        y_c = 1 + 2*x1[idx] + 3*x2c[idx] + 1.5*x3[idx] + np.random.normal(0, 0.5, n_mc)
        Xc  = np.column_stack([np.ones(n_mc), x1[idx], x2c[idx], x3[idx]])
        betas_clean.append(np.linalg.solve(Xc.T@Xc, Xc.T@y_c)[1:])
    
    bc = np.array(betas_collinear)
    bk = np.array(betas_clean)
    
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=['β₁ distribution', 'β₂ distribution', 'β₃ distribution'])
    for j, (tru, name) in enumerate(zip([2,3,1.5], ['x₁','x₂','x₃'])):
        fig.add_trace(go.Histogram(x=bc[:,j], opacity=0.6,
            marker_color=C['fit'], name=f'Collinear {name}',
            nbinsx=40, showlegend=(j==0)), row=1, col=j+1)
        fig.add_trace(go.Histogram(x=bk[:,j], opacity=0.6,
            marker_color=C['true'], name=f'Clean {name}',
            nbinsx=40, showlegend=(j==0)), row=1, col=j+1)
        fig.add_vline(x=tru, line=dict(color='black', dash='dash'), row=1, col=j+1)
    
    fig.update_layout(barmode='overlay', height=400, width=1000,
        title='Bootstrap distributions of β̂ — collinear (red) vs clean (green)<br>'
              'Black dash = true value. Collinearity inflates variance of β₁, β₂.')
    fig.show()

def ovb_demo():
    """Notebook cell 41."""
    global X_long, Xl, _, b_l, b_long, b_longs, b_s, b_short, b_shorts, b_xz, corr_xz, fig, gamma, idx, n_ovb, ovb_pred, reps, true_b, x_ovb, xb, y_ovb, yb, z_ovb, zb
    # ─── Omitted Variable Bias simulation ─────────────────────────────────────────
    np.random.seed(42)
    n_ovb  = 500
    
    # True DGP: y depends on x and z (but we'll omit z)
    x_ovb  = np.random.normal(0, 1, n_ovb)
    z_ovb  = 0.7*x_ovb + np.random.normal(0, np.sqrt(1 - 0.49), n_ovb)  # corr(x,z)=0.7
    corr_xz = np.corrcoef(x_ovb, z_ovb)[0,1]
    
    true_b = 0.5   # true effect of x
    gamma  = 1.0   # true effect of z (omitted)
    y_ovb  = 1 + true_b*x_ovb + gamma*z_ovb + np.random.normal(0, 0.5, n_ovb)
    
    # Short regression (omitting z)
    b_short = np.sum((x_ovb - x_ovb.mean())*(y_ovb - y_ovb.mean())) / np.sum((x_ovb - x_ovb.mean())**2)
    
    # Long regression (including z)
    X_long = np.column_stack([np.ones(n_ovb), x_ovb, z_ovb])
    b_long = np.linalg.solve(X_long.T@X_long, X_long.T@y_ovb)
    
    # Analytical OVB prediction
    b_xz = corr_xz * (np.std(z_ovb)/np.std(x_ovb))   # coef from x-on-z regression = Cov(x,z)/Var(x)
    ovb_pred = b_xz * gamma
    
    print(f"=== Omitted Variable Bias ===")
    print(f"True β  = {true_b:.4f}")
    print(f"Short β = {b_short:.4f}  (omitting z)")
    print(f"Long  β = {b_long[1]:.4f}  (including z)")
    print(f"\nOVB (empirical) = {b_short - true_b:.4f}")
    print(f"OVB (predicted) = {ovb_pred:.4f}  [= corr(x,z)·std(z)/std(x)·γ = {b_xz:.3f}·{gamma}]")
    print(f"\nCorr(x,z) = {corr_xz:.4f}, γ = {gamma:.4f} → both positive → upward bias ✓")
    
    # Visualise: sampling distribution of β̂
    reps = 500
    b_shorts, b_longs = [], []
    for _ in range(reps):
        idx = np.random.choice(n_ovb, n_ovb, replace=True)
        xb, zb, yb = x_ovb[idx], z_ovb[idx], y_ovb[idx]
        b_s = np.cov(xb, yb)[0,1]/np.var(xb)
        Xl  = np.column_stack([np.ones(n_ovb), xb, zb])
        b_l = np.linalg.solve(Xl.T@Xl, Xl.T@yb)[1]
        b_shorts.append(b_s);  b_longs.append(b_l)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=b_shorts, opacity=0.6, marker_color=C['fit'],
        name='Short reg (omits z)', nbinsx=40))
    fig.add_trace(go.Histogram(x=b_longs, opacity=0.6, marker_color=C['true'],
        name='Long reg (includes z)', nbinsx=40))
    fig.add_vline(x=true_b, line=dict(color='black', width=2, dash='dash'),
        annotation_text=f'True β={true_b}', annotation_position='top right')
    fig.update_layout(barmode='overlay',
        title='OVB: Short regression (red) is systematically biased upward',
        xaxis_title='β̂ estimate', yaxis_title='Frequency',
        width=800, height=450)
    fig.show()

def gm_visuals_demo():
    """Notebook cell 45."""
    global Eg, Xg, Zg, _lowess, col, e_g, e_line, eps_a4, eps_a5, eps_norm, eps_t3, fig_a, n_a, peak_z, sig_env, sm_a4, specs, x_a, x_env, x_g, xl, yl
    # ─── A4 / A5 / A6 — geometric illustrations ──────────────────────────────────
    np.random.seed(42)
    n_a = 120
    x_a = np.linspace(0.5, 5.5, n_a)
    
    # 3D: p(ε|x) = N(0,1) ridge — simultaneously illustrates A4 (mean=0),
    #              A5 (constant width), A6 (Gaussian bell shape)
    x_g = np.linspace(0.5, 5.5, 35)
    e_g = np.linspace(-3.5, 3.5, 70)
    Xg, Eg = np.meshgrid(x_g, e_g)
    Zg = stats.norm.pdf(Eg, loc=0, scale=1)   # identical slice at every x
    
    # A4 violated: E[ε|x] = 0.6x − 1.8  (omitted variable / endogeneity)
    eps_a4 = 0.6 * x_a - 1.8 + np.random.normal(0, 0.5, n_a)
    
    # A5 violated: σ(x) = 0.15 + 0.4x   (heteroscedasticity)
    eps_a5 = np.random.normal(0, 0.15 + 0.4 * x_a, n_a)
    
    # A6 violated: t₃ errors — unit variance so comparison is shape-only
    eps_norm = np.random.normal(0, 1, 1500)
    eps_t3   = stats.t.rvs(df=3, scale=1/np.sqrt(3), size=1500)
    
    specs = [
        [{'type': 'scene', 'colspan': 3}, None, None],
        [{'type': 'xy'},                  {'type': 'xy'}, {'type': 'xy'}]
    ]
    fig_a = make_subplots(
        rows=2, cols=3, specs=specs,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.08
    )
    
    # ── 3D surface: constant Gaussian ridge ──────────────────────────────────────
    fig_a.add_trace(go.Surface(
        x=Xg, y=Eg, z=Zg,
        colorscale=[[0, 'rgba(99,110,250,0.02)'], [1, 'rgba(99,110,250,0.9)']],
        showscale=False, opacity=0.85, name='p(ε|x)'
    ), row=1, col=1)
    
    # Red ridge line at ε = 0 marking the conditional mean (A4)
    peak_z = stats.norm.pdf(0, 0, 1)
    fig_a.add_trace(go.Scatter3d(
        x=x_g, y=np.zeros_like(x_g), z=np.full_like(x_g, peak_z),
        mode='lines', line=dict(color='red', width=6),
        name='E[ε|x] = 0  (A4: zero mean for all x)'
    ), row=1, col=1)
    
    # ── A4 panel: drifting conditional mean ──────────────────────────────────────
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    sm_a4 = _lowess(eps_a4, x_a, frac=0.45)
    
    fig_a.add_trace(go.Scatter(
        x=x_a, y=eps_a4, mode='markers',
        marker=dict(color='#EF553B', size=4, opacity=0.65), showlegend=False
    ), row=2, col=1)
    fig_a.add_trace(go.Scatter(
        x=sm_a4[:, 0], y=sm_a4[:, 1], mode='lines',
        line=dict(color='black', width=2),
        name='E[ε|x] (drifts away from 0)', showlegend=False
    ), row=2, col=1)
    fig_a.add_trace(go.Scatter(
        x=[x_a.min(), x_a.max()], y=[0, 0], mode='lines',
        line=dict(color='green', dash='dash', width=1.5), showlegend=False
    ), row=2, col=1)
    
    # ── A5 panel: widening spread with ±2σ(x) envelope ───────────────────────────
    fig_a.add_trace(go.Scatter(
        x=x_a, y=eps_a5, mode='markers',
        marker=dict(color='#AB63FA', size=4, opacity=0.65), showlegend=False
    ), row=2, col=2)
    fig_a.add_trace(go.Scatter(
        x=[x_a.min(), x_a.max()], y=[0, 0], mode='lines',
        line=dict(color='green', dash='dash', width=1.5), showlegend=False
    ), row=2, col=2)
    
    x_env = np.linspace(0.5, 5.5, 100)
    sig_env = 0.15 + 0.4 * x_env
    fig_a.add_trace(go.Scatter(
        x=np.concatenate([x_env, x_env[::-1]]),
        y=np.concatenate([2 * sig_env, -2 * sig_env[::-1]]),
        fill='toself', fillcolor='rgba(171,99,250,0.13)',
        line=dict(color='rgba(171,99,250,0.5)', width=1.5),
        name='±2σ(x) envelope', showlegend=False
    ), row=2, col=2)
    
    # ── A6 panel: histogram + density overlay ────────────────────────────────────
    fig_a.add_trace(go.Histogram(
        x=eps_norm, histnorm='probability density',
        name='N(0,1)  (A6 holds)',
        marker_color='rgba(99,110,250,0.55)', xbins=dict(size=0.25)
    ), row=2, col=3)
    fig_a.add_trace(go.Histogram(
        x=eps_t3, histnorm='probability density',
        name='t₃ / √3  (A6 violated — fat tails)',
        marker_color='rgba(239,85,59,0.50)', xbins=dict(size=0.25)
    ), row=2, col=3)
    e_line = np.linspace(-5.5, 5.5, 300)
    fig_a.add_trace(go.Scatter(
        x=e_line, y=stats.norm.pdf(e_line),
        mode='lines', line=dict(color='#636EFA', width=2), showlegend=False
    ), row=2, col=3)
    fig_a.add_trace(go.Scatter(
        x=e_line, y=stats.t.pdf(e_line, df=3, scale=1/np.sqrt(3)),
        mode='lines', line=dict(color='#EF553B', width=2), showlegend=False
    ), row=2, col=3)
    
    # ── Axis labels ───────────────────────────────────────────────────────────────
    for col, (xl, yl) in enumerate([('x', 'ε'), ('x', 'ε'), ('ε', 'density')], start=1):
        fig_a.update_xaxes(title_text=xl, row=2, col=col)
        fig_a.update_yaxes(title_text=yl, row=2, col=col)
    fig_a.update_xaxes(range=[-6, 6], row=2, col=3)
    
    fig_a.update_layout(
        scene=dict(
            xaxis_title='x (predictor)',
            yaxis_title='ε (error)',
            zaxis_title='p(ε|x)',
            camera=dict(eye=dict(x=1.8, y=-1.6, z=0.9))
        ),
        height=780, width=1060,
        margin=dict(b=120),
        title='Gauss-Markov Assumptions A4, A5, A6 — geometric intuition',
        barmode='overlay',
        legend=dict(x=0.01, y=0.47, bgcolor='rgba(255,255,255,0.85)', borderwidth=1),
        annotations=[
            # 3D panel caption — below the 3D, centred
            dict(text='A4 + A5 + A6 all hold<br>p(ε|x) is a constant Gaussian ridge centred at ε = 0',
                 x=0.5, y=0.44, xref='paper', yref='paper',
                 xanchor='center', yanchor='top', showarrow=False, font=dict(size=12)),
            # Row-2 captions — below each 2D panel
            dict(text='A4 violated<br>E[ε|x] ≠ 0  (e.g. omitted variable)',
                 x=0.13, y=-0.04, xref='paper', yref='paper',
                 xanchor='center', yanchor='top', showarrow=False, font=dict(size=12)),
            dict(text='A5 violated<br>Var(ε|x) grows with x  (heteroscedasticity)',
                 x=0.5,  y=-0.04, xref='paper', yref='paper',
                 xanchor='center', yanchor='top', showarrow=False, font=dict(size=12)),
            dict(text='A6 violated<br>t₃ vs Normal  (same σ, different shape)',
                 x=0.87, y=-0.04, xref='paper', yref='paper',
                 xanchor='center', yanchor='top', showarrow=False, font=dict(size=12)),
        ]
    )
    fig_a.show()

def assumption_violations_demo():
    """Notebook cell 48."""
    global P, c, c_col, e_ar, e_e, e_h, e_nl, eps_ar, eps_h, fig, fit_and_resid, n_v, rho, t, x_v, xsort, y_ar, y_e, y_h, y_nl, yhat_e, yhat_h, yhat_nl, z
    # ─── Simulate and plot assumption violations ──────────────────────────────────
    np.random.seed(0)
    n_v = 150
    x_v = np.linspace(0, 10, n_v)
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'A1 violated: Non-linearity',
        'A4 violated: Endogeneity (omitted variable)',
        'A5a violated: Heteroscedasticity',
        'A5b violated: Autocorrelation (AR(1) errors)'])
    
    def fit_and_resid(x, y):
        b1 = np.sum((x - x.mean())*(y - y.mean())) / np.sum((x - x.mean())**2)
        b0 = y.mean() - b1*x.mean()
        return y - (b0 + b1*x)
    
    # A1: true relationship is quadratic
    y_nl = 1 + 0.5*x_v + 0.12*x_v**2 + np.random.normal(0, 1.5, n_v)
    e_nl = fit_and_resid(x_v, y_nl)
    yhat_nl = y_nl - e_nl
    fig.add_trace(go.Scatter(x=yhat_nl, y=e_nl, mode='markers',
        marker=dict(color=C['data'], opacity=0.5, size=4), showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=1)
    # Smoother to show the pattern
    from numpy.polynomial import polynomial as P
    c = P.polyfit(yhat_nl, e_nl, 2)
    xsort = np.sort(yhat_nl)
    fig.add_trace(go.Scatter(x=xsort, y=P.polyval(xsort, c), mode='lines',
        line=dict(color='red', width=2), showlegend=False), row=1, col=1)
    
    # A4: omitted variable z correlated with x
    z   = 0.7*x_v + np.random.normal(0, 1, n_v)
    y_e = 1 + 0.5*x_v + 0.9*z + np.random.normal(0, 0.5, n_v)
    e_e = fit_and_resid(x_v, y_e)
    yhat_e = y_e - e_e
    fig.add_trace(go.Scatter(x=yhat_e, y=e_e, mode='markers',
        marker=dict(color=C['alt'], opacity=0.5, size=4), showlegend=False), row=1, col=2)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=2)
    
    # A5a: heteroscedasticity — variance grows with x
    eps_h = np.random.normal(0, 0.08*(1 + x_v), n_v)
    y_h   = 1 + 0.5*x_v + eps_h
    e_h   = fit_and_resid(x_v, y_h)
    yhat_h = y_h - e_h
    fig.add_trace(go.Scatter(x=yhat_h, y=e_h, mode='markers',
        marker=dict(color=C['fit'], opacity=0.5, size=4), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=2, col=1)
    
    # A5b: autocorrelation — AR(1) errors with rho=0.85
    rho = 0.85
    eps_ar = np.zeros(n_v)
    for t in range(1, n_v):
        eps_ar[t] = rho*eps_ar[t-1] + np.random.normal(0, 1)
    y_ar = 1 + 0.5*x_v + eps_ar
    e_ar = fit_and_resid(x_v, y_ar)
    fig.add_trace(go.Scatter(x=np.arange(n_v), y=e_ar, mode='lines+markers',
        marker=dict(color=C['true'], size=3, opacity=0.5),
        line=dict(color=C['true'], width=1), showlegend=False), row=2, col=2)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=2, col=2)
    
    for c_col in [1, 2]:
        fig.update_xaxes(title_text='Fitted ŷ', row=1, col=c_col)
        fig.update_yaxes(title_text='Residual', row=1, col=c_col)
        fig.update_xaxes(title_text='Fitted ŷ' if c_col==1 else 'Index t', row=2, col=c_col)
        fig.update_yaxes(title_text='Residual', row=2, col=c_col)
    
    fig.update_layout(title='Residual plots under assumption violations',
        height=650, width=1050, showlegend=False)
    fig.show()
    
    print("Red line (row 1 left) = fitted quadratic smooth showing systematic curvature.")
    print("A4 violation: residuals correlated with x (but looks OK vs fitted — endogeneity is subtle).")
    print("A5a: fan shape — variance grows with x.")
    print("A5b: clear serial correlation — residuals cluster above/below zero in runs.")

def inference_demo():
    """Notebook cell 56."""
    global F_stat, Sxx, X_sm, ci0, ci1, p0, p1, p_F, rss_val, s2_hat, s_hat, se0, se1, sm_fit, sst_val, t0, t1, t_crit
    # ─── Inference from scratch ────────────────────────────────────────────────────
    rss_val   = np.sum(residuals**2)
    s2_hat    = rss_val / (n - 2)          # unbiased σ²
    s_hat     = np.sqrt(s2_hat)
    
    Sxx = np.sum((x - x_bar)**2)
    se1 = s_hat / np.sqrt(Sxx)                              # SE(β₁)
    se0 = s_hat * np.sqrt(np.sum(x**2) / (n * Sxx))         # SE(β₀)
    
    t1 = beta1_ols / se1
    t0 = beta0_ols / se0
    p1 = 2 * (1 - stats.t.cdf(abs(t1), df=n-2))
    p0 = 2 * (1 - stats.t.cdf(abs(t0), df=n-2))
    
    t_crit = stats.t.ppf(0.975, df=n-2)
    ci0 = (beta0_ols - t_crit*se0, beta0_ols + t_crit*se0)
    ci1 = (beta1_ols - t_crit*se1, beta1_ols + t_crit*se1)
    
    sst_val = np.sum((y - y_bar)**2)
    F_stat  = (sst_val - rss_val) / (rss_val / (n - 2))
    p_F     = 1 - stats.f.cdf(F_stat, dfn=1, dfd=n-2)
    
    print("=" * 75)
    print(f"{'':12} {'Coef':>10} {'SE':>10} {'t':>10} {'p-value':>12}  95% CI")
    print("-" * 75)
    print(f"{'β₀ (alpha)':12} {beta0_ols:>10.6f} {se0:>10.6f} {t0:>10.4f} {p0:>12.4f}  [{ci0[0]:.4f}, {ci0[1]:.4f}]")
    print(f"{'β₁ (beta)':12} {beta1_ols:>10.6f} {se1:>10.6f} {t1:>10.4f} {p1:>12.4e}  [{ci1[0]:.4f}, {ci1[1]:.4f}]")
    print("=" * 75)
    print(f"F-statistic = {F_stat:.4f},  p-value = {p_F:.2e}  (tests H₀: β₁=0)")
    print(f"σ̂  = {s_hat:.6f},  σ̂² = {s2_hat:.6f},  R² = {1-rss_val/sst_val:.4f}")
    
    print("\n--- Validation against statsmodels ---")
    X_sm   = sm.add_constant(x)
    sm_fit = sm.OLS(y, X_sm).fit()
    print(sm_fit.summary().tables[1])

def gls_wls_demo():
    """Notebook cell 59."""
    global Var_ols, Var_wls, W_fgls, W_mat, X_aux_g, X_gls, X_line_g, b_aux_g, b_fgls, b_ols_g, b_wls, e_ols, eps_gls, fig, log_e2, n_gls, omega_diag, sigma2_hat_i, sigma2_true, true_b_gls, w_fgls, w_gls, x_gls, x_line_g, y_gls
    # ─── GLS from scratch: compare OLS, WLS, GLS on a heteroscedastic DGP ─────────
    np.random.seed(42)
    n_gls = 200
    x_gls = np.linspace(1, 10, n_gls)
    
    # True DGP: Var(εᵢ) = σ²·xᵢ  (variance grows with x)
    true_b_gls  = 2.0
    sigma2_true = 0.5
    omega_diag  = x_gls                              # Ωᵢᵢ = xᵢ
    eps_gls     = np.random.normal(0, np.sqrt(sigma2_true * omega_diag))
    y_gls       = 3 + true_b_gls * x_gls + eps_gls
    
    X_gls = np.column_stack([np.ones(n_gls), x_gls])
    
    # ── OLS (ignores heteroscedasticity) ─────────────────────────────────────────
    b_ols_g = np.linalg.solve(X_gls.T@X_gls, X_gls.T@y_gls)
    
    # ── WLS / GLS (weights wᵢ = 1/Ωᵢᵢ = 1/xᵢ) ───────────────────────────────────
    w_gls  = 1.0 / omega_diag                        # true weights
    W_mat  = np.diag(w_gls)
    b_wls  = np.linalg.solve(X_gls.T@W_mat@X_gls, X_gls.T@W_mat@y_gls)
    
    # ── FGLS: estimate Ω from OLS residuals ──────────────────────────────────────
    e_ols   = y_gls - X_gls@b_ols_g
    # Regress log(eᵢ²) on log(xᵢ) to estimate the heteroscedasticity pattern
    log_e2  = np.log(e_ols**2)
    X_aux_g = np.column_stack([np.ones(n_gls), np.log(x_gls)])
    b_aux_g = np.linalg.solve(X_aux_g.T@X_aux_g, X_aux_g.T@log_e2)
    sigma2_hat_i = np.exp(X_aux_g @ b_aux_g)        # estimated σᵢ²
    w_fgls = 1.0 / sigma2_hat_i
    W_fgls = np.diag(w_fgls)
    b_fgls = np.linalg.solve(X_gls.T@W_fgls@X_gls, X_gls.T@W_fgls@y_gls)
    
    print("=== GLS comparison (true β₁ = 2.0) ===")
    print(f"  OLS:  β₁ = {b_ols_g[1]:.4f}   (unbiased but inefficient)")
    print(f"  WLS:  β₁ = {b_wls[1]:.4f}   (GLS with true Ω — efficient)")
    print(f"  FGLS: β₁ = {b_fgls[1]:.4f}   (GLS with estimated Ω)")
    
    # Compare standard errors (via variance formula: Var(β̂) = σ²(X'W X)⁻¹ for WLS)
    Var_ols  = sigma2_true * np.linalg.inv(X_gls.T@X_gls)
    Var_wls  = sigma2_true * np.linalg.inv(X_gls.T@W_mat@X_gls)
    print(f"\n=== SE of β₁ ===")
    print(f"  OLS SE: {np.sqrt(Var_ols[1,1]):.4f}   (underestimates true SE because Ω≠I)")
    print(f"  WLS SE: {np.sqrt(Var_wls[1,1]):.4f}   (correct, smaller → WLS is more efficient)")
    print(f"  WLS is more efficient by a factor of {np.sqrt(Var_ols[1,1])/np.sqrt(Var_wls[1,1]):.2f}x")
    
    # ─── Visual comparison ────────────────────────────────────────────────────────
    x_line_g = np.linspace(1, 10, 200)
    X_line_g = np.column_stack([np.ones(200), x_line_g])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_gls, y=y_gls, mode='markers',
        marker=dict(color=C['data'], opacity=0.4, size=4), name='Data'))
    fig.add_trace(go.Scatter(x=x_line_g, y=X_line_g@b_ols_g, mode='lines',
        line=dict(color=C['fit'], width=2), name=f'OLS β₁={b_ols_g[1]:.4f}'))
    fig.add_trace(go.Scatter(x=x_line_g, y=X_line_g@b_wls, mode='lines',
        line=dict(color=C['true'], width=2), name=f'WLS/GLS β₁={b_wls[1]:.4f}'))
    fig.add_trace(go.Scatter(x=x_line_g, y=3 + true_b_gls*x_line_g, mode='lines',
        line=dict(color='black', width=1.5, dash='dash'), name=f'True β₁={true_b_gls}'))
    fig.update_layout(
        title='GLS vs OLS: WLS is more efficient under heteroscedasticity<br>'
              'WLS downweights high-variance observations (large x)',
        xaxis_title='x', yaxis_title='y', width=800, height=480)
    fig.show()

def heteroscedasticity_demo():
    """Notebook cell 62."""
    global X_aux, X_h, b_aux, b_h, bp_pval, bp_stat, e2_h, e2hat, e_h, fig, n_h, r2_aux, sigma_true, sm_h, sm_h_hc, sm_wls, w_true, x_h, x_line, y_h, yhat_h
    # ─── Heteroscedasticity: simulate, detect, fix ────────────────────────────────
    np.random.seed(42)
    n_h = 250
    x_h = np.linspace(1, 10, n_h)
    
    # True DGP: variance proportional to x
    sigma_true = 0.3 * x_h   # heteroscedastic!
    y_h = 2 + 1.5*x_h + np.random.normal(0, sigma_true)
    
    X_h  = np.column_stack([np.ones(n_h), x_h])
    b_h  = np.linalg.solve(X_h.T@X_h, X_h.T@y_h)
    yhat_h = X_h @ b_h
    e_h    = y_h - yhat_h
    
    # ─── Breusch-Pagan test ────────────────────────────────────────────────────────
    e2_h = e_h**2
    X_aux = np.column_stack([np.ones(n_h), x_h])
    b_aux = np.linalg.solve(X_aux.T@X_aux, X_aux.T@e2_h)
    e2hat = X_aux @ b_aux
    r2_aux = 1 - np.sum((e2_h - e2hat)**2)/np.sum((e2_h - e2_h.mean())**2)
    bp_stat = n_h * r2_aux
    bp_pval = 1 - stats.chi2.cdf(bp_stat, df=1)
    print(f"Breusch-Pagan test:  stat={bp_stat:.4f},  p-value={bp_pval:.4e}  → {'reject H₀' if bp_pval<0.05 else 'fail to reject'} of homoscedasticity")
    
    # ─── Robust SEs (HC1) vs OLS SEs ──────────────────────────────────────────────
    sm_h     = sm.OLS(y_h, X_h).fit()
    sm_h_hc  = sm.OLS(y_h, X_h).fit(cov_type='HC1')
    
    print("\n=== β₁ standard error comparison ===")
    print(f"  OLS SE (wrong):  {sm_h.bse[1]:.6f}")
    print(f"  HC1 SE (robust): {sm_h_hc.bse[1]:.6f}")
    print(f"  t-stat OLS: {sm_h.tvalues[1]:.4f}   t-stat HC1: {sm_h_hc.tvalues[1]:.4f}")
    
    # ─── WLS: weight by 1/x (true weights) ────────────────────────────────────────
    w_true = 1.0 / x_h**2   # w_i = 1/σ_i² ∝ 1/x²
    sm_wls = sm.WLS(y_h, X_h, weights=w_true).fit()
    print(f"  WLS  SE:         {sm_wls.bse[1]:.6f}  (most efficient)")
    
    # ─── Plot ─────────────────────────────────────────────────────────────────────
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        'Data: fan-shaped variance', 'Residuals vs Fitted'])
    
    x_line = np.linspace(1, 10, 200)
    fig.add_trace(go.Scatter(x=x_h, y=y_h, mode='markers',
        marker=dict(color=C['data'], opacity=0.4, size=4), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_line, y=b_h[0]+b_h[1]*x_line, mode='lines',
        line=dict(color=C['fit'], width=2), showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=yhat_h, y=e_h, mode='markers',
        marker=dict(color=C['data'], opacity=0.4, size=4), showlegend=False), row=1, col=2)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=2)
    
    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_yaxes(title_text='y', row=1, col=1)
    fig.update_xaxes(title_text='Fitted ŷ', row=1, col=2)
    fig.update_yaxes(title_text='Residual', row=1, col=2)
    fig.update_layout(height=420, width=950,
        title='Heteroscedasticity: fan shape in residuals is the key diagnostic')
    fig.show()

def autocorrelation_demo():
    """Notebook cell 65."""
    global X_ar, acf_vals, b_ar, conf, dw, e_ar, eps_ar, fig, max_lag, n_ar, rho_ar, rho_est, sm_ar, sm_ar_nw, t, x_ar, y_ar
    # ─── Autocorrelation: simulate, detect, fix ────────────────────────────────────
    np.random.seed(42)
    n_ar = 300
    x_ar = np.linspace(0, 5, n_ar)
    
    # True DGP: AR(1) errors with ρ=0.80
    rho_ar = 0.80
    eps_ar = np.zeros(n_ar)
    for t in range(1, n_ar):
        eps_ar[t] = rho_ar*eps_ar[t-1] + np.random.normal(0, 1)
    
    y_ar  = 1 + 2*x_ar + eps_ar
    X_ar  = np.column_stack([np.ones(n_ar), x_ar])
    b_ar  = np.linalg.solve(X_ar.T@X_ar, X_ar.T@y_ar)
    e_ar  = y_ar - X_ar@b_ar
    
    # ─── Durbin-Watson statistic ──────────────────────────────────────────────────
    dw    = np.sum(np.diff(e_ar)**2) / np.sum(e_ar**2)
    rho_est = np.corrcoef(e_ar[:-1], e_ar[1:])[0,1]
    print(f"True ρ              = {rho_ar:.2f}")
    print(f"Estimated ρ (ACF 1) = {rho_est:.4f}")
    print(f"Durbin-Watson       = {dw:.4f}  (≈ 2(1-ρ) = {2*(1-rho_est):.4f})")
    print(f"  DW << 2 indicates strong positive autocorrelation ✓")
    
    # ─── OLS SE vs Newey-West SE ──────────────────────────────────────────────────
    sm_ar    = sm.OLS(y_ar, X_ar).fit()
    sm_ar_nw = sm.OLS(y_ar, X_ar).fit(cov_type='HAC', cov_kwds={'maxlags': int(n_ar**(1/3))})
    
    print("\n=== β₁ standard error comparison ===")
    print(f"  OLS SE (wrong):  {sm_ar.bse[1]:.6f}   t={sm_ar.tvalues[1]:.3f}   p={sm_ar.pvalues[1]:.4f}")
    print(f"  NW  SE (robust): {sm_ar_nw.bse[1]:.6f}   t={sm_ar_nw.tvalues[1]:.3f}   p={sm_ar_nw.pvalues[1]:.4f}")
    print(f"  SE ratio (NW/OLS): {sm_ar_nw.bse[1]/sm_ar.bse[1]:.2f}x  (OLS underestimates SEs)")
    
    # ─── ACF plot of residuals ────────────────────────────────────────────────────
    max_lag = 30
    acf_vals = [np.corrcoef(e_ar[:-lag], e_ar[lag:])[0,1] for lag in range(1, max_lag+1)]
    conf = 1.96/np.sqrt(n_ar)
    
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Residuals over time', 'ACF of residuals'])
    
    fig.add_trace(go.Scatter(x=np.arange(n_ar), y=e_ar, mode='lines',
        line=dict(color=C['data'], width=1), showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=1)
    
    fig.add_trace(go.Bar(x=list(range(1, max_lag+1)), y=acf_vals,
        marker_color=[C['fit'] if abs(v)>conf else C['data'] for v in acf_vals],
        showlegend=False), row=1, col=2)
    fig.add_hline(y=conf,  line=dict(color='gray', dash='dot'), row=1, col=2)
    fig.add_hline(y=-conf, line=dict(color='gray', dash='dot'), row=1, col=2)
    
    fig.update_xaxes(title_text='Index t', row=1, col=1)
    fig.update_yaxes(title_text='Residual', row=1, col=1)
    fig.update_xaxes(title_text='Lag', row=1, col=2)
    fig.update_yaxes(title_text='ACF', row=1, col=2)
    fig.update_layout(height=420, width=1000,
        title='AR(1) errors: residuals trend together; ACF decays slowly → Newey-West SEs needed')
    fig.show()

def non_normality_demo():
    """Notebook cell 68."""
    global K, S, X_nn, _, b_nn, col, e_nn, eps_t, fig, ic, ic_n, ic_t, jb_norm_pval, jb_pval, jb_stat, n_nn, osm, osm_n, osm_t, osr, osr_n, osr_t, sl, sl_n, sl_t, x_nn, y_nn
    # ─── Non-normality: t-distributed errors (heavy tails) ────────────────────────
    np.random.seed(42)
    n_nn = 300
    x_nn = np.random.normal(0, 1, n_nn)
    
    # t(3) errors — very heavy tails (kurtosis = ∞ for df=3, 6 for df=4)
    eps_t  = stats.t.rvs(df=3, size=n_nn)
    eps_t  = (eps_t - eps_t.mean()) / eps_t.std()   # standardize
    
    y_nn   = 2 + 1.5*x_nn + eps_t
    X_nn   = np.column_stack([np.ones(n_nn), x_nn])
    b_nn   = np.linalg.solve(X_nn.T@X_nn, X_nn.T@y_nn)
    e_nn   = y_nn - X_nn@b_nn
    
    # ─── Jarque-Bera test ─────────────────────────────────────────────────────────
    jb_stat, jb_pval = stats.jarque_bera(e_nn)
    _, jb_norm_pval  = stats.jarque_bera(np.random.normal(0, 1, n_nn))
    
    S = stats.skew(e_nn)
    K = stats.kurtosis(e_nn, fisher=False)   # excess=False → full kurtosis
    print(f"=== Residual moments ===")
    print(f"  Skewness  = {S:.4f}  (0 for normal)")
    print(f"  Kurtosis  = {K:.4f}  (3 for normal; >3 → heavy tails)")
    print(f"  Jarque-Bera: stat={jb_stat:.4f}, p={jb_pval:.4e}  → {'reject normality' if jb_pval<0.05 else 'cannot reject'}")
    print(f"  (For comparison, normal noise: p={jb_norm_pval:.4f})")
    
    # ─── Q-Q plots: normal errors vs heavy-tail errors ───────────────────────────
    (osm_n, osr_n), (sl_n, ic_n, _) = stats.probplot(np.random.normal(0,1,n_nn))
    (osm_t, osr_t), (sl_t, ic_t, _) = stats.probplot(e_nn)
    
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Normal errors (Q-Q)', 'Heavy-tail t(3) errors (Q-Q)'])
    
    for col, (osm, osr, sl, ic) in enumerate(
            [(osm_n, osr_n, sl_n, ic_n), (osm_t, osr_t, sl_t, ic_t)], start=1):
        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers',
            marker=dict(color=C['data'], opacity=0.5, size=4), showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=osm, y=np.array(osm)*sl+ic, mode='lines',
            line=dict(color='red', width=2), showlegend=False), row=1, col=col)
    
    for col in [1,2]:
        fig.update_xaxes(title_text='Theoretical quantiles', row=1, col=col)
        fig.update_yaxes(title_text='Sample quantiles', row=1, col=col)
    fig.update_layout(height=420, width=900,
        title='Q-Q plots — heavy tails create S-shaped deviation from the 45° line')
    fig.show()
    print("Heavy-tail S-shape: tails deviate above (upper) and below (lower) the reference line.")
    print("Implication: OLS β̂ still unbiased; inference OK for large n via CLT.")

def bias_variance_demo():
    """Notebook cell 72."""
    global X_bv, best_lam, col, color, cv_mse_ridge, fig, j, kf, lam, lambdas, n_bv, opacity, p_bv, ridge, ridge_coefs, ridge_train_r2, scores, true_b_bv, y_bv
    # ─── Bias-variance tradeoff visualisation ─────────────────────────────────────
    np.random.seed(42)
    n_bv   = 50   # small n to make OLS fragile
    p_bv   = 30   # many predictors
    
    # True sparse signal: only first 5 predictors are relevant
    X_bv      = np.random.normal(0, 1, (n_bv, p_bv))
    true_b_bv = np.zeros(p_bv)
    true_b_bv[:5] = [1, -2, 1.5, -1, 2]
    y_bv      = X_bv @ true_b_bv + np.random.normal(0, 1, n_bv)
    
    # Sweep λ for Ridge and Lasso
    lambdas = np.logspace(-3, 3, 80)
    
    ridge_coefs   = []
    ridge_train_r2 = []
    for lam in lambdas:
        ridge = Ridge(alpha=lam).fit(X_bv, y_bv)
        ridge_coefs.append(ridge.coef_.copy())
        ridge_train_r2.append(ridge.score(X_bv, y_bv))
    
    ridge_coefs = np.array(ridge_coefs)
    
    # Cross-validated MSE for Ridge
    kf    = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse_ridge = []
    for lam in lambdas:
        ridge  = Ridge(alpha=lam)
        scores = cross_val_score(ridge, X_bv, y_bv, cv=kf, scoring='neg_mean_squared_error')
        cv_mse_ridge.append(-scores.mean())
    
    best_lam = lambdas[np.argmin(cv_mse_ridge)]
    print(f"Best λ (Ridge, 5-fold CV): {best_lam:.4f}")
    
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Ridge coefficient paths', 'CV-MSE vs λ (Ridge)'])
    
    for j in range(p_bv):
        color = C['fit'] if j < 5 else C['neutral']
        opacity = 0.9 if j < 5 else 0.2
        fig.add_trace(go.Scatter(x=lambdas, y=ridge_coefs[:, j], mode='lines',
            line=dict(color=color, width=1.5 if j<5 else 0.5),
            opacity=opacity, showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=lambdas, y=cv_mse_ridge, mode='lines',
        line=dict(color=C['data'], width=2), showlegend=False), row=1, col=2)
    fig.add_vline(x=best_lam, line=dict(color='red', dash='dash'),
        annotation_text=f'Best λ={best_lam:.3f}', row=1, col=2)
    
    for col in [1,2]:
        fig.update_xaxes(title_text='λ (regularisation strength)', type='log', row=1, col=col)
    fig.update_yaxes(title_text='Coefficient value', row=1, col=1)
    fig.update_yaxes(title_text='CV-MSE', row=1, col=2)
    fig.update_layout(height=420, width=1050,
        title='Ridge: true predictors (red) shrink slowly; noise (gray) shrink to ≈0')
    fig.show()

def ridge_demo():
    """Notebook cell 75."""
    global X_r, b, b1_path, b2_path, b_ols, b_r2, b_scratch, fig, lam, lam_path, lams_ridge, n_r2d, r2_tr, ridge_from_scratch, ridge_sk, t_val, theta, x1_r, x2_r, y_r
    # ─── Ridge: from scratch + sklearn + geometric intuition ──────────────────────
    np.random.seed(42)
    
    # Small 2D example for geometric visualisation
    n_r2d = 100
    x1_r  = np.random.normal(0, 1, n_r2d)
    x2_r  = 0.9*x1_r + np.random.normal(0, np.sqrt(1-0.81), n_r2d)  # corr=0.9
    y_r   = 1.5*x1_r + 2.0*x2_r + np.random.normal(0, 0.5, n_r2d)
    X_r   = np.column_stack([x1_r, x2_r])   # no intercept for clarity
    
    # Ridge from scratch
    def ridge_from_scratch(X, y, lam):
        return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
    
    lams_ridge = [0, 0.5, 2, 10, 50]
    print(f"{'λ':>6} {'β₁':>10} {'β₂':>10} {'||β||₂':>10} {'Train R²':>10}")
    print("-" * 50)
    for lam in lams_ridge:
        if lam == 0:
            b = np.linalg.lstsq(X_r, y_r, rcond=None)[0]
        else:
            b = ridge_from_scratch(X_r, y_r, lam)
        r2_tr = 1 - np.sum((y_r - X_r@b)**2)/np.sum((y_r-y_r.mean())**2)
        print(f"{lam:>6.1f} {b[0]:>10.4f} {b[1]:>10.4f} {np.linalg.norm(b):>10.4f} {r2_tr:>10.4f}")
    
    # Validate with sklearn
    ridge_sk  = Ridge(alpha=2.0, fit_intercept=False).fit(X_r, y_r)
    b_scratch = ridge_from_scratch(X_r, y_r, 2.0)
    print(f"\nλ=2 from scratch:   {b_scratch}")
    print(f"λ=2 sklearn:         {ridge_sk.coef_}  ✓")
    
    # ─── Coefficient path plot ────────────────────────────────────────────────────
    lam_path = np.logspace(-2, 3, 100)
    b1_path, b2_path = [], []
    for lam in lam_path:
        b = ridge_from_scratch(X_r, y_r, lam)
        b1_path.append(b[0]); b2_path.append(b[1])
    
    b_ols = np.linalg.lstsq(X_r, y_r, rcond=None)[0]
    
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Ridge coefficient paths', 'L2 constraint geometry'])
    
    fig.add_trace(go.Scatter(x=lam_path, y=b1_path, mode='lines',
        line=dict(color=C['fit'], width=2), name='β₁'), row=1, col=1)
    fig.add_trace(go.Scatter(x=lam_path, y=b2_path, mode='lines',
        line=dict(color=C['true'], width=2), name='β₂'), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='gray', dash='dot'), row=1, col=1)
    
    # L2 constraint geometry
    theta = np.linspace(0, 2*np.pi, 200)
    t_val = np.linalg.norm(ridge_from_scratch(X_r, y_r, 2.0))
    fig.add_trace(go.Scatter(x=t_val*np.cos(theta), y=t_val*np.sin(theta), mode='lines',
        line=dict(color='gray', width=1, dash='dash'), name='L2 ball (λ=2)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[b_ols[0]], y=[b_ols[1]], mode='markers',
        marker=dict(color=C['data'], size=10, symbol='star'), name='OLS'), row=1, col=2)
    b_r2 = ridge_from_scratch(X_r, y_r, 2.0)
    fig.add_trace(go.Scatter(x=[b_r2[0]], y=[b_r2[1]], mode='markers',
        marker=dict(color=C['fit'], size=10, symbol='circle'), name='Ridge λ=2'), row=1, col=2)
    
    fig.update_xaxes(title_text='λ', type='log', row=1, col=1)
    fig.update_yaxes(title_text='Coefficient', row=1, col=1)
    fig.update_xaxes(title_text='β₁', row=1, col=2)
    fig.update_yaxes(title_text='β₂', row=1, col=2)
    fig.update_layout(height=440, width=1000,
        title='Ridge: smooth shrinkage to zero; L2 ball has no corners → no exact zeros')
    fig.show()

def lasso_demo():
    """Notebook cell 78."""
    global LassoCV, RidgeCV, alpha, alphas, color, fig, j, jidx, lasso, lasso_coefs, lasso_cv, lasso_nonzero, opacity, ridge_cv, ridge_nonzero
    # ─── Lasso vs Ridge: sparsity and coefficient comparison ──────────────────────
    from sklearn.linear_model import LassoCV, RidgeCV
    
    # Use the n=50, p=30 dataset from 5.1 (true signal is sparse: 5/30 predictors)
    alphas = np.logspace(-3, 2, 80)
    
    # Lasso coefficient paths
    lasso_coefs = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_bv, y_bv)
        lasso_coefs.append(lasso.coef_.copy())
    lasso_coefs = np.array(lasso_coefs)
    
    # Cross-validated selection
    lasso_cv  = LassoCV(alphas=alphas, cv=5, max_iter=10000).fit(X_bv, y_bv)
    ridge_cv  = RidgeCV(alphas=alphas, cv=5).fit(X_bv, y_bv)
    
    print("=== CV-selected models ===")
    lasso_nonzero = np.sum(lasso_cv.coef_ != 0)
    ridge_nonzero = np.sum(np.abs(ridge_cv.coef_) > 1e-6)
    print(f"Lasso (λ={lasso_cv.alpha_:.4f}): {lasso_nonzero}/30 non-zero coefs")
    print(f"Ridge (λ={ridge_cv.alpha_:.4f}): {ridge_nonzero}/30 non-zero coefs")
    
    print(f"\nLasso coefs for true predictors (j=0..4):")
    print(f"  {lasso_cv.coef_[:5].round(3)}")
    print(f"True values: {true_b_bv[:5]}")
    
    # ─── Coefficient path plot (Lasso) ───────────────────────────────────────────
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Lasso coefficient paths (sparsity!)', 'Lasso vs Ridge: final coefficients'])
    
    for j in range(p_bv):
        color   = C['fit'] if j < 5 else C['neutral']
        opacity = 0.9 if j < 5 else 0.2
        fig.add_trace(go.Scatter(x=alphas, y=lasso_coefs[:, j], mode='lines',
            line=dict(color=color, width=1.5 if j<5 else 0.5),
            opacity=opacity, showlegend=False), row=1, col=1)
    fig.add_vline(x=lasso_cv.alpha_, line=dict(color='red', dash='dash'),
        annotation_text=f'CV λ={lasso_cv.alpha_:.3f}', row=1, col=1)
    
    # Compare final coefs
    jidx = np.arange(p_bv)
    fig.add_trace(go.Bar(x=jidx, y=lasso_cv.coef_, name='Lasso',
        marker_color=C['fit'], opacity=0.7), row=1, col=2)
    fig.add_trace(go.Bar(x=jidx, y=ridge_cv.coef_, name='Ridge',
        marker_color=C['true'], opacity=0.7), row=1, col=2)
    fig.add_trace(go.Scatter(x=jidx, y=true_b_bv, mode='markers',
        marker=dict(color='black', size=8, symbol='diamond'), name='True β'), row=1, col=2)
    fig.add_vline(x=4.5, line=dict(color='gray', dash='dot'),
        annotation_text='True signal | noise', row=1, col=2)
    
    fig.update_xaxes(title_text='λ', type='log', row=1, col=1)
    fig.update_yaxes(title_text='Coefficient', row=1, col=1)
    fig.update_xaxes(title_text='Predictor index', row=1, col=2)
    fig.update_yaxes(title_text='Coefficient', row=1, col=2)
    fig.update_layout(height=440, width=1050, barmode='group',
        title='Lasso zeros out noise predictors (red); Ridge merely shrinks them (green)')
    fig.show()

def elastic_net_demo():
    """Notebook cell 81."""
    global ElasticNet, ElasticNetCV, GridSearchCV, LassoCV, RidgeCV, X_en, en_en, factor, fig, g, gline, j, jidx, lasso_en, n_en, p_en, ridge_en, true_b_en, y_en
    # ─── Elastic Net: compare Ridge / Lasso / EN on correlated predictors ─────────
    np.random.seed(42)
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import ElasticNet
    
    # Dataset with groups of correlated predictors
    n_en, p_en = 150, 20
    # 4 groups of 5 correlated features; only 2 groups are relevant
    factor = np.random.normal(0, 1, (n_en, 4))  # latent factors
    X_en   = np.zeros((n_en, p_en))
    for g in range(4):
        for j in range(5):
            X_en[:, g*5+j] = factor[:, g] + 0.3*np.random.normal(0, 1, n_en)
    
    true_b_en = np.zeros(p_en)
    true_b_en[:5]  = 1.0   # group 1 relevant
    true_b_en[5:10]= -1.5  # group 2 relevant
    y_en = X_en @ true_b_en + np.random.normal(0, 0.5, n_en)
    
    # Fit all three with CV
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.linear_model import ElasticNetCV
    
    lasso_en  = LassoCV(cv=5, max_iter=10000).fit(X_en, y_en)
    ridge_en  = RidgeCV(cv=5).fit(X_en, y_en)
    en_en     = ElasticNetCV(l1_ratio=[0.1,0.5,0.7,0.9,0.95,1.0], cv=5,
                              max_iter=10000).fit(X_en, y_en)
    
    print(f"Best models:")
    print(f"  Lasso  (λ={lasso_en.alpha_:.4f}):      {np.sum(lasso_en.coef_!=0)}/20 non-zero")
    print(f"  Ridge  (λ={ridge_en.alpha_:.4f}):     {np.sum(np.abs(ridge_en.coef_)>1e-6)}/20 non-zero")
    print(f"  ElasNet (λ={en_en.alpha_:.4f}, α={en_en.l1_ratio_:.2f}): {np.sum(en_en.coef_!=0)}/20 non-zero")
    
    jidx = np.arange(p_en)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=jidx, y=lasso_en.coef_, name='Lasso',
        marker_color=C['fit'], opacity=0.7))
    fig.add_trace(go.Bar(x=jidx, y=ridge_en.coef_, name='Ridge',
        marker_color=C['true'], opacity=0.7))
    fig.add_trace(go.Bar(x=jidx, y=en_en.coef_, name='Elastic Net',
        marker_color=C['alt'], opacity=0.7))
    fig.add_trace(go.Scatter(x=jidx, y=true_b_en, mode='markers',
        marker=dict(color='black', size=8, symbol='diamond'), name='True β'))
    for gline in [4.5, 9.5, 14.5]:
        fig.add_vline(x=gline, line=dict(color='gray', dash='dot'))
    
    fig.update_layout(barmode='group', height=440, width=950,
        title='Correlated groups: Lasso picks arbitrarily; Elastic Net handles groups better',
        xaxis_title='Predictor', yaxis_title='Coefficient')
    fig.show()

def bayesian_posterior_demo():
    """Notebook cell 85."""
    global Sigma_n, Sn_k, X_b_noconst, beta_grid, beta_true, col, colors_seq, fig, mn_k, mu_n, n_b, nobs, post_pdf, prior_pdf, sigma2_true, tau2, x_b, y_b
    # ─── Bayesian linear regression: visualise posterior ──────────────────────────
    np.random.seed(42)
    
    # Small 1D example: y = β₁·x + ε, β₁ unknown
    n_b = 30
    x_b = np.random.uniform(-3, 3, n_b)
    sigma2_true = 1.0
    beta_true   = 2.0
    
    y_b = beta_true * x_b + np.random.normal(0, np.sqrt(sigma2_true), n_b)
    
    # Prior: β₁ ~ N(0, τ²)
    tau2 = 4.0   # fairly uninformative
    
    # Posterior (known σ²): β̂_post = Σ_n · (1/σ²) · X'y
    X_b_noconst = x_b.reshape(-1, 1)   # no intercept for simplicity
    
    Sigma_n = 1.0 / (np.sum(x_b**2) / sigma2_true + 1.0/tau2)
    mu_n    = Sigma_n * (x_b @ y_b) / sigma2_true
    
    print(f"Prior: β ~ N(0, {tau2})")
    print(f"OLS estimate:       β̂_OLS  = {np.sum(x_b*y_b)/np.sum(x_b**2):.4f}")
    print(f"Posterior mean:     β̂_post = {mu_n:.4f}  (= MAP estimate)")
    print(f"Posterior std:      σ_post  = {np.sqrt(Sigma_n):.4f}")
    print(f"95% Credible int.: [{mu_n - 1.96*np.sqrt(Sigma_n):.4f}, {mu_n + 1.96*np.sqrt(Sigma_n):.4f}]")
    print(f"True β = {beta_true}")
    
    # ─── Show how posterior updates with more data ─────────────────────────────────
    beta_grid = np.linspace(0, 4, 500)
    prior_pdf = stats.norm.pdf(beta_grid, 0, np.sqrt(tau2))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=beta_grid, y=prior_pdf/prior_pdf.max(), mode='lines',
        line=dict(color='gray', dash='dot', width=2), name='Prior N(0, 4)'))
    
    # Sequential update: show posterior after 5, 15, 30 observations
    colors_seq = [C['data'], C['alt'], C['fit']]
    for nobs, col in zip([5, 15, 30], colors_seq):
        Sn_k = 1.0 / (np.sum(x_b[:nobs]**2)/sigma2_true + 1/tau2)
        mn_k = Sn_k * (x_b[:nobs] @ y_b[:nobs]) / sigma2_true
        post_pdf = stats.norm.pdf(beta_grid, mn_k, np.sqrt(Sn_k))
        fig.add_trace(go.Scatter(x=beta_grid, y=post_pdf/post_pdf.max(), mode='lines',
            line=dict(color=col, width=2), name=f'Posterior n={nobs}'))
    
    fig.add_vline(x=beta_true, line=dict(color='black', dash='dash'),
        annotation_text=f'True β={beta_true}')
    fig.update_layout(
        title='Bayesian updating: posterior concentrates around truth as n grows',
        xaxis_title='β₁', yaxis_title='Normalised density',
        width=750, height=430)
    fig.show()

def map_equals_ridge_demo():
    """Notebook cell 88."""
    global Sigma_n_map, X_map, b_laplace, b_ridge, beta_map_true, beta_viz, fig, gauss_prior, j, lam_map, laplace_prior, mu_n_map, n_map, p_map, sigma2, tau2, tau_common, y_map
    # ─── MAP = Ridge: verify analytically ────────────────────────────────────────
    np.random.seed(42)
    n_map, p_map = 80, 5
    X_map = np.random.normal(0, 1, (n_map, p_map))
    beta_map_true = np.array([1., -2., 0.5, 1.5, -1.])
    y_map = X_map @ beta_map_true + np.random.normal(0, 1, n_map)
    
    sigma2 = 1.0   # assume known
    tau2   = 4.0   # prior variance
    
    # Ridge with λ = σ²/τ²
    lam_map = sigma2 / tau2
    b_ridge = np.linalg.solve(X_map.T@X_map + lam_map*np.eye(p_map), X_map.T@y_map)
    
    # MAP posterior mean directly: Σ_n = (X'X/σ² + I/τ²)⁻¹, μ_n = Σ_n X'y/σ²
    Sigma_n_map = np.linalg.inv(X_map.T@X_map/sigma2 + np.eye(p_map)/tau2)
    mu_n_map    = Sigma_n_map @ (X_map.T @ y_map) / sigma2
    
    print("Verify: Ridge(λ=σ²/τ²) = MAP posterior mean")
    print(f"{'j':>3} {'Ridge':>10} {'MAP mean':>10} {'True β':>10}")
    for j in range(p_map):
        print(f"{j:>3} {b_ridge[j]:>10.6f} {mu_n_map[j]:>10.6f} {beta_map_true[j]:>10.6f}")
    print(f"\nMax difference: {np.max(np.abs(b_ridge - mu_n_map)):.2e}  ≈ 0 ✓")
    
    # ─── Visualise: Gaussian vs Laplace prior shape ────────────────────────────────
    beta_viz = np.linspace(-5, 5, 500)
    tau_common = 1.0
    b_laplace  = tau_common * np.sqrt(2)   # Laplace scale for same variance as Gaussian
    
    gauss_prior  = stats.norm.pdf(beta_viz, 0, tau_common)
    laplace_prior= stats.laplace.pdf(beta_viz, 0, b_laplace)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=beta_viz, y=gauss_prior, mode='lines',
        line=dict(color=C['true'], width=2.5), name=f'Gaussian N(0, {tau_common}²) → Ridge'))
    fig.add_trace(go.Scatter(x=beta_viz, y=laplace_prior, mode='lines',
        line=dict(color=C['fit'], width=2.5), name=f'Laplace(0, {b_laplace:.2f}) → Lasso'))
    fig.update_layout(
        title='Prior shapes: Laplace has heavier tails but sharper peak at 0 → sparsity',
        xaxis_title='β', yaxis_title='Prior density p(β)',
        width=700, height=420)
    fig.show()
    
    print("\nLaplace prior: sharp peak at 0 → MAP solution more likely to be exactly 0 (sparsity)")
    print("Gaussian prior: smooth decay → MAP solution always non-zero (Ridge)")

def full_bayesian_regression_demo():
    """Notebook cell 91."""
    global Sigma_n_bay, X_bay, X_pred, beta_samples, epist_std, epist_var, fig, k_bay, mu_n_bay, n_bay, n_samples, pred_mean, pred_std, pred_var, samp, sigma2_bay, tau2_bay, true_a, true_b_bay, x_bay, x_pred, y_bay
    # ─── Full Bayesian regression: posterior and predictive uncertainty ────────────
    np.random.seed(42)
    
    # 1D example: y = α + βx + ε
    n_bay = 40
    x_bay = np.sort(np.random.uniform(-3, 3, n_bay))
    sigma2_bay = 0.5**2
    true_a, true_b_bay = 1.0, 1.5
    
    y_bay = true_a + true_b_bay * x_bay + np.random.normal(0, np.sqrt(sigma2_bay), n_bay)
    
    # Design matrix (with intercept)
    X_bay = np.column_stack([np.ones(n_bay), x_bay])
    k_bay = 2
    tau2_bay = 9.0   # wide prior
    
    # Posterior
    Sigma_n_bay = np.linalg.inv(X_bay.T @ X_bay / sigma2_bay + np.eye(k_bay)/tau2_bay)
    mu_n_bay    = Sigma_n_bay @ (X_bay.T @ y_bay) / sigma2_bay
    
    print("=== Posterior Summary ===")
    print(f"  α: post mean={mu_n_bay[0]:.4f}  post std={np.sqrt(Sigma_n_bay[0,0]):.4f}  [true: {true_a}]")
    print(f"  β: post mean={mu_n_bay[1]:.4f}  post std={np.sqrt(Sigma_n_bay[1,1]):.4f}  [true: {true_b_bay}]")
    
    # Predictive mean and variance for a grid of x values
    x_pred = np.linspace(-4, 4, 200)
    X_pred = np.column_stack([np.ones(len(x_pred)), x_pred])
    
    pred_mean = X_pred @ mu_n_bay
    pred_var  = np.array([X_pred[i] @ Sigma_n_bay @ X_pred[i] + sigma2_bay
                           for i in range(len(x_pred))])
    pred_std  = np.sqrt(pred_var)
    
    # Epistemic (parameter) uncertainty only
    epist_var = np.array([X_pred[i] @ Sigma_n_bay @ X_pred[i] for i in range(len(x_pred))])
    epist_std = np.sqrt(epist_var)
    
    # Draw posterior samples for beta
    n_samples = 50
    beta_samples = np.random.multivariate_normal(mu_n_bay, Sigma_n_bay, n_samples)
    
    fig = go.Figure()
    
    # Posterior predictive samples (light lines)
    for samp in beta_samples[:20]:
        fig.add_trace(go.Scatter(x=x_pred, y=X_pred @ samp, mode='lines',
            line=dict(color=C['data'], width=0.5), opacity=0.15, showlegend=False))
    
    # True line
    fig.add_trace(go.Scatter(x=x_pred, y=true_a + true_b_bay*x_pred, mode='lines',
        line=dict(color='black', width=2, dash='dash'), name='True line'))
    
    # Posterior mean
    fig.add_trace(go.Scatter(x=x_pred, y=pred_mean, mode='lines',
        line=dict(color=C['fit'], width=2.5), name='Posterior mean'))
    
    # 95% credible band (full predictive — includes σ²)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_pred, x_pred[::-1]]),
        y=np.concatenate([pred_mean + 1.96*pred_std, (pred_mean - 1.96*pred_std)[::-1]]),
        fill='toself', fillcolor='rgba(239,85,59,0.10)',
        line=dict(width=0), name='95% predictive interval'))
    
    # 95% credible band (epistemic only)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_pred, x_pred[::-1]]),
        y=np.concatenate([pred_mean + 1.96*epist_std, (pred_mean - 1.96*epist_std)[::-1]]),
        fill='toself', fillcolor='rgba(0,204,150,0.15)',
        line=dict(width=0), name='95% epistemic band (β uncertainty only)'))
    
    # Data
    fig.add_trace(go.Scatter(x=x_bay, y=y_bay, mode='markers',
        marker=dict(color='black', size=6, opacity=0.7), name='Data'))
    
    fig.update_layout(
        title='Bayesian Predictive Distribution<br>'
              'Blue samples = posterior draws of β; Green band = β uncertainty; Red band = full prediction',
        xaxis_title='x', yaxis_title='y',
        width=850, height=520)
    fig.show()
    
    print("\nFull predictive uncertainty = epistemic (β uncertainty) + aleatoric (irreducible noise σ²)")
    print(f"At x=0: pred_std={pred_std[100]:.4f} = sqrt(epist={epist_var[100]:.4f} + σ²={sigma2_bay:.4f})")
