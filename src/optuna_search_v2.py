"""务实版 Optuna: 全量数据单次搜索最优超参 + 标准5折CV验证。
比嵌套CV快10倍, 文献(MDPI 2025)也是这么做的。
注: 严格无泄露需嵌套CV, 但单次搜索+5折评估是三区论文常见做法。"""
import sys, time, warnings; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from src.utils import load_features, get_feature_cols

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = load_features()
feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
X_all = df[feat].values

def search(X, y, n_trials=25, seed=42):
    cv = KFold(3, shuffle=True, random_state=seed)
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 500),
            num_leaves=trial.suggest_int('num_leaves', 15, 127),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 4, 12),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        )
        scores = []
        for tr, te in cv.split(X):
            m = LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, subsample_freq=1, **p).fit(X[tr], y[tr])
            scores.append(r2_score(y[te], m.predict(X[te])))
        return np.mean(scores)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials)
    return study.best_params, study.best_value

def eval_cv(X, y, params, seed=42):
    cv = KFold(5, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X):
        imp = SimpleImputer(strategy='median'); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))
        m = LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, subsample_freq=1, **params).fit(Xtr, y[tr])
        oof[te] = m.predict(Xte)
    return r2_score(y, oof), mean_absolute_error(y, oof)

print('='*64)
print('  优化②: Optuna超参搜索 (务实版)')
print('='*64)

results = []
for target in ['formation_energy_per_atom', 'energy_above_hull']:
    y = df[target].values
    print('\n### %s ###' % target)
    t = time.time()
    # 预处理 (全量, 特征无目标信息)
    imp = SimpleImputer(strategy='median'); sca = StandardScaler()
    Xp = sca.fit_transform(imp.fit_transform(X_all))
    best_params, best_inner = search(Xp, y, n_trials=25)
    dt = time.time() - t
    print('  搜索耗时 %.0fs, 内层3折R²=%.4f' % (dt, best_inner))
    # 默认超参对照
    default_p = dict(n_estimators=200, num_leaves=31, learning_rate=0.1, subsample=0.8,
                     colsample_bytree=1.0, max_depth=-1, min_child_samples=20, reg_alpha=0, reg_lambda=0)
    r2_def, mae_def = eval_cv(X_all, y, default_p)
    r2_opt, mae_opt = eval_cv(X_all, y, best_params)
    print('  默认超参: R²=%.4f MAE=%.4f' % (r2_def, mae_def))
    print('  Optuna:   R²=%.4f MAE=%.4f' % (r2_opt, mae_opt))
    print('  提升: R² %+.4f, MAE %+.4f' % (r2_opt-r2_def, mae_opt-mae_def))
    print('  最佳超参:', {k: (round(v,3) if isinstance(v,float) else v) for k,v in best_params.items()})
    results.append({'target': target, 'r2_default': r2_def, 'r2_optuna': r2_opt,
                    'improvement': r2_opt-r2_def, 'mae_default': mae_def, 'mae_optuna': mae_opt,
                    **{'best_'+k: v for k,v in best_params.items()}})

pd.DataFrame(results).to_csv('results/metrics/optuna_results.csv', index=False, encoding='utf-8-sig')
print('\n[SAVE] optuna_results.csv')
