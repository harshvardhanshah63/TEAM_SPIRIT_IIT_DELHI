#!/usr/bin/env python3
"""
Corporate Heist - Nightingale Top 5% Hire Prediction
Team solution for INNOV8 4.0
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from rapidfuzz import fuzz
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. PARSING HELPERS
# ============================================================

def parse_experience(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if 'fresher' in s or s in ['0', '0 years', '<1 year', '0 yrs']:
        return 0.0
    nums = re.findall(r'[\d.]+', s)
    if not nums:
        return np.nan
    v = float(nums[0])
    if '>' in s or '+' in s:
        v = max(v, 20.0) if v >= 20 else v + 1
    return v

def parse_notice_days(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if any(x in s for x in ['immediate', 'available now', '0 days']):
        return 0.0
    nums = re.findall(r'[\d.]+', s)
    if not nums:
        return np.nan
    v = float(nums[0])
    if 'month' in s:
        v *= 30
    return v

def parse_ctc_lpa(val):
    """Normalize CTC to approximate LPA (lakhs per annum)."""
    if pd.isna(val):
        return np.nan
    s = str(val).lower().replace(',', '').replace('₹', '').replace('inr', '').strip()
    nums = re.findall(r'[\d.]+', s)
    if not nums:
        return np.nan
    v = float(nums[0])
    if '$' in str(val) or 'usd' in s:
        # rough USD to INR LPA (assume ~83 INR/USD, annual)
        v = v * 83 / 100000  # if absolute USD
        if v > 500:  # already large? treat as annual USD
            v = float(nums[0]) * 83 / 100000
    elif 'cr' in s:
        v *= 100
    elif 'lakh' in s or 'lpa' in s or 'l' in s and v < 1000:
        pass  # already LPA
    elif v > 100000:  # absolute INR
        v = v / 100000
    elif v > 500:  # maybe thousands
        v = v / 100
    return v

def parse_score(val, max_possible=100):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if s in ['nan', '', '-', 'none', 'na']:
        return np.nan
    # outstanding, excellent etc -> high
    if any(x in s for x in ['outstanding', 'excellent', 'a+', 'a']):
        return 90.0
    if 'good' in s:
        return 70.0
    nums = re.findall(r'[\d.]+', s)
    if not nums:
        return np.nan
    v = float(nums[0])
    if '/10' in s or (v <= 10 and max_possible == 100):
        v = v * 10
    if '/5' in s or (v <= 5 and 'rating' in s):
        v = v * 20
    return min(v, 100.0)

def parse_public_code(val):
    if pd.isna(val):
        return 0.0
    s = str(val).lower().strip()
    if s in ['nan', '', 'not tracked', 'none', '-']:
        return 0.0
    nums = re.findall(r'[\d.]+', s)
    if not nums:
        return 0.0
    v = float(nums[0])
    if '+' in s:
        v += 5
    return v

def parse_age(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

def count_skills(val):
    if pd.isna(val):
        return 0
    s = str(val)
    # split on common separators
    parts = re.split(r'[,;|/>]+', s)
    return len([p for p in parts if p.strip()])

def parse_career_growth(career_path, total_exp):
    """Estimate title inflation: number of distinct senior titles / years."""
    if pd.isna(career_path) or pd.isna(total_exp) or total_exp <= 0:
        return 0.0
    s = str(career_path).lower()
    senior_keywords = ['senior', 'lead', 'principal', 'staff', 'director', 'vp', 'head', 'chief', 'manager', 'architect']
    # count promotions-ish
    titles = re.split(r'[>|→\->]+', s)
    n_titles = len([t for t in titles if t.strip()])
    n_senior = sum(1 for t in titles if any(k in t for k in senior_keywords))
    # growth rate
    if total_exp < 1:
        return n_senior * 2
    return n_senior / max(total_exp, 1.0)

def extract_years_from_path(career_path):
    if pd.isna(career_path):
        return np.nan
    s = str(career_path).lower()
    # find all durations
    months = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mo|months?)', s)
    years = re.findall(r'(\d+(?:\.\d+)?)\s*(?:y|yrs?|years?)', s)
    total_m = sum(float(m) for m in months) + sum(float(y)*12 for y in years)
    return total_m / 12.0 if total_m > 0 else np.nan

# ============================================================
# 2. CLEANING & FEATURE ENGINEERING
# ============================================================

def clean_and_featurize(df, is_test=False):
    df = df.copy()
    
    # Numeric parses
    df['exp_years'] = df['total_experience'].apply(parse_experience)
    df['notice_days'] = df['notice_period'].apply(parse_notice_days)
    df['ctc_lpa'] = df['current_ctc'].apply(parse_ctc_lpa)
    df['exp_ctc_lpa'] = df['expected_ctc'].apply(parse_ctc_lpa)
    df['tech_score'] = df['technical_assessment'].apply(lambda x: parse_score(x, 100))
    df['apt_score'] = df['aptitude_score'].apply(lambda x: parse_score(x, 10))
    df['age_clean'] = df['age'].apply(parse_age)
    df['skills_count'] = df['skills'].apply(count_skills)
    df['path_years'] = df['career_path'].apply(extract_years_from_path)
    
    # Use path_years to fill exp if missing
    df['exp_years'] = df['exp_years'].fillna(df['path_years'])
    
    # Career growth / inflation score
    df['title_inflation'] = df.apply(
        lambda r: parse_career_growth(r['career_path'], r['exp_years']), axis=1
    )
    
    # Public code (test only)
    if is_test and 'public_code_contributions' in df.columns:
        df['code_contrib'] = df['public_code_contributions'].apply(parse_public_code)
    else:
        df['code_contrib'] = 0.0
    
    # Binary / simple cats
    df['kpi_yes'] = df['kpi_met'].astype(str).str.upper().str.contains('Y|YES|TRUE|1').astype(int)
    df['overtime_yes'] = df['overtime_history'].astype(str).str.lower().str.contains('yes|y|true').astype(int)
    df['enrolled'] = df['currently_enrolled'].astype(str).str.lower().str.contains('yes|enrolled|course').astype(int)
    
    # last_job_change
    def parse_job_change(v):
        if pd.isna(v): return np.nan
        s = str(v).lower()
        if 'never' in s: return 99.0
        nums = re.findall(r'[\d.]+', s)
        return float(nums[0]) if nums else np.nan
    df['job_change_yrs'] = df['last_job_change'].apply(parse_job_change)
    
    # Awards / trainings
    df['has_awards'] = (~df['awards'].isna() & (df['awards'].astype(str).str.lower() != 'none') & 
                        (df['awards'].astype(str) != '-') & (df['awards'].astype(str) != '')).astype(int)
    df['trainings'] = pd.to_numeric(df['trainings_last_year'], errors='coerce').fillna(0)
    df['train_hrs'] = pd.to_numeric(df['training_hours'], errors='coerce').fillna(0)
    
    # Graduation age proxy
    df['grad_year'] = pd.to_numeric(df['graduation_year'], errors='coerce')
    df['years_since_grad'] = 2026 - df['grad_year']  # assume current ~2026
    df['exp_vs_grad'] = df['exp_years'] - (df['years_since_grad'] - 0.5)  # rough
    
    # Inconsistency flags (fabricated detection)
    df['flag_exp_age'] = ((df['exp_years'] > (df['age_clean'] - 18)) & df['age_clean'].notna()).astype(int)
    df['flag_fast_growth'] = (df['title_inflation'] > 1.5).astype(int)  # aggressive threshold
    df['flag_ctc_unreal'] = ((df['ctc_lpa'] > 150) | (df['exp_ctc_lpa'] > 200)).astype(int)
    df['flag_notice_long'] = (df['notice_days'] > 60).astype(int)
    df['flag_inconsistent'] = (
        df['flag_exp_age'] | df['flag_fast_growth'] | 
        ((df['exp_years'] > 25) & (df['age_clean'] < 40))
    ).astype(int)
    
    # Simple institute prestige (will downweight)
    top_institutes = ['iit', 'nit', 'bits', 'iiit', 'iisc', 'iim', 'isb']
    df['is_top_inst'] = df['institute'].astype(str).str.lower().apply(
        lambda x: any(t in x for t in top_institutes)
    ).astype(int)
    
    # Role encoding later
    return df

# ============================================================
# 3. DEDUPLICATION
# ============================================================

def find_duplicates(df, threshold=90):
    """Find likely same-person groups via name + email/phone fuzzy."""
    # For efficiency, group by exact email first, then name fuzzy within
    df = df.copy()
    df['email_norm'] = df['email'].astype(str).str.lower().str.strip()
    df['name_norm'] = df['full_name'].astype(str).str.lower().str.strip()
    df['phone_norm'] = df['phone'].astype(str).str.replace(r'\D', '', regex=True)
    
    # Exact email matches
    email_counts = df['email_norm'].value_counts()
    dup_emails = set(email_counts[email_counts > 1].index)
    
    # For ranking we keep one per person; mark others
    df['is_dup'] = df['email_norm'].isin(dup_emails)
    
    # Also name + phone
    name_phone = df.groupby(['name_norm', 'phone_norm']).size()
    multi = name_phone[name_phone > 1]
    # simplistic; for production would do more
    return df

# ============================================================
# 4. MODEL TRAINING
# ============================================================

def prepare_features(df, label_encoders=None, fit=False):
    feature_cols = [
        'exp_years', 'notice_days', 'ctc_lpa', 'exp_ctc_lpa', 'tech_score', 'apt_score',
        'age_clean', 'skills_count', 'title_inflation', 'code_contrib',
        'kpi_yes', 'overtime_yes', 'enrolled', 'job_change_yrs',
        'has_awards', 'trainings', 'train_hrs', 'years_since_grad',
        'is_top_inst', 'flag_inconsistent', 'flag_notice_long', 'flag_fast_growth',
        'num_employers'
    ]
    
    # numeric employers
    df['num_employers'] = pd.to_numeric(df['num_employers'], errors='coerce').fillna(1)
    
    # Categoricals
    cat_cols = ['applied_role', 'recruitment_channel', 'company_type', 'company_size', 'major', 'degree']
    
    if fit:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('UNK'))
            label_encoders[col] = le
    else:
        for col in cat_cols:
            le = label_encoders[col]
            # handle unseen
            vals = df[col].astype(str).fillna('UNK')
            known = set(le.classes_)
            vals = vals.apply(lambda x: x if x in known else 'UNK')
            # if UNK not in classes, map to 0
            if 'UNK' not in le.classes_:
                df[col + '_enc'] = 0
            else:
                df[col + '_enc'] = le.transform(vals)
    
    feature_cols += [c + '_enc' for c in cat_cols]
    
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, label_encoders, feature_cols

def train_model(train_df):
    train_df = clean_and_featurize(train_df, is_test=False)
    y = train_df['post_hire_score']
    
    X, les, feats = prepare_features(train_df, fit=True)
    
    # simple split for local val
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=42)
    
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[])
    
    # full retrain
    model.fit(X, y)
    
    # feature importance
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("Top features:")
    print(imp.head(15))
    
    return model, les, feats, train_df

# ============================================================
# 5. SCORING & RANKING WITH RULES
# ============================================================

def score_and_rank(test_df, model, les, feats):
    test_df = clean_and_featurize(test_df, is_test=True)
    X, _, _ = prepare_features(test_df, label_encoders=les, fit=False)
    
    # Align columns
    for c in feats:
        if c not in X.columns:
            X[c] = 0
    X = X[feats]
    
    pred = model.predict(X)
    test_df['pred_score'] = pred
    
    # === RULE-BASED ADJUSTMENTS (from insider note) ===
    
    # 1. Boost high public code contributions (dozens or more)
    # sustained record -> fast-track
    code_boost = np.where(
        test_df['code_contrib'] >= 30, 15.0,
        np.where(test_df['code_contrib'] >= 15, 8.0,
        np.where(test_df['code_contrib'] >= 8, 3.0, 0.0))
    )
    
    # 2. Penalize long notice (> 2 months = 60 days)
    notice_penalty = np.where(test_df['notice_days'] > 60, -25.0,
                     np.where(test_df['notice_days'] > 45, -8.0, 0.0))
    
    # 3. Penalize unrealistic career growth
    growth_penalty = np.where(test_df['flag_fast_growth'] == 1, -20.0,
                     np.where(test_df['title_inflation'] > 1.0, -8.0, 0.0))
    
    # 4. Reduce importance of college prestige (actually slight penalty for top if not compensating)
    # Insider: new panel ignores pedigree; old-boys still get leg up, but overall reduce
    prestige_adj = np.where(test_df['is_top_inst'] == 1, -2.0, 1.5)  # slight boost to non-top
    
    # 5. Extra boost for high tech_score + good fit signals
    tech_boost = np.where(test_df['tech_score'] >= 70, 5.0,
                 np.where(test_df['tech_score'] >= 50, 2.0, 0.0))
    
    # 6. Penalize fabricated / inconsistent
    fake_penalty = np.where(test_df['flag_inconsistent'] == 1, -30.0, 0.0)
    
    # Combined adjusted score
    test_df['adj_score'] = (
        test_df['pred_score'] 
        + code_boost 
        + notice_penalty 
        + growth_penalty 
        + prestige_adj 
        + tech_boost 
        + fake_penalty
    )
    
    # Dedup: keep highest score per normalized email / name
    test_df['email_norm'] = test_df['email'].astype(str).str.lower().str.strip()
    test_df['name_norm'] = test_df['full_name'].astype(str).str.lower().str.strip()
    
    # Sort by adj_score desc, then drop duplicate persons keeping first
    test_df = test_df.sort_values('adj_score', ascending=False)
    
    # Remove exact email dups
    test_df = test_df.drop_duplicates(subset=['email_norm'], keep='first')
    # Also soft name dups (same name different email - keep higher score)
    test_df = test_df.drop_duplicates(subset=['name_norm'], keep='first')
    
    # Further filter extreme fakes
    test_df = test_df[test_df['flag_inconsistent'] == 0]
    # Still keep some if score high? but insider says never hire fake
    # Re-apply after filter? No, already filtered.
    
    # Take top 500
    top500 = test_df.head(500).copy()
    top500 = top500.reset_index(drop=True)
    top500['rank'] = range(1, 501)
    
    submission = top500[['rank', 'candidate_id']]
    return submission, test_df, top500

# ============================================================
# 6. VALIDATION ON DEV
# ============================================================

def validate_on_dev(model, les, feats):
    dev = pd.read_csv('dev.csv')
    winners = pd.read_csv('dev_winners.csv')
    winner_ids = set(winners['candidate_id'])
    
    dev_clean = clean_and_featurize(dev, is_test=False)
    X, _, _ = prepare_features(dev_clean, label_encoders=les, fit=False)
    for c in feats:
        if c not in X.columns:
            X[c] = 0
    X = X[feats]
    pred = model.predict(X)
    dev_clean['pred_score'] = pred
    
    # Apply similar rules (no code_contrib)
    notice_penalty = np.where(dev_clean['notice_days'] > 60, -25.0,
                     np.where(dev_clean['notice_days'] > 45, -8.0, 0.0))
    growth_penalty = np.where(dev_clean['flag_fast_growth'] == 1, -20.0,
                     np.where(dev_clean['title_inflation'] > 1.0, -8.0, 0.0))
    prestige_adj = np.where(dev_clean['is_top_inst'] == 1, -2.0, 1.5)
    tech_boost = np.where(dev_clean['tech_score'] >= 70, 5.0,
                 np.where(dev_clean['tech_score'] >= 50, 2.0, 0.0))
    fake_penalty = np.where(dev_clean['flag_inconsistent'] == 1, -30.0, 0.0)
    
    dev_clean['adj_score'] = (
        dev_clean['pred_score'] + notice_penalty + growth_penalty +
        prestige_adj + tech_boost + fake_penalty
    )
    
    dev_clean = dev_clean.sort_values('adj_score', ascending=False)
    # take top 150
    top150 = set(dev_clean.head(150)['candidate_id'])
    
    overlap = len(top150 & winner_ids)
    print(f"\n=== DEV VALIDATION ===")
    print(f"Overlap with true winners (top 150): {overlap} / 150 = {overlap/150*100:.1f}%")
    
    # Also check ranking quality
    dev_clean['is_winner'] = dev_clean['candidate_id'].isin(winner_ids)
    mean_rank_winners = dev_clean[dev_clean['is_winner']].index.to_series().mean() if hasattr(dev_clean.index, 'to_series') else np.nan
    # better: position
    positions = []
    for i, row in enumerate(dev_clean.itertuples()):
        if row.candidate_id in winner_ids:
            positions.append(i+1)
    if positions:
        print(f"Mean position of winners: {np.mean(positions):.1f}")
        print(f"Median position of winners: {np.median(positions):.1f}")
        print(f"Winners in top 300: {sum(1 for p in positions if p <= 300)}")
    return overlap

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    print("Training model...")
    model, les, feats, train_clean = train_model(train)
    
    print("\nValidating on dev...")
    validate_on_dev(model, les, feats)
    
    print("\nScoring test set...")
    submission, scored_test, top500 = score_and_rank(test, model, les, feats)
    
    print(f"\nSubmission shape: {submission.shape}")
    print(submission.head(10))
    print("...")
    print(submission.tail(5))
    
    # Sanity
    print(f"\nUnique candidates: {submission['candidate_id'].nunique()}")
    print(f"Code contrib in top 500 (mean): {top500['code_contrib'].mean():.1f}")
    print(f"Notice days mean top500: {top500['notice_days'].mean():.1f}")
    print(f"Fast growth flags in top500: {top500['flag_fast_growth'].sum()}")
    
    submission.to_csv('submission.csv', index=False)
    print("\nWrote submission.csv")
    
    # Also save full scored for analysis
    scored_test[['candidate_id', 'pred_score', 'adj_score', 'code_contrib', 
                 'notice_days', 'title_inflation', 'tech_score']].to_csv('scored_test.csv', index=False)
