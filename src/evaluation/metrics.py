import re
from typing import Any
import pandas as pd
from rapidfuzz import fuzz
from src.extraction.schema import ExtractionValue

_FRENCH_MONTHS = {
    "janv": 1, "janvier": 1, "fev": 2, "févr": 2, "fevrier": 2, "février": 2, "mars": 3, "avr": 4, "avril": 4,
    "mai": 5, "juin": 6, "juil": 7, "juillet": 7, "aout": 8, "août": 8, "sept": 9, "septembre": 9,
    "oct": 10, "octobre": 10, "nov": 11, "novembre": 11, "dec": 12, "déc": 12, "decembre": 12, "décembre": 12,
}

_FUZZY_THRESHOLD = 85
_FUZZY_ELIGIBLE_FIELDS = {
    "diag_histologique", "diag_integre", "tumeur_position",
    "activite_professionnelle", "chimios", "localisation_chir",
    "localisation_radiotherapie", "neuroncologue", "neurochirurgien",
    "radiotherapeute", "infos_deces", "autre_trouble",
}

_ALTERATION_WEIGHT = 0.5

def _try_parse_date(s: str) -> str | None:
    """Attempt to parse a date string to YYYY-MM-DD canonical form."""
    s = s.strip().lower()
    # DD/MM/YYYY or DD.MM.YYYY or DD-MM-YYYY
    m = re.match(r'^(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{4})$', s)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    # YYYY-MM-DD
    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    
    # "12 mars 2021"
    m = re.match(r'^(\d{1,2})\s+([a-zéû]+)\s+(\d{4})$', s)
    if m and m.group(2) in _FRENCH_MONTHS:
        return f"{m.group(3)}-{_FRENCH_MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
        
    # "mars 2021"
    m = re.match(r'^([a-zéû]+)\s+(\d{4})$', s)
    if m and m.group(1) in _FRENCH_MONTHS:
        return f"{m.group(2)}-{_FRENCH_MONTHS[m.group(1)]:02d}"

    # month_abbrev-YY (e.g., "avr-10" → 2010-04)
    m = re.match(r'^([a-zéû]+)-(\d{2})$', s)
    if m and m.group(1) in _FRENCH_MONTHS:
        year = 2000 + int(m.group(2))
        return f"{year}-{_FRENCH_MONTHS[m.group(1)]:02d}"
    # Year only (e.g., "2008")
    m = re.match(r'^(\d{4})$', s)
    if m:
        return m.group(1)
    return None

def compute_per_feature_metrics(
    predicted: dict[str, ExtractionValue],
    ground_truth: dict[str, Any]
) -> dict[str, dict[str, int]]:
    """
    Compare a single document's predicted features against the ground truth.
    
    Parameters
    ----------
    predicted : dict[str, ExtractionValue]
        The features dictionary from an ExtractionResult.
    ground_truth : dict[str, Any]
        The ground truth annotations dictionary.
        
    Returns
    -------
    dict[str, dict[str, int]]
        Metrics counts per feature: TP, TN, FP_hallucination, FN_omission, alteration.
    """
    results = {}
    # Only score features present in ground truth.  Predicted fields without
    # ground truth (e.g. new schema fields with no annotation data) are
    # silently skipped — we cannot evaluate them.
    all_features = set(ground_truth.keys())
    
    for feature in all_features:
        p_val = predicted[feature].value if feature in predicted and predicted[feature] else None
        
        gt_entry = ground_truth.get(feature)
        if isinstance(gt_entry, dict):
            g_val = gt_entry.get("value")
        else:
            g_val = gt_entry
            
        tp = tn = fp_hallucination = fn_omission = alteration = 0
        
        def normalize(v):
            if v is None:
                return None
            if isinstance(v, bool):
                return "oui" if v else "non"
            if isinstance(v, (int, float)):
                if v == int(v):
                    return str(int(v))
                return str(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("", "na", "n/a", "null", "none"):
                    return None
                try:
                    f = float(s)
                    if f == int(f):
                        return str(int(f))
                except ValueError:
                    pass
                return s
            return str(v).strip().lower()
            
        p_val_norm = normalize(p_val)
        g_val_norm = normalize(g_val)
        
        # Attempt date normalization and chimios normalization
        if isinstance(p_val_norm, str) and isinstance(g_val_norm, str):
            p_date = _try_parse_date(p_val_norm)
            g_date = _try_parse_date(g_val_norm)
            if p_date and g_date:
                # Compare at the coarsest shared granularity
                min_len = min(len(p_date), len(g_date))
                p_val_norm = p_date[:min_len]
                g_val_norm = g_date[:min_len]
            elif " + " in p_val_norm and " + " in g_val_norm:
                p_val_norm = " + ".join(sorted(p_val_norm.split(" + ")))
                g_val_norm = " + ".join(sorted(g_val_norm.split(" + ")))
            elif p_val_norm.startswith("mute (") and g_val_norm.startswith("mute ("):
                p_span = p_val_norm[6:-1]
                g_span = g_val_norm[6:-1]
                if fuzz.ratio(p_span, g_span) >= _FUZZY_THRESHOLD:
                    p_val_norm = g_val_norm
            elif g_val_norm == "mute" and p_val_norm.startswith("mute ("):
                # Backwards compatibility: GS assumes "mute" without span
                p_val_norm = g_val_norm

        
        if p_val_norm == g_val_norm:
            if p_val_norm is not None:
                tp = 1
            else:
                tn = 1
        elif (feature in _FUZZY_ELIGIBLE_FIELDS
              and isinstance(p_val_norm, str) and isinstance(g_val_norm, str)
              and fuzz.ratio(p_val_norm, g_val_norm) >= _FUZZY_THRESHOLD):
            tp = 1
        else:
            if g_val_norm is None:
                fp_hallucination = 1
            elif p_val_norm is None:
                fn_omission = 1
            else:
                alteration = 1
                
        results[feature] = {
            "TP": tp,
            "TN": tn,
            "FP_hallucination": fp_hallucination,
            "FN_omission": fn_omission,
                "alteration": alteration,
            "extraction_tier": predicted[feature].extraction_tier if feature in predicted and predicted[feature] else "unknown"
        }
        
    return results

def compute_aggregate_metrics(all_results: list[dict[str, dict[str, int]]]) -> pd.DataFrame:
    """
    Aggregate per-feature metrics across all documents and compute overall metrics.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by feature name with columns for TP, TN, FP, FN, P, R, F1, and rates.
    """
    aggregated = {}
    for doc_result in all_results:
        for feature, counts in doc_result.items():
            if feature not in aggregated:
                aggregated[feature] = {"TP": 0, "TN": 0, "FP_hallucination": 0, "FN_omission": 0, "alteration": 0, "tiers": []}
            for k, v in counts.items():
                if k == "extraction_tier":
                    if v != "unknown":
                        aggregated[feature]["tiers"].append(v)
                else:
                    aggregated[feature][k] += v
                
    records = []
    for feature, counts in aggregated.items():
        tp = counts["TP"]
        tn = counts["TN"]
        fp_h = counts["FP_hallucination"]
        fn_o = counts["FN_omission"]
        alt = counts["alteration"]
        tiers = counts["tiers"]
        predominant_tier = max(set(tiers), key=tiers.count) if tiers else "unknown"
        
        fp_total = fp_h + _ALTERATION_WEIGHT * alt
        fn_total = fn_o + _ALTERATION_WEIGHT * alt
        
        precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
        recall = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        hallucination_rate = fp_h / (fp_h + tn) if (fp_h + tn) > 0 else 0.0
        
        actual_positives = tp + fn_o + alt
        alteration_rate = alt / actual_positives if actual_positives > 0 else 0.0
        omission_rate = fn_o / actual_positives if actual_positives > 0 else 0.0
        
        records.append({
            "feature": feature,
            "TP": tp,
            "TN": tn,
            "FP": fp_total,
            "FN": fn_total,
            "FP_hallucination": fp_h,
            "FN_omission": fn_o,
            "alteration": alt,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Hallucination_Rate": hallucination_rate,
            "Alteration_Rate": alteration_rate,
            "Omission_Rate": omission_rate,
            "Predominant_Tier": predominant_tier
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("feature")
    return df


# ---------------------------------------------------------------------------
# Per-category aggregate metrics (Phase 5.1)
# ---------------------------------------------------------------------------

_FEATURE_CATEGORIES: dict[str, list[str]] = {
    "demographics": [
        "date_rcp", "annee_de_naissance", "sexe",
        "activite_professionnelle", "antecedent_tumoral",
    ],
    "care_team": [
        "neuroncologue", "neurochirurgien", "radiotherapeute",
        "anatomo_pathologiste", "localisation_radiotherapie", "localisation_chir",
    ],
    "dates": [
        "date_chir", "chm_date_debut", "chm_date_fin",
        "rx_date_debut", "rx_date_fin", "date_1er_symptome",
        "exam_radio_date_decouverte", "date_progression", "date_deces", "dn_date",
    ],
    "symptoms": [
        "epilepsie_1er_symptome", "ceph_hic_1er_symptome", "deficit_1er_symptome",
        "cognitif_1er_symptome", "autre_trouble_1er_symptome",
        "epilepsie", "ceph_hic", "deficit", "cognitif", "autre_trouble",
        "ik_clinique",
    ],
    "radiology": [
        "contraste_1er_symptome", "prise_de_contraste",
        "oedeme_1er_symptome", "calcif_1er_symptome",
    ],
    "tumour_location": [
        "tumeur_lateralite", "tumeur_position", "dominance_cerebrale",
    ],
    "evolution": [
        "evol_clinique", "reponse_radiologique",
        "progress_clinique", "progress_radiologique",
        "survie_globale", "infos_deces",
    ],
    "treatment": [
        "chimios", "chimio_protocole", "chm_cycles",
        "type_chirurgie", "qualite_exerese",
        "rx_dose", "rx_fractionnement",
        "anti_epileptiques", "essai_therapeutique", "corticoides", "optune",
    ],
    "bio_ihc": [
        "ihc_idh1", "ihc_p53", "ihc_atrx", "ihc_fgfr3", "ihc_braf",
        "ihc_hist_h3k27m", "ihc_hist_h3k27me3", "ihc_egfr_hirsch",
        "ihc_gfap", "ihc_olig2", "ihc_ki67", "ihc_mmr",
    ],
    "bio_molecular": [
        "mol_idh1", "mol_idh2", "mol_tert", "mol_CDKN2A", "mol_h3f3a",
        "mol_hist1h3b", "mol_braf", "mol_mgmt", "mol_fgfr1", "mol_egfr_mut",
        "mol_prkca", "mol_p53", "mol_pten", "mol_cic", "mol_fubp1", "mol_atrx",
    ],
    "bio_chromosomal": [
        "ch1p", "ch19q", "ch1p19q_codel",
        "ch10p", "ch10q", "ch7p", "ch7q", "ch9p", "ch9q",
        "ampli_mdm2", "ampli_cdk4", "ampli_egfr", "ampli_met", "ampli_mdm4",
        "fusion_fgfr", "fusion_ntrk", "fusion_autre",
    ],
    "bio_diagnosis": [
        "diag_histologique", "diag_integre", "classification_oms", "grade",
        "histo_necrose", "histo_pec", "histo_mitoses",
        "aspect_cellulaire", "num_labo",
    ],
}


def _validate_feature_categories() -> None:
    """Warn if any schema field is missing from _FEATURE_CATEGORIES."""
    try:
        from src.extraction.schema import ALL_FIELDS_BY_NAME
    except ImportError:
        return
    categorized: set[str] = set()
    for fields in _FEATURE_CATEGORIES.values():
        categorized.update(fields)
    missing = set(ALL_FIELDS_BY_NAME.keys()) - categorized
    if missing:
        import warnings
        warnings.warn(
            f"Fields missing from _FEATURE_CATEGORIES in metrics.py: {sorted(missing)}"
        )


_validate_feature_categories()


def compute_category_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute macro-average F1 per feature category.

    Parameters
    ----------
    df : pd.DataFrame
        Per-feature metrics DataFrame (as returned by ``compute_aggregate_metrics``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by category with columns: F1_mean, Precision_mean,
        Recall_mean, num_features.
    """
    if df.empty:
        return pd.DataFrame()

    records = []
    for category, features in _FEATURE_CATEGORIES.items():
        present = [f for f in features if f in df.index and df.loc[f, "Predominant_Tier"] != "unknown"]
        num_total_features = len(features)
        
        if not present:
            records.append({
                "category": category,
                "F1_mean": 0.0,
                "Precision_mean": 0.0,
                "Recall_mean": 0.0,
                "num_features_attempted": 0,
                "num_features_total": num_total_features,
            })
            continue

        subset = df.loc[present]
        records.append({
            "category": category,
            "F1_mean": subset["F1"].mean(),
            "Precision_mean": subset["Precision"].mean(),
            "Recall_mean": subset["Recall"].mean(),
            "num_features_attempted": len(present),
            "num_features_total": num_total_features,
        })

    cat_df = pd.DataFrame(records)
    if not cat_df.empty:
        cat_df = cat_df.set_index("category")
    return cat_df

def compute_tier_category_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute F1 mean per category, broken down by Predominant_Tier."""
    if df.empty or "Predominant_Tier" not in df.columns:
        return pd.DataFrame()
        
    records = []
    for category, features in _FEATURE_CATEGORIES.items():
        present = [f for f in features if f in df.index and df.loc[f, "Predominant_Tier"] != "unknown"]
        if not present:
            continue
            
        subset = df.loc[present]
        for tier in subset["Predominant_Tier"].unique():
            tier_subset = subset[subset["Predominant_Tier"] == tier]
            if not tier_subset.empty:
                records.append({
                    "category": category,
                    "tier": tier,
                    "F1_mean": tier_subset["F1"].mean(),
                    "num_features": len(tier_subset)
                })
                
    if records:
        tier_df = pd.DataFrame(records)
        return tier_df.set_index(["category", "tier"])
    return pd.DataFrame()
