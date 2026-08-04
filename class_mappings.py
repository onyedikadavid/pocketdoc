# class_mappings.py

CLASS_NAMES = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]

CLEAN_LABELS = {
    "Acne and Rosacea Photos": "Acne & Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Actinic Keratosis / Pre-Malignant Lesion",
    "Atopic Dermatitis Photos": "Atopic Dermatitis",
    "Bullous Disease Photos": "Bullous Blistering Disease",
    "Cellulitis Impetigo and other Bacterial Infections": "Bacterial Infection (Cellulitis/Impetigo)",
    "Eczema Photos": "Eczema",
    "Exanthems and Drug Eruptions": "Drug-Induced Skin Eruption",
    "Hair Loss Photos Alopecia and other Hair Diseases": "Alopecia / Hair Loss Condition",
    "Herpes HPV and other STDs Photos": "Viral / STD Lesion",
    "Light Diseases and Disorders of Pigmentation": "Pigmentation Disorder",
    "Lupus and other Connective Tissue diseases": "Lupus / Connective Tissue Disease",
    "Melanoma Skin Cancer Nevi and Moles": "Melanoma / Skin Mole Evaluation",
    "Nail Fungus and other Nail Disease": "Nail Fungus / Nail Disorder",
    "Poison Ivy Photos and other Contact Dermatitis": "Contact Dermatitis / Poison Ivy",
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis / Lichen Planus",
    "Scabies Lyme Disease and other Infestations and Bites": "Parasitic Bite / Scabies / Lyme",
    "Seborrheic Keratoses and other Benign Tumors": "Seborrheic Keratosis (Benign)",
    "Systemic Disease": "Systemic Disease Rash",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Fungal Infection (Ringworm/Candida)",
    "Urticaria Hives": "Urticaria / Hives",
    "Vascular Tumors": "Vascular Tumor / Lesion",
    "Vasculitis Photos": "Vasculitis",
    "Warts Molluscum and other Viral Infections": "Warts / Viral Infection"
}

URGENT_CLASSES = {
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Melanoma Skin Cancer Nevi and Moles",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Vasculitis Photos",
    "Systemic Disease"
}

MODERATE_CLASSES = {
    "Bullous Disease Photos",
    "Lupus and other Connective Tissue diseases",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Herpes HPV and other STDs Photos",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Exanthems and Drug Eruptions"
}

def determine_triage_level(top_prediction_class: str, confidence: float) -> str:
    if top_prediction_class in URGENT_CLASSES and confidence >= 0.25:
        return "urgent"
    elif top_prediction_class in MODERATE_CLASSES or (confidence >= 0.60):
        return "moderate"
    else:
        return "low"