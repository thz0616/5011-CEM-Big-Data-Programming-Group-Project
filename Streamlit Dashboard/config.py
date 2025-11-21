# config.py
# Configuration constants for the Student Success Analytics Dashboard

import os

# ========================
# Database Configuration
# ========================
DATABASE_NAME = "student_dashboard.db"
DATABASE_PATH = os.path.join(os.getcwd(), DATABASE_NAME)

# ========================
# Model Configuration
# ========================
MODEL_ASSETS_DIR = "mlp_model_assets"
MODEL_PATH = os.path.join(MODEL_ASSETS_DIR, "mlp_model.keras")  # Added .keras extension
SCALER_PATH = os.path.join(MODEL_ASSETS_DIR, "scaler.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_ASSETS_DIR, "feature_columns.joblib")
META_JSON_PATH = "meta.json"

# Training data
TRAINING_DATA_PATH = "finaldropoutgraduate.csv"

# Model parameters (matches reference MLP implementation for 0.9215 accuracy)
MODEL_CONFIG = {
    'hidden_layer_1': 128,
    'hidden_layer_2': 64,
    'dropout_1': 0.5,
    'dropout_2': 0.3,
    'learning_rate': 1e-3,
    'epochs': 100,  # Matches reference code for reproducibility
    'batch_size': 64,
    'test_size': 0.2,
    'random_state': 42
}

# ========================
# Risk Thresholds
# ========================
RISK_THRESHOLDS = {
    'mild': (0.50, 0.65),
    'moderate': (0.65, 0.85),
    'severe': (0.85, 1.00)
}

RISK_COLORS = {
    'None': '#28a745',      # Green
    'Mild': '#FFD700',      # Gold/Yellow
    'Moderate': '#FF8C00',  # Dark Orange
    'Severe': '#DC143C'     # Crimson Red
}

# ========================
# Default User Credentials
# ========================
DEFAULT_ADMIN = {
    'username': 'admin',
    'password': 'admin123',
    'full_name': 'System Administrator',
    'role': 'admin'
}

DEFAULT_EDUCATOR_PASSWORD = '0000'

# ========================
# Student Features (36 attributes)
# ========================
STUDENT_FEATURES = [
    'Marital status',
    'Application mode',
    'Application order',
    'Course',
    'Daytime/evening attendance',
    'Previous qualification',
    'Previous qualification (grade)',
    'Nacionality',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    'Admission grade',
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'Age at enrollment',
    'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

# Mapping from display names to database column names
FEATURE_NAME_MAPPING = {
    'Marital status': 'marital_status',
    'Application mode': 'application_mode',
    'Application order': 'application_order',
    'Course': 'course',
    'Daytime/evening attendance': 'daytime_evening_attendance',
    'Previous qualification': 'previous_qualification',
    'Previous qualification (grade)': 'previous_qualification_grade',
    'Nacionality': 'nacionality',
    "Mother's qualification": 'mothers_qualification',
    "Father's qualification": 'fathers_qualification',
    "Mother's occupation": 'mothers_occupation',
    "Father's occupation": 'fathers_occupation',
    'Admission grade': 'admission_grade',
    'Displaced': 'displaced',
    'Educational special needs': 'educational_special_needs',
    'Debtor': 'debtor',
    'Tuition fees up to date': 'tuition_fees_up_to_date',
    'Gender': 'gender',
    'Scholarship holder': 'scholarship_holder',
    'Age at enrollment': 'age_at_enrollment',
    'International': 'international',
    'Curricular units 1st sem (credited)': 'curricular_units_1st_sem_credited',
    'Curricular units 1st sem (enrolled)': 'curricular_units_1st_sem_enrolled',
    'Curricular units 1st sem (evaluations)': 'curricular_units_1st_sem_evaluations',
    'Curricular units 1st sem (approved)': 'curricular_units_1st_sem_approved',
    'Curricular units 1st sem (grade)': 'curricular_units_1st_sem_grade',
    'Curricular units 1st sem (without evaluations)': 'curricular_units_1st_sem_without_evaluations',
    'Curricular units 2nd sem (credited)': 'curricular_units_2nd_sem_credited',
    'Curricular units 2nd sem (enrolled)': 'curricular_units_2nd_sem_enrolled',
    'Curricular units 2nd sem (evaluations)': 'curricular_units_2nd_sem_evaluations',
    'Curricular units 2nd sem (approved)': 'curricular_units_2nd_sem_approved',
    'Curricular units 2nd sem (grade)': 'curricular_units_2nd_sem_grade',
    'Curricular units 2nd sem (without evaluations)': 'curricular_units_2nd_sem_without_evaluations',
    'Unemployment rate': 'unemployment_rate',
    'Inflation rate': 'inflation_rate',
    'GDP': 'gdp'
}

# Reverse mapping
DB_TO_DISPLAY_MAPPING = {v: k for k, v in FEATURE_NAME_MAPPING.items()}

# ========================
# Session Configuration
# ========================
SESSION_TIMEOUT_MINUTES = 60

# ========================
# UI Configuration
# ========================
PAGE_TITLE = "Student Success Analytics"
PAGE_ICON = "📊"
LAYOUT = "wide"

# ========================
# File Upload Configuration
# ========================
MAX_CSV_SIZE_MB = 10
ALLOWED_CSV_EXTENSIONS = ['.csv']
CSV_SEPARATOR = ';'

# ========================
# Visualization Configuration
# ========================
PLOT_HEIGHT = 500
PLOT_WIDTH = 800
HEATMAP_COLORSCALE = 'RdBu_r'

# ========================
# Feature Value Label Mappings (Codebook)
# ========================
FEATURE_VALUE_LABELS = {
    'Marital status': {
        1: 'Single',
        2: 'Married',
        3: 'Widower',
        4: 'Divorced',
        5: 'Facto union',
        6: 'Legally separated'
    },
    'marital_status': {  # DB column name
        1: 'Single',
        2: 'Married',
        3: 'Widower',
        4: 'Divorced',
        5: 'Facto union',
        6: 'Legally separated'
    },
    'Application mode': {
        1: '1st phase - general contingent',
        2: 'Ordinance No. 612/93',
        5: '1st phase - special contingent (Azores Island)',
        7: 'Holders of other higher courses',
        10: 'Ordinance No. 854-B/99',
        15: 'International student (bachelor)',
        16: '1st phase - special contingent (Madeira Island)',
        17: '2nd phase - general contingent',
        18: '3rd phase - general contingent',
        26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
        27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
        39: 'Over 23 years old',
        42: 'Transfer',
        43: 'Change of course',
        44: 'Technological specialization diploma holders',
        51: 'Change of institution/course',
        53: 'Short cycle diploma holders',
        57: 'Change of institution/course (International)'
    },
    'application_mode': {  # DB column name
        1: '1st phase - general contingent',
        2: 'Ordinance No. 612/93',
        5: '1st phase - special contingent (Azores Island)',
        7: 'Holders of other higher courses',
        10: 'Ordinance No. 854-B/99',
        15: 'International student (bachelor)',
        16: '1st phase - special contingent (Madeira Island)',
        17: '2nd phase - general contingent',
        18: '3rd phase - general contingent',
        26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
        27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
        39: 'Over 23 years old',
        42: 'Transfer',
        43: 'Change of course',
        44: 'Technological specialization diploma holders',
        51: 'Change of institution/course',
        53: 'Short cycle diploma holders',
        57: 'Change of institution/course (International)'
    },
    'Application order': {
        0: '1st choice',
        1: '2nd choice',
        2: '3rd choice',
        3: '4th choice',
        4: '5th choice',
        5: '6th choice',
        6: '7th choice',
        7: '8th choice',
        8: '9th choice',
        9: '10th choice'
    },
    'application_order': {  # DB column name
        0: '1st choice',
        1: '2nd choice',
        2: '3rd choice',
        3: '4th choice',
        4: '5th choice',
        5: '6th choice',
        6: '7th choice',
        7: '8th choice',
        8: '9th choice',
        9: '10th choice'
    },
    'Course': {
        33: 'Biofuel Production Technologies',
        171: 'Animation and Multimedia Design',
        8014: 'Social Service (evening)',
        9003: 'Agronomy',
        9070: 'Communication Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering',
        9130: 'Equinculture',
        9147: 'Management',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising and Marketing Management',
        9773: 'Journalism and Communication',
        9853: 'Basic Education',
        9991: 'Management (evening)'
    },
    'course': {  # DB column name
        33: 'Biofuel Production Technologies',
        171: 'Animation and Multimedia Design',
        8014: 'Social Service (evening)',
        9003: 'Agronomy',
        9070: 'Communication Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering',
        9130: 'Equinculture',
        9147: 'Management',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising and Marketing Management',
        9773: 'Journalism and Communication',
        9853: 'Basic Education',
        9991: 'Management (evening)'
    },
    'Daytime/evening attendance': {
        1: 'Daytime',
        0: 'Evening'
    },
    'daytime_evening_attendance': {  # DB column name
        1: 'Daytime',
        0: 'Evening'
    },
    'Previous qualification': {
        1: 'Secondary education',
        2: "Higher education - bachelor's",
        3: 'Higher education - degree',
        4: "Higher education - master's",
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year not completed',
        10: '11th year not completed',
        12: 'Other - 11th year',
        14: '10th year',
        15: '10th year not completed',
        19: 'Basic education 3rd cycle',
        38: 'Basic education 2nd cycle',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)'
    },
    'previous_qualification': {  # DB column name
        1: 'Secondary education',
        2: "Higher education - bachelor's",
        3: 'Higher education - degree',
        4: "Higher education - master's",
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year not completed',
        10: '11th year not completed',
        12: 'Other - 11th year',
        14: '10th year',
        15: '10th year not completed',
        19: 'Basic education 3rd cycle',
        38: 'Basic education 2nd cycle',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)'
    },
    'Nacionality': {
        1: 'Portuguese',
        2: 'German',
        6: 'Spanish',
        11: 'Italian',
        13: 'Dutch',
        14: 'English',
        17: 'Lithuanian',
        21: 'Angolan',
        22: 'Cape Verdean',
        24: 'Guinean',
        25: 'Mozambican',
        26: 'Santomean',
        32: 'Turkish',
        41: 'Brazilian',
        62: 'Romanian',
        100: 'Moldovan',
        101: 'Mexican',
        103: 'Ukrainian',
        105: 'Russian',
        108: 'Cuban',
        109: 'Colombian'
    },
    'nacionality': {  # DB column name
        1: 'Portuguese',
        2: 'German',
        6: 'Spanish',
        11: 'Italian',
        13: 'Dutch',
        14: 'English',
        17: 'Lithuanian',
        21: 'Angolan',
        22: 'Cape Verdean',
        24: 'Guinean',
        25: 'Mozambican',
        26: 'Santomean',
        32: 'Turkish',
        41: 'Brazilian',
        62: 'Romanian',
        100: 'Moldovan',
        101: 'Mexican',
        103: 'Ukrainian',
        105: 'Russian',
        108: 'Cuban',
        109: 'Colombian'
    },
    "Mother's occupation": {
        0: 'Student',
        1: 'Representative of legislative/executive bodies, directors, managers',
        2: 'Specialists in intellectual and scientific activities',
        3: 'Intermediate level technicians and professions',
        4: 'Administrative staff',
        5: 'Personal services, security and safety workers, and sellers',
        6: 'Farmers and skilled workers in agriculture, fisheries and forestry',
        7: 'Skilled workers in industry, construction and craftsmen',
        8: 'Installation and machine operators and assembly workers',
        9: 'Unskilled workers',
        10: 'Armed Forces Professions',
        90: 'Other Situation',
        99: '(blank/unknown)'
    },
    'mothers_occupation': {  # DB column name
        0: 'Student',
        1: 'Representative of legislative/executive bodies, directors, managers',
        2: 'Specialists in intellectual and scientific activities',
        3: 'Intermediate level technicians and professions',
        4: 'Administrative staff',
        5: 'Personal services, security and safety workers, and sellers',
        6: 'Farmers and skilled workers in agriculture, fisheries and forestry',
        7: 'Skilled workers in industry, construction and craftsmen',
        8: 'Installation and machine operators and assembly workers',
        9: 'Unskilled workers',
        10: 'Armed Forces Professions',
        90: 'Other Situation',
        99: '(blank/unknown)'
    },
    "Father's occupation": {
        0: 'Student',
        1: 'Representative of legislative/executive bodies, directors, managers',
        2: 'Specialists in intellectual and scientific activities',
        3: 'Intermediate level technicians and professions',
        4: 'Administrative staff',
        5: 'Personal services, security and safety workers, and sellers',
        6: 'Farmers and skilled workers in agriculture, fisheries and forestry',
        7: 'Skilled workers in industry, construction and craftsmen',
        8: 'Installation and machine operators and assembly workers',
        9: 'Unskilled workers',
        10: 'Armed Forces Professions',
        90: 'Other Situation',
        99: '(blank/unknown)'
    },
    'fathers_occupation': {  # DB column name
        0: 'Student',
        1: 'Representative of legislative/executive bodies, directors, managers',
        2: 'Specialists in intellectual and scientific activities',
        3: 'Intermediate level technicians and professions',
        4: 'Administrative staff',
        5: 'Personal services, security and safety workers, and sellers',
        6: 'Farmers and skilled workers in agriculture, fisheries and forestry',
        7: 'Skilled workers in industry, construction and craftsmen',
        8: 'Installation and machine operators and assembly workers',
        9: 'Unskilled workers',
        10: 'Armed Forces Professions',
        90: 'Other Situation',
        99: '(blank/unknown)'
    },
    "Mother's qualification": {
        1: 'Secondary education',
        2: "Higher education - bachelor's",
        3: 'Higher education - degree',
        4: "Higher education - master's",
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year not completed',
        10: '11th year not completed',
        12: 'Other - 11th year',
        14: '10th year',
        15: '10th year not completed',
        19: 'Basic education 3rd cycle',
        38: 'Basic education 2nd cycle',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)'
    },
    'mothers_qualification': {  # DB column name
        1: 'Secondary education',
        2: "Higher education - bachelor's",
        3: 'Higher education - degree',
        4: "Higher education - master's",
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year not completed',
        10: '11th year not completed',
        12: 'Other - 11th year',
        14: '10th year',
        15: '10th year not completed',
        19: 'Basic education 3rd cycle',
        38: 'Basic education 2nd cycle',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)'
    },
    "Father's qualification": {
        1: 'Secondary education',
        2: "Higher education - bachelor's",
        3: 'Higher education - degree',
        4: "Higher education - master's",
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year not completed',
        10: '11th year not completed',
        12: 'Other - 11th year',
        14: '10th year',
        15: '10th year not completed',
        19: 'Basic education 3rd cycle',
        38: 'Basic education 2nd cycle',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)'
    },
    'fathers_qualification': {  # DB column name
        1: 'Secondary education',
        2: "Higher education - bachelor's",
        3: 'Higher education - degree',
        4: "Higher education - master's",
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year not completed',
        10: '11th year not completed',
        12: 'Other - 11th year',
        14: '10th year',
        15: '10th year not completed',
        19: 'Basic education 3rd cycle',
        38: 'Basic education 2nd cycle',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)'
    },
    'Displaced': {
        1: 'Yes',
        0: 'No'
    },
    'displaced': {  # DB column name
        1: 'Yes',
        0: 'No'
    },
    'Educational special needs': {
        1: 'Yes',
        0: 'No'
    },
    'educational_special_needs': {  # DB column name
        1: 'Yes',
        0: 'No'
    },
    'Debtor': {
        1: 'Yes',
        0: 'No'
    },
    'debtor': {  # DB column name
        1: 'Yes',
        0: 'No'
    },
    'Tuition fees up to date': {
        1: 'Yes',
        0: 'No'
    },
    'tuition_fees_up_to_date': {  # DB column name
        1: 'Yes',
        0: 'No'
    },
    'Gender': {
        1: 'Male',
        0: 'Female'
    },
    'gender': {  # DB column name
        1: 'Male',
        0: 'Female'
    },
    'Scholarship holder': {
        1: 'Yes',
        0: 'No'
    },
    'scholarship_holder': {  # DB column name
        1: 'Yes',
        0: 'No'
    },
    'International': {
        1: 'Yes',
        0: 'No'
    },
    'international': {  # DB column name
        1: 'Yes',
        0: 'No'
    }
}
