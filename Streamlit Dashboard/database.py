# database.py
# SQLite database operations for Student Success Analytics Dashboard

import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from config import *
from auth import hash_password


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn


def initialize_database():
    """Create database tables and default admin account."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'educator')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create students table with all 36 features
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            studentID INTEGER PRIMARY KEY AUTOINCREMENT,
            marital_status REAL,
            application_mode REAL,
            application_order REAL,
            course REAL,
            daytime_evening_attendance REAL,
            previous_qualification REAL,
            previous_qualification_grade REAL,
            nacionality REAL,
            mothers_qualification REAL,
            fathers_qualification REAL,
            mothers_occupation REAL,
            fathers_occupation REAL,
            admission_grade REAL,
            displaced REAL,
            educational_special_needs REAL,
            debtor REAL,
            tuition_fees_up_to_date REAL,
            gender REAL,
            scholarship_holder REAL,
            age_at_enrollment REAL,
            international REAL,
            curricular_units_1st_sem_credited REAL,
            curricular_units_1st_sem_enrolled REAL,
            curricular_units_1st_sem_evaluations REAL,
            curricular_units_1st_sem_approved REAL,
            curricular_units_1st_sem_grade REAL,
            curricular_units_1st_sem_without_evaluations REAL,
            curricular_units_2nd_sem_credited REAL,
            curricular_units_2nd_sem_enrolled REAL,
            curricular_units_2nd_sem_evaluations REAL,
            curricular_units_2nd_sem_approved REAL,
            curricular_units_2nd_sem_grade REAL,
            curricular_units_2nd_sem_without_evaluations REAL,
            unemployment_rate REAL,
            inflation_rate REAL,
            gdp REAL,
            prediction_probability REAL,
            risk_level TEXT,
            assigned_educator_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assigned_educator_id) REFERENCES users(id)
        )
    ''')
    
    # Create model_metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_features TEXT NOT NULL,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_path TEXT,
            scaler_path TEXT,
            feature_columns_path TEXT
        )
    ''')
    
    # Check if admin exists
    cursor.execute("SELECT * FROM users WHERE username = ?", (DEFAULT_ADMIN['username'],))
    if cursor.fetchone() is None:
        # Create default admin account
        hashed_pwd = hash_password(DEFAULT_ADMIN['password'])
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
            (DEFAULT_ADMIN['username'], hashed_pwd, DEFAULT_ADMIN['full_name'], DEFAULT_ADMIN['role'])
        )
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")


# ======================
# User Management
# ======================

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_all_educators() -> List[Dict[str, Any]]:
    """Get all educator accounts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, full_name, created_at FROM users WHERE role = 'educator' ORDER BY full_name")
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_all_usernames() -> List[str]:
    """Get all existing usernames."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users")
    rows = cursor.fetchall()
    conn.close()
    
    return [row['username'] for row in rows]


def create_educator(username: str, password: str, full_name: str) -> Tuple[bool, str]:
    """
    Create a new educator account.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Hash password
        hashed_pwd = hash_password(password)
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
            (username, hashed_pwd, full_name, 'educator')
        )
        
        conn.commit()
        conn.close()
        return True, f"Educator '{full_name}' created successfully"
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists"
    except Exception as e:
        return False, f"Error creating educator: {e}"


def update_educator(user_id: int, username: str, full_name: str) -> Tuple[bool, str]:
    """Update educator details."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE users SET username = ?, full_name = ? WHERE id = ? AND role = 'educator'",
            (username, full_name, user_id)
        )
        
        conn.commit()
        conn.close()
        
        if cursor.rowcount > 0:
            return True, "Educator updated successfully"
        else:
            return False, "Educator not found"
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists"
    except Exception as e:
        return False, f"Error updating educator: {e}"


def delete_educator(user_id: int) -> Tuple[bool, str]:
    """Delete educator account."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Unassign students first
        cursor.execute("UPDATE students SET assigned_educator_id = NULL WHERE assigned_educator_id = ?", (user_id,))
        
        # Delete educator
        cursor.execute("DELETE FROM users WHERE id = ? AND role = 'educator'", (user_id,))
        
        conn.commit()
        conn.close()
        
        if cursor.rowcount > 0:
            return True, "Educator deleted successfully"
        else:
            return False, "Educator not found"
    except Exception as e:
        return False, f"Error deleting educator: {e}"


def update_user_password(user_id: int, new_password: str) -> Tuple[bool, str]:
    """Update user password."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        hashed_pwd = hash_password(new_password)
        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hashed_pwd, user_id))
        
        conn.commit()
        conn.close()
        
        if cursor.rowcount > 0:
            return True, "Password updated successfully"
        else:
            return False, "User not found"
    except Exception as e:
        return False, f"Error updating password: {e}"


def update_user_username(user_id: int, new_username: str) -> Tuple[bool, str]:
    """Update user username (for educators only)."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("UPDATE users SET username = ? WHERE id = ? AND role = 'educator'", (new_username, user_id))
        
        conn.commit()
        conn.close()
        
        if cursor.rowcount > 0:
            return True, "Username updated successfully"
        else:
            return False, "Cannot update username"
    except sqlite3.IntegrityError:
        return False, f"Username '{new_username}' already exists"
    except Exception as e:
        return False, f"Error updating username: {e}"


# ======================
# Student Management
# ======================

def add_student(student_data: Dict[str, Any]) -> Tuple[bool, str, Optional[int]]:
    """
    Add a new student to the database.
    
    Args:
        student_data: Dictionary with student features (database column names)
        
    Returns:
        Tuple of (success, message, student_id)
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Build INSERT query dynamically
        columns = list(student_data.keys())
        placeholders = ', '.join(['?'] * len(columns))
        column_names = ', '.join(columns)
        
        query = f"INSERT INTO students ({column_names}) VALUES ({placeholders})"
        values = [student_data[col] for col in columns]
        
        cursor.execute(query, values)
        student_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return True, "Student added successfully", student_id
    except Exception as e:
        return False, f"Error adding student: {e}", None


def add_students_bulk(students_df: pd.DataFrame) -> Tuple[bool, str, int]:
    """
    Add multiple students from DataFrame.
    
    Returns:
        Tuple of (success, message, count_added)
    """
    try:
        conn = get_connection()
        
        # Ensure DataFrame has only valid columns
        valid_columns = [col for col in students_df.columns if col in FEATURE_NAME_MAPPING.values()]
        students_df = students_df[valid_columns]
        
        # Write to database
        students_df.to_sql('students', conn, if_exists='append', index=False)
        
        count = len(students_df)
        conn.close()
        
        return True, f"Successfully added {count} students", count
    except Exception as e:
        return False, f"Error adding students: {e}", 0


def get_all_students() -> pd.DataFrame:
    """Get all students as DataFrame."""
    conn = get_connection()
    query = "SELECT * FROM students ORDER BY studentID DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_student_by_id(student_id: int) -> Optional[Dict[str, Any]]:
    """Get student by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE studentID = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_students_by_educator(educator_id: int) -> pd.DataFrame:
    """Get all students assigned to an educator."""
    conn = get_connection()
    query = "SELECT * FROM students WHERE assigned_educator_id = ? ORDER BY studentID DESC"
    df = pd.read_sql_query(query, conn, params=(educator_id,))
    conn.close()
    return df


def assign_student_to_educator(student_id: int, educator_id: Optional[int]) -> Tuple[bool, str]:
    """Assign a student to an educator."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE students SET assigned_educator_id = ?, updated_at = CURRENT_TIMESTAMP WHERE studentID = ?",
            (educator_id, student_id)
        )
        
        conn.commit()
        conn.close()
        
        if cursor.rowcount > 0:
            return True, "Student assigned successfully"
        else:
            return False, "Student not found"
    except Exception as e:
        return False, f"Error assigning student: {e}"


def assign_students_bulk(student_ids: List[int], educator_id: int) -> Tuple[bool, str]:
    """Assign multiple students to an educator."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        placeholders = ', '.join(['?'] * len(student_ids))
        query = f"UPDATE students SET assigned_educator_id = ?, updated_at = CURRENT_TIMESTAMP WHERE studentID IN ({placeholders})"
        
        cursor.execute(query, [educator_id] + student_ids)
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return True, f"Successfully assigned {count} students"
    except Exception as e:
        return False, f"Error assigning students: {e}"


def update_student_prediction(student_id: int, probability: float, risk_level: str) -> Tuple[bool, str]:
    """Update student prediction results."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE students SET prediction_probability = ?, risk_level = ?, updated_at = CURRENT_TIMESTAMP WHERE studentID = ?",
            (probability, risk_level, student_id)
        )
        
        conn.commit()
        conn.close()
        
        return True, "Prediction updated"
    except Exception as e:
        return False, f"Error updating prediction: {e}"


def delete_student(student_id: int) -> Tuple[bool, str]:
    """Delete a student."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM students WHERE studentID = ?", (student_id,))
        
        conn.commit()
        conn.close()
        
        if cursor.rowcount > 0:
            return True, "Student deleted successfully"
        else:
            return False, "Student not found"
    except Exception as e:
        return False, f"Error deleting student: {e}"


def get_student_features_for_prediction(student_id: int) -> Optional[Dict[str, float]]:
    """Get student features in correct format for prediction."""
    student = get_student_by_id(student_id)
    
    if student is None:
        return None
    
    # Extract only the 36 feature columns
    features = {}
    for db_col in FEATURE_NAME_MAPPING.values():
        if db_col in student:
            features[db_col] = student[db_col]
    
    return features


def get_at_risk_students_count() -> Dict[str, int]:
    """Get count of students by risk level (including None/not yet predicted)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            CASE 
                WHEN risk_level IS NULL THEN 'None'
                ELSE risk_level
            END as risk_level,
            COUNT(*) as count
        FROM students
        GROUP BY risk_level
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    counts = {
        'None': 0,
        'Mild': 0,
        'Moderate': 0,
        'Severe': 0
    }
    
    for row in rows:
        if row['risk_level']:
            counts[row['risk_level']] = row['count']
    
    return counts


# ======================
# Model Metadata
# ======================

def save_model_metadata(trained_features: List[str], model_path: str, scaler_path: str, 
                       feature_columns_path: str) -> Tuple[bool, str]:
    """Save model training metadata."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        import json
        features_json = json.dumps(trained_features)
        
        cursor.execute("""
            INSERT INTO model_metadata (trained_features, model_path, scaler_path, feature_columns_path)
            VALUES (?, ?, ?, ?)
        """, (features_json, model_path, scaler_path, feature_columns_path))
        
        conn.commit()
        conn.close()
        
        return True, "Model metadata saved"
    except Exception as e:
        return False, f"Error saving metadata: {e}"


def get_latest_model_metadata() -> Optional[Dict[str, Any]]:
    """Get the most recent model metadata."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM model_metadata
        ORDER BY training_date DESC
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        metadata = dict(row)
        # Parse JSON features
        import json
        metadata['trained_features'] = json.loads(metadata['trained_features'])
        return metadata
    
    return None


def get_database_stats() -> Dict[str, int]:
    """Get database statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as count FROM students")
    total_students = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM students WHERE prediction_probability IS NOT NULL")
    predicted_students = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM students WHERE assigned_educator_id IS NOT NULL")
    assigned_students = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM users WHERE role = 'educator'")
    total_educators = cursor.fetchone()['count']
    
    conn.close()
    
    return {
        'total_students': total_students,
        'predicted_students': predicted_students,
        'assigned_students': assigned_students,
        'total_educators': total_educators
    }
