# app.py - Part 1
# Main Streamlit Application for Student Success Analytics Dashboard

import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Import custom modules
import database as db
import auth
from config import *
from utils import *
import model_training
import prediction
import feature_ranking
import visualization as viz

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'


# Authentication
def show_login_page():
    st.markdown('<div class="main-header">🎓 Student Success Analytics</div>', unsafe_allow_html=True)
    st.markdown("### Login to Continue")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    user = auth.authenticate_user(username, password, db)
                    
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.success(f"Welcome, {user['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        st.info("**Default Admin:** admin / admin123")


def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()


# Main function
def main():
    init_session_state()
    
    # Initialize database - check for tables, not just file existence
    try:
        import sqlite3
        needs_init = False
        
        if not os.path.exists(DATABASE_PATH):
            needs_init = True
        else:
            # Check if tables exist
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if cursor.fetchone() is None:
                needs_init = True
            conn.close()
        
        if needs_init:
            db.initialize_database()
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        db.initialize_database()
    
    # Check and train model on first run
    if not st.session_state.model_trained and check_training_data_exists():
        if not check_model_exists():
            st.info("🎯 Training model for the first time... Check your terminal/console for progress updates!")
            st.write("Training typically takes 1-2 minutes. Progress will appear in the terminal output.")
            
            # Train model (progress will show in terminal/console)
            success, message = model_training.train_model(TRAINING_DATA_PATH)
            
            if success:
                st.session_state.model_trained = True
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ Model training failed: {message}")
        else:
            st.session_state.model_trained = True
    
    # Show login or main app
    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_main_app()


def show_main_app():
    user = st.session_state.user
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {user['full_name']}")
        st.markdown(f"**Role:** {user['role'].title()}")
        st.markdown("---")
        
        if user['role'] == 'admin':
            pages = [
                "📊 Dashboard",
                "👨‍🏫 Manage Educators", 
                "👨‍🎓 Manage Students",
                "📈 Training Data Distribution",
                "📊 Input Data Distribution",
                "🗺️ Heatmap",
                "🏆 Factor Ranking",
                "👤 Profile"
            ]
        else:
            pages = [
                "📚 My Students",
                "👤 Profile"
            ]
        
        for page in pages:
            if st.button(page, use_container_width=True):
                st.session_state.page = page
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Main content
    page = st.session_state.page
    
    if user['role'] == 'admin':
        if "Dashboard" in page:
            show_admin_dashboard()
        elif "Manage Educators" in page:
            show_manage_educators()
        elif "Manage Students" in page:
            show_manage_students()
        elif "Training Data Distribution" in page:
            show_training_distribution()
        elif "Input Data Distribution" in page:
            show_input_distribution()
        elif "Heatmap" in page:
            show_heatmap()
        elif "Factor Ranking" in page:
            show_factor_ranking()
        elif "Profile" in page:
            show_profile()
    else:
        if "My Students" in page:
            show_educator_students()
        elif "Profile" in page:
            show_profile()


def show_admin_dashboard():
    st.markdown('<div class="main-header">📊 Admin Dashboard</div>', unsafe_allow_html=True)
    
    stats = db.get_database_stats()
    risk_counts = db.get_at_risk_students_count()
    stats['risk_counts'] = risk_counts
    
    # Display metrics using Streamlit native components
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👨‍🎓 Total Students",
            value=stats.get('total_students', 0)
        )
        st.metric(
            label="🟢 None Risk",
            value=stats.get('risk_counts', {}).get('None', 0)
        )
    
    with col2:
        st.metric(
            label="🎯 Predicted Students", 
            value=stats.get('predicted_students', 0)
        )
        st.metric(
            label="🟡 Mild Risk",
            value=stats.get('risk_counts', {}).get('Mild', 0)
        )
    
    with col3:
        st.metric(
            label="🟠 Moderate Risk",
            value=stats.get('risk_counts', {}).get('Moderate', 0)
        )
        st.metric(
            label="🔴 Severe Risk",
            value=stats.get('risk_counts', {}).get('Severe', 0)
        )
    
    with col4:
        st.metric(
            label="👨‍🏫 Total Educators",
            value=stats.get('total_educators', 0)
        )
    
    st.markdown("---")
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add Student", use_container_width=True):
            st.session_state.page = '👨‍🎓 Manage Students'
            st.rerun()
    
    with col2:
        if st.button("🎯 Run Predictions", use_container_width=True):
            with st.spinner("Running predictions..."):
                success, message, count = prediction.predict_all_students()
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    with col3:
        if st.button("👨‍🏫 Manage Educators", use_container_width=True):
            st.session_state.page = '👨‍🏫 Manage Educators'
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Student Distribution")
    
    students_df = db.get_all_students()
    
    if len(students_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(viz.create_risk_distribution_pie(risk_counts), use_container_width=True)
        with col2:
            st.plotly_chart(viz.create_risk_bar_chart(risk_counts), use_container_width=True)
    else:
        st.info("No students found in database.")


def show_manage_educators():
    st.markdown('<div class="main-header">👨‍🏫 Manage Educators</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📋 View Educators", "➕ Add New"])
    
    with tab1:
        educators = db.get_all_educators()
        
        if len(educators) > 0:
            for educator in educators:
                with st.expander(f"👤 {educator['full_name']} (@{educator['username']})"):
                    st.write(f"**ID:** {educator['id']}")
                    st.write(f"**Username:** {educator['username']}")
                    
                    if st.button("🗑️ Delete", key=f"del_{educator['id']}"):
                        success, message = db.delete_educator(educator['id'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        else:
            st.info("No educators found.")
    
    with tab2:
        with st.form("add_educator"):
            full_name = st.text_input("Full Name")
            st.info("Username: auto-generated | Password: 0000")
            
            if st.form_submit_button("Create", use_container_width=True):
                if full_name:
                    base_username = extract_first_name(full_name)
                    username = generate_unique_username(base_username, db.get_all_usernames())
                    success, message = db.create_educator(username, DEFAULT_EDUCATOR_PASSWORD, full_name)
                    
                    if success:
                        st.success(f"✅ Created! Username: {username}")
                        st.rerun()
                    else:
                        st.error(message)


def show_manage_students():
    st.markdown('<div class="main-header">👨‍🎓 Manage Students</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 View", "➕ Add Manually", "📁 Upload CSV", "🔗 Assign", "🗑️ Remove Student"])
    
    with tab1:
        students_df = db.get_all_students()
        if len(students_df) > 0:
            st.write(f"**Total:** {len(students_df)}")
            # Show all students (removed .head(50) limit)
            st.dataframe(students_df[['studentID', 'risk_level', 'prediction_probability', 'assigned_educator_id']], use_container_width=True, height=600)
        else:
            st.info("No students found.")
    
    with tab2:
        st.markdown("### Add Student Manually")
        with st.form("add_student"):
            cols = st.columns(3)
            student_data = {}
            
            dropdown_fields = {
                'Marital status',
                'Application mode',
                'Application order',
                'Course',
                'Daytime/evening attendance',
                'Previous qualification',
                'Nacionality',
                "Mother's occupation",
                "Father's occupation",
                'Displaced',
                'Educational special needs',
                'Debtor',
                'Tuition fees up to date',
                'Gender',
                'Scholarship holder',
                'International'
            }
            
            for idx, (display_name, db_col) in enumerate(FEATURE_NAME_MAPPING.items()):
                with cols[idx % 3]:
                    if display_name in dropdown_fields and display_name in FEATURE_VALUE_LABELS:
                        options = sorted(FEATURE_VALUE_LABELS[display_name].items(), key=lambda kv: kv[0])
                        selected = st.selectbox(display_name, options, format_func=lambda x: x[1])
                        student_data[db_col] = float(selected[0])
                    else:
                        student_data[db_col] = st.number_input(display_name, value=0.0, step=0.1)
            
            if st.form_submit_button("Add Student"):
                success, message, sid = db.add_student(student_data)
                if success:
                    st.success(f"✅ {message} (ID: {sid})")
                else:
                    st.error(message)
    
    with tab3:
        st.markdown("### Upload CSV File")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, sep=';')
                is_valid, missing = validate_csv_features(df, STUDENT_FEATURES)
                
                if not is_valid:
                    st.error(f"❌ Missing features: {', '.join(missing)}")
                else:
                    df_mapped = map_csv_columns_to_db(df)
                    st.write(f"Preview: {len(df_mapped)} students")
                    st.dataframe(df_mapped.head())
                    
                    if st.button("Import Students"):
                        success, message, count = db.add_students_bulk(df_mapped)
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(message)
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab4:
        st.markdown("### Assign Students to Educators")
        students_df = db.get_all_students()
        educators = db.get_all_educators()
        
        if len(students_df) > 0 and len(educators) > 0:
            selected_students = st.multiselect(
                "Select Students",
                options=students_df['studentID'].tolist(),
                format_func=lambda x: f"Student ID: {x}"
            )
            
            educator_dict = {f"{e['full_name']} (@{e['username']})": e['id'] for e in educators}
            selected_educator = st.selectbox("Assign to Educator", list(educator_dict.keys()))
            
            if st.button("Assign Selected"):
                if selected_students:
                    educator_id = educator_dict[selected_educator]
                    success, message = db.assign_students_bulk(selected_students, educator_id)
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please select students")
        else:
            st.info("Add students and educators first.")

    with tab5:
        st.markdown("### Remove Student")
        students_df = db.get_all_students()
        if len(students_df) == 0:
            st.info("No students available to remove.")
        else:
            sid = st.selectbox(
                "Select Student ID",
                options=students_df['studentID'].tolist(),
                format_func=lambda x: f"Student ID: {x}"
            )
            if st.button("Delete Selected Student", key="delete_student_button"):
                success, message = db.delete_student(int(sid))
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)


def show_training_distribution():
    st.markdown('<div class="main-header">📈 Training Data Distribution</div>', unsafe_allow_html=True)
    
    if not check_training_data_exists():
        st.error("Training data not found!")
        return
    
    df = parse_csv_with_separator(TRAINING_DATA_PATH, CSV_SEPARATOR)
    
    blocked_display = {"Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"}
    blocked_db = {FEATURE_NAME_MAPPING[name] for name in blocked_display if name in FEATURE_NAME_MAPPING}
    feature_options = [col for col in df.columns.tolist() if col not in blocked_display and col not in blocked_db]
    
    feature = st.selectbox("Select Feature", feature_options)
    
    if feature:
        fig = viz.create_distribution_plot(df[feature], feature)
        st.plotly_chart(fig, use_container_width=True)


def show_input_distribution():
    st.markdown('<div class="main-header">📊 Input Data Distribution</div>', unsafe_allow_html=True)
    
    students_df = db.get_all_students()
    
    if len(students_df) == 0:
        st.info("No student data available.")
        return
    
    blocked_display = {"Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"}
    blocked_db = {FEATURE_NAME_MAPPING[name] for name in blocked_display if name in FEATURE_NAME_MAPPING}
    feature_cols = [col for col in students_df.columns if col in FEATURE_NAME_MAPPING.values() and col not in blocked_db]
    display_names = [get_display_name(col) for col in feature_cols]
    
    selected_display = st.selectbox("Select Feature", display_names)
    selected_col = get_db_column_name(selected_display)
    
    if selected_col in students_df.columns:
        fig = viz.create_distribution_plot(students_df[selected_col], selected_display)
        st.plotly_chart(fig, use_container_width=True)


def show_heatmap():
    st.markdown('<div class="main-header">🗺️ Correlation Heatmap</div>', unsafe_allow_html=True)
    
    if not check_training_data_exists():
        st.error("Training data not found!")
        return
    
    with st.spinner("Generating heatmap..."):
        fig = viz.create_correlation_heatmap(TRAINING_DATA_PATH)
        st.plotly_chart(fig, use_container_width=True)


def show_factor_ranking():
    st.markdown('<div class="main-header">🏆 Feature Importance Ranking</div>', unsafe_allow_html=True)
    
    if not check_training_data_exists():
        st.error("Training data not found!")
        return
    
    with st.spinner("Calculating feature importance..."):
        success, message, importance_df = feature_ranking.calculate_feature_importance(TRAINING_DATA_PATH)
        
        if success:
            st.success(f"✅ {message}")
            
            top_n = st.slider("Number of top features", 10, 30, 20)
            fig = viz.create_feature_importance_chart(importance_df, top_n)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Feature Importance Table")
            st.dataframe(importance_df.head(top_n), use_container_width=True)
        else:
            st.error(f"❌ {message}")


def show_model_retraining():
    st.markdown('<div class="main-header">🔄 Model Retraining</div>', unsafe_allow_html=True)
    
    st.markdown("### Upload New Training Data")
    st.warning("This will replace the current training data and retrain the model.")
    
    uploaded_file = st.file_uploader("Upload Training CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=';')
            
            if 'Target' not in df.columns and 'target' not in df.columns:
                st.error("❌ Target column not found!")
            else:
                st.write(f"**Rows:** {len(df)}")
                st.write(f"**Columns:** {len(df.columns)}")
                st.dataframe(df.head())
                
                if st.button("Start Retraining"):
                    st.info("🔄 Retraining model... Check terminal/console for progress!")
                    
                    # Save uploaded file
                    temp_path = "temp_training_data.csv"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Retrain (progress shown in terminal)
                    success, message = model_training.retrain_model_with_new_data(temp_path, TRAINING_DATA_PATH)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.session_state.model_trained = True
                    else:
                        st.error(f"❌ {message}")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        except Exception as e:
            st.error(f"Error: {e}")


def show_profile():
    st.markdown('<div class="main-header">👤 Profile</div>', unsafe_allow_html=True)
    
    user = st.session_state.user
    
    st.markdown(f"### {user['full_name']}")
    st.write(f"**Username:** {user['username']}")
    st.write(f"**Role:** {user['role'].title()}")
    
    st.markdown("---")
    st.markdown("### Change Password")
    
    with st.form("change_password"):
        current_pw = st.text_input("Current Password", type="password")
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Update Password"):
            if not current_pw or not new_pw or not confirm_pw:
                st.error("All fields required")
            elif new_pw != confirm_pw:
                st.error("Passwords don't match")
            else:
                user_data = db.get_user_by_id(user['id'])
                if auth.verify_password(current_pw, user_data['password_hash']):
                    success, message = db.update_user_password(user['id'], new_pw)
                    if success:
                        st.success("✅ Password updated!")
                    else:
                        st.error(message)
                else:
                    st.error("Current password incorrect")
    
    if user['role'] == 'educator':
        st.markdown("---")
        st.markdown("### Change Username")
        
        with st.form("change_username"):
            new_username = st.text_input("New Username")
            
            if st.form_submit_button("Update Username"):
                if new_username:
                    if auth.check_username_available(new_username, db, user['id']):
                        success, message = db.update_user_username(user['id'], new_username)
                        if success:
                            st.success("✅ Username updated! Please login again.")
                            logout()
                        else:
                            st.error(message)
                    else:
                        st.error("Username already taken")


def show_educator_students():
    st.markdown('<div class="main-header">📚 My Students</div>', unsafe_allow_html=True)
    
    user_id = st.session_state.user['id']
    students_df = db.get_students_by_educator(user_id)
    
    if len(students_df) == 0:
        st.info("No students assigned to you yet.")
        return
    
    st.write(f"**Total Students:** {len(students_df)}")
    
    counts_series = students_df['risk_level'].fillna('None').replace('', 'None').value_counts()
    risk_counts = {
        'None': int(counts_series.get('None', 0)),
        'Mild': int(counts_series.get('Mild', 0)),
        'Moderate': int(counts_series.get('Moderate', 0)),
        'Severe': int(counts_series.get('Severe', 0))
    }
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.create_risk_distribution_pie(risk_counts), use_container_width=True)
    with col2:
        st.plotly_chart(viz.create_risk_bar_chart(risk_counts), use_container_width=True)
    
    at_risk = students_df[students_df['risk_level'].isin(['Mild', 'Moderate', 'Severe'])]
    
    if len(at_risk) > 0:
        st.markdown("### 🚨 At-Risk Students")
        
        for _, student in at_risk.iterrows():
            color = get_risk_color(student['risk_level'])
            prob = format_probability(student['prediction_probability']) if pd.notna(student['prediction_probability']) else 'N/A'
            
            with st.expander(f"Student ID: {student['studentID']} - {student['risk_level']} Risk"):
                st.markdown(f"**Risk Level:** <span style='background-color:{color}; padding:5px; border-radius:3px; color:white;'>{student['risk_level']}</span>", unsafe_allow_html=True)
                st.write(f"**Dropout Probability:** {prob}")
                st.write(f"**Student ID:** {student['studentID']}")
    else:
        st.success("✅ No at-risk students!")
    
    st.markdown("---")
    st.markdown("### All My Students")
    st.write(f"**Total Students:** {len(students_df)}")
    st.dataframe(students_df[['studentID', 'risk_level', 'prediction_probability']], use_container_width=True, height=600)


if __name__ == "__main__":
    main()
