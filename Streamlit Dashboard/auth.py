# auth.py
# Authentication and user management functions

import bcrypt
from typing import Optional, Dict, Any


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password as string
    """
    # Convert password to bytes
    password_bytes = password.encode('utf-8')
    
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Return as string
    return hashed.decode('utf-8')


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password: Plain text password to verify
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        # Convert to bytes
        password_bytes = password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        
        # Check password
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        print(f"Error verifying password: {e}")
        return False


def authenticate_user(username: str, password: str, db_module) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user with username and password.
    
    Args:
        username: Username
        password: Plain text password
        db_module: Database module with get_user_by_username function
        
    Returns:
        User dictionary if authenticated, None otherwise
    """
    # Get user from database
    user = db_module.get_user_by_username(username)
    
    if user is None:
        return None
    
    # Verify password
    if verify_password(password, user['password_hash']):
        # Return user info without password
        return {
            'id': user['id'],
            'username': user['username'],
            'full_name': user['full_name'],
            'role': user['role'],
            'created_at': user['created_at']
        }
    
    return None


def is_admin(user: Dict[str, Any]) -> bool:
    """Check if user is an administrator."""
    return user.get('role') == 'admin'


def is_educator(user: Dict[str, Any]) -> bool:
    """Check if user is an educator."""
    return user.get('role') == 'educator'


def validate_password_strength(password: str, min_length: int = 4) -> tuple[bool, str]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        min_length: Minimum required length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters long"
    
    return True, ""


def check_username_available(username: str, db_module, exclude_user_id: Optional[int] = None) -> bool:
    """
    Check if username is available.
    
    Args:
        username: Username to check
        db_module: Database module
        exclude_user_id: User ID to exclude from check (for updates)
        
    Returns:
        True if available, False otherwise
    """
    user = db_module.get_user_by_username(username)
    
    if user is None:
        return True
    
    # If updating own username, allow it
    if exclude_user_id is not None and user['id'] == exclude_user_id:
        return True
    
    return False
