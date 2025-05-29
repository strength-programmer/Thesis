import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Change port from 5000 to 5433 and update password
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:042401@localhost:5433/employee_db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24))