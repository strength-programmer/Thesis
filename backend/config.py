import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Change port from 5000 to 5432 and update password
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:1978@localhost:5000/employee_db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24))