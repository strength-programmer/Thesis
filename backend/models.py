from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Employee(db.Model):
    __tablename__ = 'employees'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(50), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    photo_url = db.Column(db.String(255), nullable=True)
    role = db.Column(db.String(50), nullable=False)
    hire_date = db.Column(db.Date, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    status = db.Column(db.String(20), nullable=False)
    department = db.Column(db.String(50), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'employeeId': self.employee_id,
            'fullName': self.full_name,
            'photoUrl': self.photo_url,
            'role': self.role,
            'hireDate': self.hire_date.isoformat() if self.hire_date else None,
            'email': self.email,
            'phone': self.phone,
            'status': self.status,
            'department': self.department
        } 