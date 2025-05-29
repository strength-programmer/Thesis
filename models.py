from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(20), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    photo_url = db.Column(db.String(200))
    role = db.Column(db.String(50))
    hire_date = db.Column(db.Date)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    status = db.Column(db.String(20))
    department = db.Column(db.String(50))
    
    def to_dict(self):
        return {
            'id': self.id,
            'employeeId': self.employee_id,
            'fullName': self.full_name,
            'photoUrl': self.photo_url,
            'role': self.role,
            'hireDate': self.hire_date.strftime('%Y-%m-%d') if self.hire_date else None,
            'email': self.email,
            'phone': self.phone,
            'status': self.status,
            'department': self.department
        } 