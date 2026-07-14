"""
SQLite Database utilities for NL2SQL research.
Generates schema from dataset queries and executes SQL.
"""
import re
import sqlite3
import os
import random
import string


# Comprehensive schema covering main tables from the dataset
SCHEMA_SQL = """
-- Core schema for NL2SQL research demonstration
CREATE TABLE IF NOT EXISTS staff_directory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id VARCHAR(20) UNIQUE,
    full_name VARCHAR(100),
    email VARCHAR(100),
    department VARCHAR(50),
    role VARCHAR(50),
    is_active INTEGER DEFAULT 1,
    hire_date DATE,
    salary DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id VARCHAR(20) UNIQUE,
    name VARCHAR(100),
    email VARCHAR(100),
    department VARCHAR(50),
    position VARCHAR(50),
    hire_date DATE,
    salary DECIMAL(10,2),
    manager_id INTEGER,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(20) UNIQUE,
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(50),
    country VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id VARCHAR(30) UNIQUE,
    serial_number VARCHAR(50),
    asset_name VARCHAR(100),
    asset_type VARCHAR(50),
    pharmacies_id INTEGER,
    status VARCHAR(20) DEFAULT 'active',
    purchase_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS asset_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name VARCHAR(100),
    description TEXT
);

CREATE TABLE IF NOT EXISTS hazards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title VARCHAR(200),
    description TEXT,
    severity VARCHAR(20),
    status VARCHAR(20) DEFAULT 'open',
    reported_by INTEGER,
    facility_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name VARCHAR(200),
    event_type_id INTEGER,
    facility_id INTEGER,
    users_id INTEGER,
    prescriber_id INTEGER,
    harm_level_id INTEGER,
    status VARCHAR(20),
    event_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    user_id INTEGER,
    action_type VARCHAR(50),
    action_date DATE,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_staff_roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100),
    role_order INTEGER,
    facility_id INTEGER,
    forms_id INTEGER
);

CREATE TABLE IF NOT EXISTS event_type (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name VARCHAR(100),
    description TEXT,
    harm_level_id INTEGER
);

CREATE TABLE IF NOT EXISTS facilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    facility_name VARCHAR(200),
    address TEXT,
    city VARCHAR(50),
    state VARCHAR(50),
    phone VARCHAR(20),
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS forms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    form_name VARCHAR(200),
    form_type VARCHAR(50),
    facility_id INTEGER,
    created_by INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    doctor_id INTEGER,
    facility_id INTEGER,
    appointment_date DATETIME,
    duration_minutes INTEGER DEFAULT 30,
    status VARCHAR(20) DEFAULT 'scheduled',
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name VARCHAR(200),
    item_code VARCHAR(50) UNIQUE,
    quantity INTEGER DEFAULT 0,
    unit_price DECIMAL(10,2),
    facility_id INTEGER,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_number VARCHAR(30) UNIQUE,
    customer_id INTEGER,
    total_amount DECIMAL(12,2),
    tax_amount DECIMAL(12,2),
    status VARCHAR(20) DEFAULT 'pending',
    due_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    paid_at DATETIME
);

CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id INTEGER,
    item_id INTEGER,
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id INTEGER,
    receiver_id INTEGER,
    subject VARCHAR(200),
    body TEXT,
    is_read INTEGER DEFAULT 0,
    sent_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action VARCHAR(100),
    table_name VARCHAR(50),
    record_id INTEGER,
    details TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caller_id INTEGER,
    receiver_id INTEGER,
    duration_seconds INTEGER,
    call_type VARCHAR(20),
    status VARCHAR(20),
    started_at DATETIME,
    ended_at DATETIME
);

CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER,
    category VARCHAR(50),
    amount DECIMAL(10,2),
    description TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    approved_at DATETIME,
    approved_by INTEGER
);

CREATE TABLE IF NOT EXISTS holidays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    holiday_name VARCHAR(100),
    holiday_date DATE,
    facility_id INTEGER,
    is_mandatory INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS newsletter (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email VARCHAR(100) UNIQUE,
    name VARCHAR(100),
    subscribed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1,
    preferences TEXT
);

CREATE TABLE IF NOT EXISTS exams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_name VARCHAR(200),
    patient_id INTEGER,
    doctor_id INTEGER,
    facility_id INTEGER,
    exam_date DATE,
    results TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS login (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    username VARCHAR(100) UNIQUE,
    password_hash VARCHAR(255),
    last_login DATETIME,
    failed_attempts INTEGER DEFAULT 0,
    is_locked INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    entity_type VARCHAR(50),
    entity_id INTEGER,
    comment_text TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_deleted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS address_book (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    contact_name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def get_sample_inserts():
    """Generate realistic sample data for all tables."""
    inserts = []

    # staff_directory
    for i in range(1, 21):
        inserts.append(f"""INSERT INTO staff_directory (employee_id, full_name, email, department, role, is_active, hire_date, salary)
        VALUES ('EMP{i:03d}', 'Employee {i}', 'emp{i}@company.com', 
        '{["HR","IT","Finance","Operations","Medical"][i%5]}', 
        '{["Manager","Analyst","Coordinator","Specialist","Director"][i%5]}',
        {1 if i % 4 != 0 else 0}, '202{i%5}-0{(i%12)+1:02d}-15', {40000 + i * 2500});""")

    # employees
    for i in range(1, 21):
        inserts.append(f"""INSERT INTO employees (employee_id, name, email, department, position, hire_date, salary, is_active)
        VALUES ('E{i:04d}', 'Staff Member {i}', 'staff{i}@org.com',
        '{["Nursing","Pharmacy","Admin","IT","Finance"][i%5]}',
        '{["Nurse","Pharmacist","Admin","Developer","Accountant"][i%5]}',
        '20{10+i%13}-0{(i%12)+1:02d}-10', {35000 + i * 1800}, {1 if i < 18 else 0});""")

    # customers
    for i in range(1, 16):
        inserts.append(f"""INSERT INTO customers (customer_id, name, email, phone, city, country, is_active)
        VALUES ('CUST{i:04d}', 'Customer {i}', 'customer{i}@email.com',
        '555-{1000+i:04d}', '{["New York","London","Toronto","Sydney","Mumbai"][i%5]}',
        '{["USA","UK","Canada","Australia","India"][i%5]}', {1 if i < 14 else 0});""")

    # events
    for i in range(1, 21):
        inserts.append(f"""INSERT INTO events (event_name, event_type_id, facility_id, users_id, harm_level_id, status, event_date)
        VALUES ('Event {i}', {(i%5)+1}, {(i%4)+1}, {i}, {(i%3)+1},
        '{["open","closed","pending","resolved"][i%4]}', '2024-0{(i%12)+1:02d}-{(i%28)+1:02d}');""")

    # facilities
    for i in range(1, 6):
        inserts.append(f"""INSERT INTO facilities (facility_name, address, city, state, phone, is_active)
        VALUES ('Facility {i}', '{i} Main Street', '{["Chicago","Houston","Phoenix","Philadelphia","Dallas"][i-1]}',
        '{["IL","TX","AZ","PA","TX"][i-1]}', '555-{2000+i:04d}', 1);""")

    # hazards
    for i in range(1, 16):
        inserts.append(f"""INSERT INTO hazards (title, description, severity, status, reported_by, facility_id)
        VALUES ('Hazard Report {i}', 'Description of hazard {i}',
        '{["low","medium","high","critical"][i%4]}',
        '{["open","resolved","in_progress"][i%3]}', {i%10+1}, {i%5+1});""")

    # invoices
    for i in range(1, 16):
        inserts.append(f"""INSERT INTO invoices (invoice_number, customer_id, total_amount, tax_amount, status, due_date)
        VALUES ('INV-2024-{i:04d}', {(i%15)+1}, {100+i*50:.2f}, {(100+i*50)*0.1:.2f},
        '{["paid","pending","overdue"][i%3]}', '2024-{(i%12)+1:02d}-28');""")

    # inventory
    for i in range(1, 16):
        inserts.append(f"""INSERT INTO inventory (item_name, item_code, quantity, unit_price, facility_id)
        VALUES ('Item {i}', 'ITEM-{i:04d}', {10+i*5}, {5.00+i*2:.2f}, {i%5+1});""")

    return inserts


def initialize_db(db_path: str = "nl2sql_research.db") -> sqlite3.Connection:
    """Initialize SQLite database with schema and sample data."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Create tables
    cursor.executescript(SCHEMA_SQL)

    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM employees")
    count = cursor.fetchone()[0]

    if count == 0:
        inserts = get_sample_inserts()
        for insert in inserts:
            try:
                cursor.execute(insert)
            except Exception as e:
                pass  # Skip failed inserts

    conn.commit()
    return conn


def sanitize_and_adapt_query(sql: str) -> str:
    """
    Adapt a query with placeholder values to work with our demo DB.
    Replaces specific IDs, dates, etc. with known-good values.
    """
    import re

    # Replace specific numeric IDs with small numbers in range
    sql = re.sub(r'\b(?<!=)\s*=\s*(\d{4,})\b', lambda m: f' = {int(m.group(1)) % 10 + 1}', sql)

    # Replace long string literals with generic ones
    sql = re.sub(r"= '[A-Z0-9]{5,}'", "= 'ABC123'", sql)

    # Normalize table aliases if needed
    return sql


def execute_sql_safe(conn: sqlite3.Connection, sql: str, limit: int = 50):
    """
    Execute SQL safely and return (columns, rows, error).
    Only allows SELECT statements.
    """
    sql_clean = sql.strip().rstrip(';')

    # Only allow SELECT
    if not re.match(r'^\s*(SELECT|WITH)\s', sql_clean, re.IGNORECASE):
        return None, None, "Only SELECT queries are supported for demo execution."

    # Adapt query for our DB
    adapted = sanitize_and_adapt_query(sql_clean)

    # Add LIMIT if not present
    if 'LIMIT' not in adapted.upper():
        adapted = f"{adapted} LIMIT {limit}"

    try:
        cursor = conn.cursor()
        cursor.execute(adapted)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return columns, [list(row) for row in rows], None
    except Exception as e:
        return None, None, str(e)
