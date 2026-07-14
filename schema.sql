-- NL2SQL Research — MySQL Schema
-- For use with MySQL Workbench
-- Run: mysql -u root -p nl2sql_research < schema.sql

CREATE DATABASE IF NOT EXISTS nl2sql_research;
USE nl2sql_research;

CREATE TABLE IF NOT EXISTS staff_directory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id VARCHAR(20) UNIQUE,
    full_name VARCHAR(100),
    email VARCHAR(100),
    department VARCHAR(50),
    role VARCHAR(50),
    is_active TINYINT(1) DEFAULT 1,
    hire_date DATE,
    salary DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id VARCHAR(20) UNIQUE,
    name VARCHAR(100),
    email VARCHAR(100),
    department VARCHAR(50),
    position VARCHAR(50),
    hire_date DATE,
    salary DECIMAL(10,2),
    manager_id INT,
    is_active TINYINT(1) DEFAULT 1
);

CREATE TABLE IF NOT EXISTS customers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(20) UNIQUE,
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(50),
    country VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active TINYINT(1) DEFAULT 1
);

CREATE TABLE IF NOT EXISTS assets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    asset_id VARCHAR(30) UNIQUE,
    serial_number VARCHAR(50),
    asset_name VARCHAR(100),
    asset_type VARCHAR(50),
    pharmacies_id INT,
    status VARCHAR(20) DEFAULT 'active',
    purchase_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS asset_types (
    id INT AUTO_INCREMENT PRIMARY KEY,
    type_name VARCHAR(100),
    description TEXT
);

CREATE TABLE IF NOT EXISTS hazards (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(200),
    description TEXT,
    severity ENUM('low','medium','high','critical'),
    status VARCHAR(20) DEFAULT 'open',
    reported_by INT,
    facility_id INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME
);

CREATE TABLE IF NOT EXISTS events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event_name VARCHAR(200),
    event_type_id INT,
    facility_id INT,
    users_id INT,
    prescriber_id INT,
    harm_level_id INT,
    status VARCHAR(20),
    event_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_actions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event_id INT,
    user_id INT,
    action_type VARCHAR(50),
    action_date DATE,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_staff_roles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    role_order INT,
    facility_id INT,
    forms_id INT
);

CREATE TABLE IF NOT EXISTS event_type (
    id INT AUTO_INCREMENT PRIMARY KEY,
    type_name VARCHAR(100),
    description TEXT,
    harm_level_id INT
);

CREATE TABLE IF NOT EXISTS facilities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    facility_name VARCHAR(200),
    address TEXT,
    city VARCHAR(50),
    state VARCHAR(50),
    phone VARCHAR(20),
    is_active TINYINT(1) DEFAULT 1
);

CREATE TABLE IF NOT EXISTS forms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    form_name VARCHAR(200),
    form_type VARCHAR(50),
    facility_id INT,
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active TINYINT(1) DEFAULT 1
);

CREATE TABLE IF NOT EXISTS appointments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    doctor_id INT,
    facility_id INT,
    appointment_date DATETIME,
    duration_minutes INT DEFAULT 30,
    status VARCHAR(20) DEFAULT 'scheduled',
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inventory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    item_name VARCHAR(200),
    item_code VARCHAR(50) UNIQUE,
    quantity INT DEFAULT 0,
    unit_price DECIMAL(10,2),
    facility_id INT,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS invoices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    invoice_number VARCHAR(30) UNIQUE,
    customer_id INT,
    total_amount DECIMAL(12,2),
    tax_amount DECIMAL(12,2),
    status ENUM('pending','paid','overdue','cancelled') DEFAULT 'pending',
    due_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    paid_at DATETIME
);

CREATE TABLE IF NOT EXISTS order_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    invoice_id INT,
    item_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sender_id INT,
    receiver_id INT,
    subject VARCHAR(200),
    body TEXT,
    is_read TINYINT(1) DEFAULT 0,
    sent_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    action VARCHAR(100),
    table_name VARCHAR(50),
    record_id INT,
    details TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS calls (
    id INT AUTO_INCREMENT PRIMARY KEY,
    caller_id INT,
    receiver_id INT,
    duration_seconds INT,
    call_type VARCHAR(20),
    status VARCHAR(20),
    started_at DATETIME,
    ended_at DATETIME
);

CREATE TABLE IF NOT EXISTS expenses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT,
    category VARCHAR(50),
    amount DECIMAL(10,2),
    description TEXT,
    status ENUM('pending','approved','rejected') DEFAULT 'pending',
    submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    approved_at DATETIME,
    approved_by INT
);

CREATE TABLE IF NOT EXISTS holidays (
    id INT AUTO_INCREMENT PRIMARY KEY,
    holiday_name VARCHAR(100),
    holiday_date DATE,
    facility_id INT,
    is_mandatory TINYINT(1) DEFAULT 1
);

CREATE TABLE IF NOT EXISTS newsletter (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(100) UNIQUE,
    name VARCHAR(100),
    subscribed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active TINYINT(1) DEFAULT 1,
    preferences TEXT
);

CREATE TABLE IF NOT EXISTS exams (
    id INT AUTO_INCREMENT PRIMARY KEY,
    exam_name VARCHAR(200),
    patient_id INT,
    doctor_id INT,
    facility_id INT,
    exam_date DATE,
    results TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS login (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    username VARCHAR(100) UNIQUE,
    password_hash VARCHAR(255),
    last_login DATETIME,
    failed_attempts INT DEFAULT 0,
    is_locked TINYINT(1) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS comments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    entity_type VARCHAR(50),
    entity_id INT,
    comment_text TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_deleted TINYINT(1) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS address_book (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    contact_name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
