-- ============================================================================
-- Database Initialization and Seed Data for Local GenAI Assistant
-- Creates tables: customers, products, orders
-- ============================================================================

-- Drop tables in reverse foreign-key order if they exist
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- 1. Customers Table
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    email VARCHAR(200) UNIQUE NOT NULL,
    country VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Products Table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price NUMERIC(10, 2) NOT NULL CHECK (price >= 0),
    stock INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0)
);

-- 3. Orders Table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    order_date DATE NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    total_amount NUMERIC(12, 2) NOT NULL CHECK (total_amount >= 0),
    status VARCHAR(50) NOT NULL CHECK (status IN ('completed', 'pending', 'refunded', 'cancelled'))
);

-- Indices for performance on foreign keys and commonly filtered columns
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_product_id ON orders(product_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(status);

-- ============================================================================
-- SEED DATA
-- ============================================================================

-- Seed Customers
INSERT INTO customers (id, name, email, country, created_at) VALUES
(1, 'TechCorp Solutions', 'billing@techcorp.com', 'United States', '2025-01-15 09:30:00Z'),
(2, 'Acme Logistics', 'accounts@acmelogistics.com', 'Canada', '2025-02-10 11:15:00Z'),
(3, 'Global Health Systems', 'procurement@globalhealth.org', 'United Kingdom', '2025-03-01 14:00:00Z'),
(4, 'Apex Retailers', 'finance@apexretail.com', 'Australia', '2025-03-22 08:45:00Z'),
(5, 'CloudScale Networks', 'ops@cloudscale.io', 'Germany', '2025-04-05 16:20:00Z'),
(6, 'Quantum Innovations', 'admin@quantuminnovate.com', 'United States', '2025-05-12 10:00:00Z'),
(7, 'Nexus Media Works', 'contact@nexusmedia.com', 'France', '2025-06-18 13:30:00Z'),
(8, 'Horizon Robotics', 'support@horizonrobotics.com', 'Japan', '2025-07-01 09:00:00Z'),
(9, 'Vanguard Security', 'security@vanguardsec.com', 'United States', '2025-08-14 15:45:00Z'),
(10, 'Peak Enterprises', 'orders@peakenterprises.com', 'Singapore', '2025-09-02 12:10:00Z');

-- Reset customer sequence
SELECT setval('customers_id_seq', (SELECT MAX(id) FROM customers));

-- Seed Products
INSERT INTO products (id, name, category, price, stock) VALUES
(1, 'Enterprise Cloud Server Rack', 'Infrastructure', 4500.00, 15),
(2, 'AI Developer Workstation', 'Hardware', 2800.00, 30),
(3, 'Developer Pro Laptop 16"', 'Computers', 1950.00, 50),
(4, 'UltraWide 4K Dual Monitor', 'Displays', 750.00, 80),
(5, 'Ergonomic Executive Chair', 'Furniture', 420.00, 45),
(6, 'High-Throughput Enterprise Router', 'Networking', 1250.00, 25),
(7, 'Hardware Security Token Key', 'Security', 65.00, 300),
(8, 'NVMe Enterprise SSD 4TB', 'Storage', 380.00, 120),
(9, 'Smart Conference Room Audio/Video Bar', 'Collaboration', 890.00, 40),
(10, 'Mechanical Backlit Office Keyboard', 'Accessories', 120.00, 150);

-- Reset product sequence
SELECT setval('products_id_seq', (SELECT MAX(id) FROM products));

-- Seed Orders
-- Including orders with various statuses ('completed', 'refunded', 'pending', 'cancelled')
-- across recent months for revenue and refund queries
INSERT INTO orders (id, customer_id, product_id, order_date, quantity, total_amount, status) VALUES
(1, 1, 1, '2026-08-05', 3, 13500.00, 'completed'),
(2, 1, 3, '2026-08-12', 2, 3900.00, 'completed'),
(3, 2, 2, '2026-08-15', 2, 5600.00, 'completed'),
(4, 3, 6, '2026-08-20', 4, 5000.00, 'completed'),
(5, 4, 4, '2026-08-22', 5, 3750.00, 'completed'),
(6, 5, 1, '2026-08-28', 1, 4500.00, 'completed'),
(7, 6, 2, '2026-09-02', 4, 11200.00, 'completed'),
(8, 7, 5, '2026-09-05', 8, 3360.00, 'completed'),
(9, 8, 3, '2026-09-08', 3, 5850.00, 'completed'),
(10, 9, 7, '2026-09-10', 50, 3250.00, 'completed'),
(11, 1, 8, '2026-09-14', 6, 2280.00, 'completed'),
(12, 10, 2, '2026-09-18', 3, 8400.00, 'completed'),
-- Refunded orders in last month (September 2026) for the combined RAG+SQL refund queries
(13, 2, 4, '2026-09-20', 2, 1500.00, 'refunded'),
(14, 4, 5, '2026-09-22', 1, 420.00, 'refunded'),
(15, 3, 9, '2026-09-25', 1, 890.00, 'refunded'),
(16, 5, 3, '2026-09-27', 2, 3900.00, 'completed'),
(17, 6, 1, '2026-09-28', 2, 9000.00, 'completed'),
(18, 7, 8, '2026-09-29', 10, 3800.00, 'completed'),
(19, 8, 6, '2026-09-30', 2, 2500.00, 'completed'),
(20, 9, 2, '2026-10-01', 2, 5600.00, 'completed'),
(21, 10, 1, '2026-10-02', 1, 4500.00, 'completed'),
(22, 1, 4, '2026-10-03', 4, 3000.00, 'completed'),
(23, 2, 10, '2026-10-04', 15, 1800.00, 'completed'),
(24, 3, 3, '2026-10-04', 2, 3900.00, 'pending'),
(25, 4, 8, '2026-10-05', 5, 1900.00, 'completed'),
(26, 6, 7, '2026-10-05', 20, 1300.00, 'completed'),
(27, 7, 2, '2026-10-06', 1, 2800.00, 'pending'),
(28, 8, 5, '2026-10-06', 4, 1680.00, 'completed'),
(29, 9, 9, '2026-10-06', 2, 1780.00, 'completed'),
(30, 10, 4, '2026-10-07', 3, 2250.00, 'completed');

-- Reset order sequence
SELECT setval('orders_id_seq', (SELECT MAX(id) FROM orders));
