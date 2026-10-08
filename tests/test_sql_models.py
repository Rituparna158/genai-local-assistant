"""
Unit tests for SQLAlchemy 2.0 ORM Models (Customer, Product, Order).
Validates model creation, constraints, relationships, and serialization.
"""

from datetime import date
from decimal import Decimal
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from src.sql_agent.models import Base, Customer, Order, Product


@pytest.fixture
def in_memory_db():
    """Provides a fresh in-memory SQLite database session for model testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session: Session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_customer_creation_and_serialization(in_memory_db: Session):
    """Verifies creating a customer and serializing with to_dict()."""
    cust = Customer(
        name="Nova Technologies",
        email="contact@novatech.io",
        country="Germany",
    )
    in_memory_db.add(cust)
    in_memory_db.commit()
    in_memory_db.refresh(cust)

    assert cust.id is not None
    assert cust.name == "Nova Technologies"
    assert cust.email == "contact@novatech.io"
    assert cust.country == "Germany"

    data = cust.to_dict()
    assert data["name"] == "Nova Technologies"
    assert data["email"] == "contact@novatech.io"
    assert data["country"] == "Germany"
    assert "created_at" in data
    assert repr(cust).startswith("<Customer(")


def test_product_creation_and_serialization(in_memory_db: Session):
    """Verifies creating a product and serializing with to_dict()."""
    prod = Product(
        name="Wireless Ergonomic Mouse",
        category="Accessories",
        price=Decimal("49.99"),
        stock=150,
    )
    in_memory_db.add(prod)
    in_memory_db.commit()
    in_memory_db.refresh(prod)

    assert prod.id is not None
    assert prod.price == Decimal("49.99")
    assert prod.stock == 150

    data = prod.to_dict()
    assert data["name"] == "Wireless Ergonomic Mouse"
    assert data["price"] == 49.99
    assert data["stock"] == 150
    assert repr(prod).startswith("<Product(")


def test_order_relationship_and_foreign_keys(in_memory_db: Session):
    """Verifies creating an order with foreign keys and navigating relationships."""
    cust = Customer(
        name="Atlas Corp",
        email="orders@atlascorp.com",
        country="United States",
    )
    prod = Product(
        name="Developer Monitor 27\"",
        category="Displays",
        price=Decimal("350.00"),
        stock=20,
    )
    in_memory_db.add_all([cust, prod])
    in_memory_db.commit()

    order = Order(
        customer_id=cust.id,
        product_id=prod.id,
        order_date=date(2026, 9, 15),
        quantity=2,
        total_amount=Decimal("700.00"),
        status="completed",
    )
    in_memory_db.add(order)
    in_memory_db.commit()
    in_memory_db.refresh(order)

    # Test relationship navigation
    assert order.customer.name == "Atlas Corp"
    assert order.product.name == "Developer Monitor 27\""
    assert len(cust.orders) == 1
    assert cust.orders[0].id == order.id

    # Test order serialization
    order_dict = order.to_dict()
    assert order_dict["quantity"] == 2
    assert order_dict["total_amount"] == 700.00
    assert order_dict["status"] == "completed"
    assert order_dict["order_date"] == "2026-09-15"
    assert repr(order).startswith("<Order(")
