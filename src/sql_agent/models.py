"""
SQLAlchemy 2.0 ORM Models for Database Entities.
Defines declarative mappings for customers, products, and orders tables.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List
from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative Base class for all SQLAlchemy ORM models."""
    pass


class Customer(Base):
    """
    Customer entity representation.
    Tracks corporate/individual clients, locations, and creation timestamps.
    """
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    orders: Mapped[List["Order"]] = relationship(
        "Order",
        back_populates="customer",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the customer model to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "country": self.country,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<Customer(id={self.id}, name='{self.name}', country='{self.country}')>"


class Product(Base):
    """
    Product catalog entity representation.
    Tracks item name, category, unit price, and available inventory stock.
    """
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        CheckConstraint("price >= 0", name="check_product_price_non_negative"),
        CheckConstraint("stock >= 0", name="check_product_stock_non_negative"),
    )

    # Relationships
    orders: Mapped[List["Order"]] = relationship(
        "Order",
        back_populates="product",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the product model to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "price": float(self.price) if self.price is not None else 0.0,
            "stock": self.stock,
        }

    def __repr__(self) -> str:
        return f"<Product(id={self.id}, name='{self.name}', price={self.price}, stock={self.stock})>"


class Order(Base):
    """
    Order transaction entity representation.
    Tracks purchase history linking customer and product with status and financials.
    """
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    order_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    __table_args__ = (
        CheckConstraint("quantity > 0", name="check_order_quantity_positive"),
        CheckConstraint("total_amount >= 0", name="check_order_total_non_negative"),
        CheckConstraint(
            "status IN ('completed', 'pending', 'refunded', 'cancelled')",
            name="check_order_status_valid",
        ),
    )

    # Relationships
    customer: Mapped["Customer"] = relationship("Customer", back_populates="orders")
    product: Mapped["Product"] = relationship("Product", back_populates="orders")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the order model to a dictionary."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "product_id": self.product_id,
            "order_date": self.order_date.isoformat() if self.order_date else None,
            "quantity": self.quantity,
            "total_amount": float(self.total_amount) if self.total_amount is not None else 0.0,
            "status": self.status,
        }

    def __repr__(self) -> str:
        return (
            f"<Order(id={self.id}, customer_id={self.customer_id}, "
            f"product_id={self.product_id}, total={self.total_amount}, status='{self.status}')>"
        )
