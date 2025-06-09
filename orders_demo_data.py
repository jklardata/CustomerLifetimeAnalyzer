import hashlib
import random
from datetime import datetime, timedelta
from decimal import Decimal
from app import db
from models import ShopifyStore, Order
import logging


class OrdersDemoDataGenerator:
    """Generate realistic demo orders for CLV analysis without storing customer data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Product categories for realistic orders
        self.products = [
            {"name": "Wireless Headphones", "price": 89.99, "category": "Electronics"},
            {"name": "Coffee Mug", "price": 14.99, "category": "Home"},
            {"name": "Running Shoes", "price": 129.99, "category": "Sports"},
            {"name": "Phone Case", "price": 24.99, "category": "Electronics"},
            {"name": "Yoga Mat", "price": 39.99, "category": "Sports"},
            {"name": "Candle", "price": 19.99, "category": "Home"},
            {"name": "Laptop Stand", "price": 49.99, "category": "Electronics"},
            {"name": "Water Bottle", "price": 17.99, "category": "Sports"},
            {"name": "Notebook", "price": 12.99, "category": "Office"},
            {"name": "Bluetooth Speaker", "price": 79.99, "category": "Electronics"}
        ]
    
    def hash_customer_id(self, customer_id: str) -> str:
        """Create anonymous hash for customer grouping"""
        return hashlib.sha256(str(customer_id).encode()).hexdigest()
    
    def generate_demo_orders(self, store, num_customers=100, max_orders_per_customer=5):
        """Generate realistic demo orders with anonymous customer grouping"""
        try:
            # Clear existing orders
            Order.query.filter_by(store_id=store.id).delete()
            db.session.commit()
            
            orders_created = 0
            
            # Generate customer patterns
            for customer_num in range(1, num_customers + 1):
                customer_hash = self.hash_customer_id(f"demo_customer_{customer_num}")
                
                # Random number of orders per customer (1-5)
                num_orders = random.randint(1, max_orders_per_customer)
                
                # Customer's first order date (last 365 days)
                first_order_date = datetime.utcnow() - timedelta(days=random.randint(1, 365))
                
                for order_num in range(num_orders):
                    # Calculate order date (subsequent orders spread over time)
                    if order_num == 0:
                        order_date = first_order_date
                    else:
                        # Subsequent orders 7-90 days after previous
                        days_after_first = random.randint(7, min(90 * order_num, 300))
                        order_date = first_order_date + timedelta(days=days_after_first)
                    
                    # Generate order value
                    base_value = random.uniform(25, 200)
                    # Returning customers tend to spend more
                    if order_num > 0:
                        base_value *= random.uniform(1.1, 1.4)
                    
                    order = Order(
                        shopify_order_id=f"demo_order_{customer_num}_{order_num + 1}_{random.randint(1000, 9999)}",
                        store_id=store.id,
                        customer_hash=customer_hash,
                        order_number=f"#{1000 + orders_created}",
                        total_price=Decimal(str(round(base_value, 2))),
                        subtotal_price=Decimal(str(round(base_value * 0.9, 2))),
                        total_tax=Decimal(str(round(base_value * 0.1, 2))),
                        currency="USD",
                        financial_status="paid",
                        fulfillment_status="fulfilled",
                        created_at=order_date,
                        updated_at=order_date,
                        processed_at=order_date,
                        order_sequence=order_num + 1,
                        days_since_first_order=(order_date - first_order_date).days,
                        is_returned=random.random() < 0.05,  # 5% return rate
                        shopify_data={
                            "id": f"demo_order_{customer_num}_{order_num + 1}",
                            "order_number": f"#{1000 + orders_created}",
                            "demo": True,
                            "customer_segment": self._get_customer_segment(order_num, base_value)
                        }
                    )
                    
                    db.session.add(order)
                    orders_created += 1
            
            db.session.commit()
            self.logger.info(f"Generated {orders_created} demo orders for {num_customers} anonymous customers")
            return orders_created
            
        except Exception as e:
            self.logger.error(f"Error generating demo orders: {str(e)}")
            db.session.rollback()
            return 0
    
    def _get_customer_segment(self, order_num, order_value):
        """Determine customer segment based on behavior"""
        if order_num >= 3 and order_value > 100:
            return "high_value"
        elif order_num >= 2:
            return "repeat_customer"
        else:
            return "new_customer"
    
    def populate_demo_data(self):
        """Create complete demo dataset for CLV analysis"""
        try:
            # Get or create demo store
            store = ShopifyStore.query.filter_by(shop_domain="clv-test-store.myshopify.com").first()
            
            if not store:
                self.logger.warning("No store found for demo data generation")
                return 0
            
            # Generate demo orders
            orders_created = self.generate_demo_orders(store, num_customers=100, max_orders_per_customer=5)
            
            self.logger.info(f"Demo data generation complete: {orders_created} orders")
            return orders_created
            
        except Exception as e:
            self.logger.error(f"Error in demo data population: {str(e)}")
            return 0
    
    def clear_demo_data(self):
        """Clear all demo data"""
        try:
            store = ShopifyStore.query.filter_by(shop_domain="clv-test-store.myshopify.com").first()
            if store:
                Order.query.filter_by(store_id=store.id).delete()
                db.session.commit()
                self.logger.info("Demo data cleared")
        except Exception as e:
            self.logger.error(f"Error clearing demo data: {str(e)}")
            db.session.rollback()