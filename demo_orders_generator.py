"""
Comprehensive demo orders generator for CLV platform
Generates realistic order data without customer dependencies
"""
import random
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from app import db
from models import ShopifyStore, Order, Product, OrderLineItem

class DemoOrdersGenerator:
    """Generate realistic demo orders for CLV analysis"""
    
    def __init__(self):
        self.product_categories = [
            "Electronics", "Clothing", "Home & Garden", "Sports", "Books",
            "Beauty", "Toys", "Automotive", "Health", "Food"
        ]
        
        self.customer_segments = {
            "high_value": {"min_orders": 4, "max_orders": 8, "min_value": 150, "max_value": 500},
            "medium_value": {"min_orders": 2, "max_orders": 4, "min_value": 75, "max_value": 200},
            "low_value": {"min_orders": 1, "max_orders": 2, "min_value": 25, "max_value": 100}
        }
    
    def hash_customer_id(self, customer_id: str) -> str:
        """Create anonymous hash for customer grouping"""
        return hashlib.sha256(f"demo_customer_{customer_id}".encode()).hexdigest()
    
    def get_or_create_demo_store(self):
        """Get or create demo store"""
        store = ShopifyStore.query.filter_by(shop_domain="demo-clv-store.myshopify.com").first()
        if not store:
            store = ShopifyStore()
            store.shop_domain = "demo-clv-store.myshopify.com"
            store.access_token = "demo_token_123"
            store.shop_id = "demo_shop_001"
            store.shop_name = "CLV Demo Store"
            store.email = "demo@clvstore.com"
            db.session.add(store)
            db.session.commit()
        return store
    
    def generate_demo_products(self, store):
        """Generate demo products if none exist"""
        if Product.query.filter_by(store_id=store.id).count() > 0:
            return
        
        products_data = [
            {"title": "Wireless Bluetooth Headphones", "category": "Electronics", "price": 89.99},
            {"title": "Cotton T-Shirt", "category": "Clothing", "price": 24.99},
            {"title": "Garden Plant Pot", "category": "Home & Garden", "price": 15.99},
            {"title": "Running Shoes", "category": "Sports", "price": 129.99},
            {"title": "Programming Book", "category": "Books", "price": 39.99},
            {"title": "Face Moisturizer", "category": "Beauty", "price": 34.99},
            {"title": "LEGO Building Set", "category": "Toys", "price": 79.99},
            {"title": "Car Phone Holder", "category": "Automotive", "price": 19.99},
            {"title": "Vitamin Supplements", "category": "Health", "price": 29.99},
            {"title": "Organic Coffee Beans", "category": "Food", "price": 22.99}
        ]
        
        for i, product_data in enumerate(products_data):
            product = Product()
            product.store_id = store.id
            product.shopify_product_id = f"demo_product_{i+1}"
            product.title = product_data["title"]
            product.category = product_data["category"]
            product.price = Decimal(str(product_data["price"]))
            product.inventory_quantity = random.randint(50, 200)
            product.units_sold = 0
            product.total_sales = Decimal('0.00')
            product.return_rate = random.uniform(0.02, 0.08)
            db.session.add(product)
        
        db.session.commit()
    
    def generate_demo_orders(self, store, num_customers=100):
        """Generate realistic demo orders"""
        products = Product.query.filter_by(store_id=store.id).all()
        if not products:
            self.generate_demo_products(store)
            products = Product.query.filter_by(store_id=store.id).all()
        
        orders_created = 0
        
        for customer_num in range(num_customers):
            customer_hash = self.hash_customer_id(str(customer_num))
            
            # Determine customer segment
            segment_type = random.choices(
                ["high_value", "medium_value", "low_value"],
                weights=[20, 50, 30]
            )[0]
            
            segment = self.customer_segments[segment_type]
            num_orders = random.randint(segment["min_orders"], segment["max_orders"])
            
            # Generate first order date
            first_order_date = datetime.utcnow() - timedelta(days=random.randint(30, 365))
            
            for order_num in range(num_orders):
                # Calculate order date
                if order_num == 0:
                    order_date = first_order_date
                else:
                    days_between = random.randint(7, 90)
                    order_date = first_order_date + timedelta(days=days_between * order_num)
                
                # Generate order value based on segment
                base_value = random.uniform(segment["min_value"], segment["max_value"])
                
                # Returning customers tend to spend more
                if order_num > 0:
                    base_value *= random.uniform(1.1, 1.4)
                
                order = Order()
                order.shopify_order_id = f"demo_order_{customer_num}_{order_num + 1}_{random.randint(1000, 9999)}"
                order.store_id = store.id
                order.customer_hash = customer_hash
                order.order_number = f"#{1000 + orders_created}"
                order.total_price = Decimal(str(round(base_value, 2)))
                order.subtotal_price = Decimal(str(round(base_value * 0.9, 2)))
                order.total_tax = Decimal(str(round(base_value * 0.1, 2)))
                order.currency = "USD"
                order.financial_status = "paid"
                order.fulfillment_status = "fulfilled"
                order.created_at = order_date
                order.updated_at = order_date
                order.processed_at = order_date
                order.order_sequence = order_num + 1
                order.days_since_first_order = (order_date - first_order_date).days
                order.is_returned = random.random() < 0.05  # 5% return rate
                order.shopify_data = {
                    "id": f"demo_order_{customer_num}_{order_num + 1}",
                    "order_number": f"#{1000 + orders_created}",
                    "demo": True,
                    "customer_segment": segment_type
                }
                
                db.session.add(order)
                db.session.flush()  # Get order ID
                
                # Generate line items for order
                self.generate_line_items(order, products, base_value)
                
                orders_created += 1
        
        db.session.commit()
        return orders_created
    
    def generate_line_items(self, order, products, order_total):
        """Generate realistic line items for an order"""
        num_items = random.randint(1, min(4, len(products)))
        selected_products = random.sample(products, num_items)
        
        remaining_total = float(order_total)
        
        for i, product in enumerate(selected_products):
            if i == len(selected_products) - 1:
                # Last item gets remaining total
                item_total = remaining_total
            else:
                # Random portion of remaining total
                item_total = remaining_total * random.uniform(0.2, 0.6)
                remaining_total -= item_total
            
            quantity = random.randint(1, 3)
            price_per_item = max(item_total / quantity, 1.0)
            
            line_item = OrderLineItem()
            line_item.order_id = order.id
            line_item.product_id = product.id
            line_item.store_id = order.store_id
            line_item.variant_title = f"{product.title} - Standard"
            line_item.quantity = quantity
            line_item.price = Decimal(str(round(price_per_item, 2)))
            line_item.total_discount = Decimal('0.00')
            line_item.is_returned = random.random() < 0.03
            line_item.shopify_data = {
                "product_id": product.shopify_product_id,
                "variant_id": f"variant_{product.id}",
                "demo": True
            }
            
            db.session.add(line_item)
            
            # Update product sales
            product.units_sold += quantity
            product.total_sales += Decimal(str(round(item_total, 2)))
    
    def clear_demo_data(self):
        """Clear all demo data"""
        store = ShopifyStore.query.filter_by(shop_domain="demo-clv-store.myshopify.com").first()
        if store:
            # Clear orders and related data
            OrderLineItem.query.filter_by(store_id=store.id).delete()
            Order.query.filter_by(store_id=store.id).delete()
            Product.query.filter_by(store_id=store.id).delete()
            db.session.commit()
    
    def populate_demo_data(self):
        """Create complete demo dataset"""
        try:
            # Get or create demo store
            store = self.get_or_create_demo_store()
            
            # Clear existing demo data
            self.clear_demo_data()
            
            # Generate fresh demo data
            self.generate_demo_products(store)
            orders_created = self.generate_demo_orders(store, 100)
            
            print(f"Demo data generated: {orders_created} orders for store {store.shop_domain}")
            return store
            
        except Exception as e:
            db.session.rollback()
            print(f"Error generating demo data: {str(e)}")
            raise