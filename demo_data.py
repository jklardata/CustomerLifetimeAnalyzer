import random
from datetime import datetime, timedelta
from decimal import Decimal
from app import db
from models import ShopifyStore, Customer, Order
import logging

class DemoDataGenerator:
    """Generate realistic demo data for CLV platform"""
    
    def __init__(self):
        self.demo_store_domain = "demo-store.myshopify.com"
        self.first_names = [
            "Emma", "Liam", "Olivia", "Noah", "Ava", "William", "Sophia", "James",
            "Isabella", "Oliver", "Charlotte", "Benjamin", "Amelia", "Elijah", "Mia",
            "Lucas", "Harper", "Mason", "Evelyn", "Logan", "Abigail", "Alexander",
            "Emily", "Ethan", "Elizabeth", "Jacob", "Sofia", "Michael", "Avery",
            "Daniel", "Ella", "Henry", "Madison", "Jackson", "Scarlett", "Sebastian"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
            "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King"
        ]
        
        self.email_domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
            "aol.com", "protonmail.com", "mail.com"
        ]
    
    def create_demo_store(self):
        """Create or get the demo store"""
        store = ShopifyStore.query.filter_by(shop_domain=self.demo_store_domain).first()
        
        if not store:
            store = ShopifyStore(
                shop_domain=self.demo_store_domain,
                access_token="demo_token_12345",
                shop_id="demo_shop_001",
                shop_name="CLV Demo Store",
                email="demo@clvstore.com"
            )
            db.session.add(store)
            db.session.commit()
            logging.info(f"Created demo store: {store.shop_name}")
        
        return store
    
    def generate_demo_customers(self, store, count=150):
        """Generate demo customers with realistic data"""
        customers_created = 0
        
        for i in range(count):
            # Check if customer already exists
            shopify_customer_id = f"demo_customer_{i+1:04d}"
            existing = Customer.query.filter_by(
                shopify_customer_id=shopify_customer_id,
                store_id=store.id
            ).first()
            
            if existing:
                continue
            
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            email = f"{first_name.lower()}.{last_name.lower()}@{random.choice(self.email_domains)}"
            
            # Customer registration date (past 2 years)
            created_date = datetime.utcnow() - timedelta(days=random.randint(30, 730))
            
            customer = Customer(
                shopify_customer_id=shopify_customer_id,
                store_id=store.id,
                email=email,
                first_name=first_name,
                last_name=last_name,
                created_at=created_date,
                shopify_data={
                    "id": shopify_customer_id,
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "created_at": created_date.isoformat(),
                    "state": "enabled",
                    "total_spent": "0.00",
                    "orders_count": 0
                }
            )
            
            db.session.add(customer)
            customers_created += 1
        
        db.session.commit()
        logging.info(f"Created {customers_created} demo customers")
        return customers_created
    
    def generate_demo_orders(self, store, max_orders_per_customer=8):
        """Generate demo orders for customers"""
        customers = Customer.query.filter_by(store_id=store.id).all()
        orders_created = 0
        
        for customer in customers:
            # Random number of orders per customer (weighted towards fewer orders)
            num_orders = random.choices(
                range(0, max_orders_per_customer + 1),
                weights=[15, 25, 20, 15, 10, 8, 5, 2, 1]  # Most customers have 1-3 orders
            )[0]
            
            total_spent = Decimal('0.00')
            customer_orders = []
            
            for order_num in range(num_orders):
                # Order date between customer creation and now
                days_since_registration = (datetime.utcnow() - customer.created_at).days
                if days_since_registration <= 0:
                    order_date = customer.created_at + timedelta(days=1)
                else:
                    order_date = customer.created_at + timedelta(days=random.randint(1, days_since_registration))
                
                # Realistic order values (log-normal distribution)
                base_price = random.lognormvariate(4.0, 0.8)  # Mean around $55, with long tail
                order_total = round(max(15.0, min(500.0, base_price)), 2)  # Clamp between $15-$500
                
                # Occasional high-value orders
                if random.random() < 0.05:  # 5% chance
                    order_total = round(random.uniform(500, 2000), 2)
                
                # Some returns (5% chance)
                is_returned = random.random() < 0.05
                
                shopify_order_id = f"demo_order_{customer.shopify_customer_id}_{order_num+1:03d}"
                
                order = Order(
                    shopify_order_id=shopify_order_id,
                    store_id=store.id,
                    customer_id=customer.id,
                    order_number=f"#{1000 + orders_created + 1}",
                    total_price=Decimal(str(order_total)),
                    subtotal_price=Decimal(str(order_total * 0.9)),
                    total_tax=Decimal(str(order_total * 0.1)),
                    currency="USD",
                    financial_status=random.choice(["paid", "paid", "paid", "pending", "refunded"]),
                    fulfillment_status=random.choice(["fulfilled", "fulfilled", "partial", "unfulfilled"]),
                    created_at=order_date,
                    updated_at=order_date,
                    processed_at=order_date,
                    is_returned=is_returned,
                    return_reason="Defective item" if is_returned else None,
                    shopify_data={
                        "id": shopify_order_id,
                        "order_number": f"#{1000 + orders_created + 1}",
                        "total_price": str(order_total),
                        "created_at": order_date.isoformat(),
                        "customer": {
                            "id": customer.shopify_customer_id,
                            "email": customer.email
                        }
                    }
                )
                
                db.session.add(order)
                customer_orders.append(order)
                total_spent += Decimal(str(order_total))
                orders_created += 1
            
            # Update customer totals
            customer.orders_count = num_orders
            customer.total_spent = total_spent
        
        db.session.commit()
        logging.info(f"Created {orders_created} demo orders")
        return orders_created
    
    def populate_demo_data(self):
        """Populate complete demo dataset"""
        try:
            # Create demo store
            store = self.create_demo_store()
            
            # Check if data already exists
            existing_customers = Customer.query.filter_by(store_id=store.id).count()
            existing_orders = Order.query.filter_by(store_id=store.id).count()
            
            if existing_customers > 0 and existing_orders > 0:
                logging.info(f"Demo data already exists: {existing_customers} customers, {existing_orders} orders")
                return store, existing_customers, existing_orders
            
            # Generate customers and orders
            customers_count = self.generate_demo_customers(store, 150)
            orders_count = self.generate_demo_orders(store)
            
            # Calculate CLV for all customers
            from clv_calculator import CLVCalculator
            clv_calculator = CLVCalculator()
            clv_updates = clv_calculator.calculate_store_clv(store)
            
            logging.info(f"Demo data populated: {customers_count} customers, {orders_count} orders, {clv_updates} CLV calculations")
            return store, customers_count, orders_count
            
        except Exception as e:
            logging.error(f"Error populating demo data: {str(e)}")
            db.session.rollback()
            raise