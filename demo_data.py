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
        
        self.products = [
            {"title": "Premium Organic Cotton T-Shirt", "price": 29.99, "category": "Apparel"},
            {"title": "Wireless Bluetooth Headphones", "price": 79.99, "category": "Electronics"},
            {"title": "Eco-Friendly Water Bottle", "price": 24.99, "category": "Lifestyle"},
            {"title": "Artisan Coffee Blend - 1kg", "price": 34.99, "category": "Food & Beverage"},
            {"title": "Premium Yoga Mat", "price": 49.99, "category": "Fitness"},
            {"title": "Handcrafted Leather Wallet", "price": 89.99, "category": "Accessories"},
            {"title": "Smart Fitness Tracker", "price": 149.99, "category": "Electronics"},
            {"title": "Organic Skincare Set", "price": 69.99, "category": "Beauty"},
            {"title": "Bamboo Kitchen Utensil Set", "price": 39.99, "category": "Home & Garden"},
            {"title": "Vintage Style Sunglasses", "price": 54.99, "category": "Accessories"},
            {"title": "Protein Powder - Vanilla", "price": 44.99, "category": "Health"},
            {"title": "Memory Foam Pillow", "price": 59.99, "category": "Home & Garden"},
            {"title": "Stainless Steel Watch", "price": 199.99, "category": "Accessories"},
            {"title": "Essential Oil Diffuser", "price": 34.99, "category": "Home & Garden"},
            {"title": "Running Shoes - Performance", "price": 129.99, "category": "Footwear"},
        ]
        
        self.financial_statuses = ["paid", "pending", "refunded", "cancelled"]
        self.fulfillment_statuses = ["fulfilled", "pending", "partial", "unfulfilled"]
    
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
                        },
                        "line_items": self.generate_line_items(order_total)
                    }
                )
                
                db.session.add(order)
                customer_orders.append(order)
                total_spent += Decimal(str(order_total))
                orders_created += 1
            
            # Update customer totals
            customer.orders_count = num_orders
            customer.total_spent = total_spent
        
        # Generate abandoned carts for 10% of customers
        abandoned_customers = random.sample(customers, max(1, len(customers) // 10))
        for customer in abandoned_customers:
            self.generate_abandoned_cart(customer)
        
        db.session.commit()
        logging.info(f"Created {orders_created} demo orders with product data and abandoned carts")
        return orders_created
    
    def generate_line_items(self, order_total):
        """Generate realistic line items for an order"""
        # Determine number of items (1-4, weighted towards fewer items)
        num_items = random.choices([1, 2, 3, 4], weights=[50, 30, 15, 5])[0]
        
        # Select random products
        selected_products = random.sample(self.products, min(num_items, len(self.products)))
        
        line_items = []
        remaining_total = order_total
        
        for i, product in enumerate(selected_products):
            # For the last item, use remaining total; otherwise distribute proportionally
            if i == len(selected_products) - 1:
                item_total = remaining_total
            else:
                # Allocate 20-60% of remaining total to this item
                allocation_pct = random.uniform(0.2, 0.6)
                item_total = remaining_total * allocation_pct
                remaining_total -= item_total
            
            # Calculate quantity and price
            base_price = product["price"]
            max_qty = max(1, int(item_total / base_price))
            quantity = random.randint(1, min(3, max_qty))
            
            # Adjust price to match allocated total
            if quantity > 0:
                adjusted_price = round(item_total / quantity, 2)
            else:
                adjusted_price = base_price
                quantity = 1
            
            line_items.append({
                "id": random.randint(100000, 999999),
                "product_id": random.randint(10000, 99999),
                "variant_id": random.randint(10000, 99999),
                "title": product["title"],
                "variant_title": self.get_variant_title(product["category"]),
                "vendor": "Demo Store",
                "quantity": quantity,
                "price": str(adjusted_price),
                "grams": random.randint(100, 2000),
                "sku": f"SKU-{random.randint(1000, 9999)}",
                "variant_inventory_management": "shopify",
                "properties": [],
                "product_exists": True,
                "fulfillable_quantity": quantity,
                "fulfillment_service": "manual",
                "total_discount": "0.00",
                "fulfillment_status": None,
                "tax_lines": [],
                "name": f"{product['title']} - {self.get_variant_title(product['category'])}",
                "custom": False,
                "gift_card": False,
                "taxable": True,
                "tip": False
            })
        
        return line_items
    
    def get_variant_title(self, category):
        """Get appropriate variant title based on product category"""
        variants = {
            "Apparel": ["Small", "Medium", "Large", "X-Large"],
            "Footwear": ["Size 7", "Size 8", "Size 9", "Size 10", "Size 11"],
            "Electronics": ["Black", "White", "Silver"],
            "Beauty": ["Default"],
            "Health": ["Vanilla", "Chocolate", "Strawberry"],
            "Accessories": ["Black", "Brown", "Silver"]
        }
        
        category_variants = variants.get(category, ["Default"])
        return random.choice(category_variants)
    
    def generate_abandoned_cart(self, customer):
        """Generate abandoned cart data for a customer"""
        # Select 1-3 products for abandoned cart
        num_items = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        selected_products = random.sample(self.products, min(num_items, len(self.products)))
        
        line_items = []
        total_amount = 0
        
        for product in selected_products:
            quantity = random.choices([1, 2], weights=[80, 20])[0]
            item_price = product["price"]
            line_total = item_price * quantity
            total_amount += line_total
            
            line_items.append({
                "id": random.randint(100000, 999999),
                "product_id": random.randint(10000, 99999),
                "title": product["title"],
                "quantity": quantity,
                "price": str(item_price),
                "variant_title": self.get_variant_title(product["category"]),
                "vendor": "Demo Store",
                "sku": f"SKU-{random.randint(1000, 9999)}"
            })
        
        # Abandoned cart created 1-7 days ago
        abandoned_date = datetime.utcnow() - timedelta(days=random.randint(1, 7))
        
        # Store abandoned cart info in customer's shopify_data
        if not customer.shopify_data:
            customer.shopify_data = {}
        
        customer.shopify_data["abandoned_checkout"] = {
            "id": f"abandoned_{random.randint(100000, 999999)}",
            "created_at": abandoned_date.isoformat(),
            "updated_at": abandoned_date.isoformat(),
            "line_items": line_items,
            "total_price": str(total_amount),
            "currency": "USD",
            "customer_locale": "en",
            "abandoned_checkout_url": f"https://demo-store.myshopify.com/cart/c/{random.randint(100000, 999999)}"
        }
        
        db.session.add(customer)
    
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