import requests
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

class ShopifyClient:
    """Client for interacting with Shopify API"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = None
        self.rate_limit_delay = 0.5  # Default delay between requests
        
    def set_access_token(self, access_token: str):
        """Set the access token for API requests"""
        self.access_token = access_token
    
    def get_access_token(self, shop_domain: str, code: str) -> Optional[str]:
        """Exchange authorization code for access token"""
        try:
            url = f"https://{shop_domain}/admin/oauth/access_token"
            data = {
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'code': code
            }
            
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get('access_token')
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting access token: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting access token: {str(e)}")
            return None
    
    def make_request(self, shop_domain: str, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated request to Shopify API with retries and rate limiting"""
        if not self.access_token:
            logging.error("No access token set")
            return None
        
        url = f"https://{shop_domain}/admin/api/2023-10/{endpoint}"
        headers = {
            'X-Shopify-Access-Token': self.access_token,
            'Content-Type': 'application/json'
        }
        
        logging.info(f"Making request to: {url}")
        logging.info(f"Access token available: {'Yes' if self.access_token else 'No'}")
        if self.access_token:
            logging.info(f"Access token starts with: {self.access_token[:10]}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                response = requests.get(url, headers=headers, params=params or {}, timeout=30)
                
                logging.info(f"Response status code: {response.status_code}")
                logging.info(f"Response headers: {dict(response.headers)}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 2))
                    logging.warning(f"Rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                if response.status_code != 200:
                    logging.error(f"API error: {response.status_code} - {response.text}")
                    
                response.raise_for_status()
                json_data = response.json()
                logging.info(f"Response data keys: {list(json_data.keys()) if json_data else 'None'}")
                return json_data
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def get_shop_info(self, shop_domain: str) -> Optional[Dict]:
        """Get shop information"""
        try:
            result = self.make_request(shop_domain, 'shop.json')
            return result.get('shop') if result else None
        except Exception as e:
            logging.error(f"Error getting shop info: {str(e)}")
            return None
    
    def get_customers(self, shop_domain: str, limit: int = 250) -> List[Dict]:
        """Get customers with pagination"""
        customers = []
        params = {'limit': limit}
        
        try:
            logging.info(f"Requesting customers from {shop_domain} with limit {limit}")
            while True:
                result = self.make_request(shop_domain, 'customers.json', params)
                logging.info(f"API response received: {result is not None}")
                
                if not result:
                    logging.warning("No result from customers API call")
                    break
                    
                if 'customers' not in result:
                    logging.warning(f"No 'customers' key in result. Keys: {list(result.keys()) if result else 'None'}")
                    break
                
                batch = result['customers']
                customers.extend(batch)
                
                logging.info(f"Fetched {len(batch)} customers, total: {len(customers)}")
                
                # Check for pagination
                if len(batch) < limit:
                    break
                
                # Update params for next page
                if batch:
                    params['since_id'] = batch[-1]['id']
        
        except Exception as e:
            logging.error(f"Error fetching customers: {str(e)}")
        
        return customers
    
    def get_orders(self, shop_domain: str, limit: int = 250) -> List[Dict]:
        """Get orders with pagination"""
        orders = []
        params = {
            'limit': limit,
            'status': 'any',  # Include all orders
            'fulfillment_status': 'any'
        }
        
        try:
            while True:
                result = self.make_request(shop_domain, 'orders.json', params)
                if not result or 'orders' not in result:
                    break
                
                batch = result['orders']
                orders.extend(batch)
                
                logging.info(f"Fetched {len(batch)} orders, total: {len(orders)}")
                
                # Check for pagination
                if len(batch) < limit:
                    break
                
                # Update params for next page
                if batch:
                    params['since_id'] = batch[-1]['id']
        
        except Exception as e:
            logging.error(f"Error fetching orders: {str(e)}")
        
        return orders
    
    def get_abandoned_checkouts(self, shop_domain: str, limit: int = 250) -> List[Dict]:
        """Get abandoned checkouts"""
        checkouts = []
        params = {'limit': limit}
        
        try:
            while True:
                result = self.make_request(shop_domain, 'checkouts.json', params)
                if not result or 'checkouts' not in result:
                    break
                
                batch = result['checkouts']
                checkouts.extend(batch)
                
                logging.info(f"Fetched {len(batch)} abandoned checkouts, total: {len(checkouts)}")
                
                # Check for pagination
                if len(batch) < limit:
                    break
                
                # Update params for next page
                if batch:
                    params['since_id'] = batch[-1]['id']
        
        except Exception as e:
            logging.error(f"Error fetching abandoned checkouts: {str(e)}")
        
        return checkouts
    
    def get_customer_orders(self, shop_domain: str, customer_id: str) -> List[Dict]:
        """Get orders for a specific customer"""
        try:
            endpoint = f'customers/{customer_id}/orders.json'
            result = self.make_request(shop_domain, endpoint)
            return result.get('orders', []) if result else []
        except Exception as e:
            logging.error(f"Error fetching customer orders: {str(e)}")
            return []
    
    @staticmethod
    def parse_datetime(datetime_str: str) -> Optional[datetime]:
        """Parse Shopify datetime string"""
        try:
            # Shopify uses ISO 8601 format
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except Exception as e:
            logging.error(f"Error parsing datetime {datetime_str}: {str(e)}")
            return None
    
    def validate_webhook(self, data: bytes, hmac_header: str) -> bool:
        """Validate Shopify webhook HMAC"""
        import hmac
        import hashlib
        import base64
        
        try:
            calculated_hmac = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    data,
                    digestmod=hashlib.sha256
                ).digest()
            ).decode()
            
            return hmac.compare_digest(calculated_hmac, hmac_header)
        except Exception as e:
            logging.error(f"Error validating webhook: {str(e)}")
            return False
