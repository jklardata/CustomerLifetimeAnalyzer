import os
import ssl
import certifi

def fix_certificates():
    # Tell Python to use certifi's certificates
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    
    # Create a default SSL context that uses the certifi certificates
    default_context = ssl.create_default_context()
    default_context.load_verify_locations(cafile=certifi.where())
    
    return default_context

if __name__ == '__main__':
    fix_certificates()
    print("Certificates configured successfully!") 