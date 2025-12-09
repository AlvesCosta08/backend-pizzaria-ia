# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'uma-chave-secreta-muito-forte-aqui')
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///pizzaria.db')
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://backend:5000/api')