# Backend - National Cancer Registry API

FastAPI-based backend for the National Cancer Registry Platform.

## 🚀 Quick Deploy to EC2

```bash
# Make script executable
chmod +x deploy-ec2.sh

# Run deployment
./deploy-ec2.sh
```

## 📦 What's Included

- **FastAPI Application** - Modern Python web framework
- **SQLAlchemy ORM** - Database models and queries
- **JWT Authentication** - Secure token-based auth
- **Alembic Migrations** - Database version control
- **Docker Support** - Containerized deployment
- **API Documentation** - Auto-generated Swagger docs

## 🏗️ Project Structure

```
Backend/
├── api/
│   └── v1/              # API endpoints
│       ├── auth.py      # Authentication
│       ├── users.py     # User management
│       ├── patients.py  # Patient records
│       ├── research.py  # Research data
│       └── ...
├── core/
│   ├── config.py        # Configuration
│   ├── deps.py          # Dependencies
│   └── security.py      # Security utilities
├── db/
│   ├── models/          # Database models
│   └── session.py       # Database session
├── services/
│   └── ml_sandbox.py    # ML execution service
├── main.py              # Application entry point
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables
```

## 🔧 Local Development

### Prerequisites
- Python 3.11+
- PostgreSQL
- Redis (optional)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your configuration

# Run migrations
alembic upgrade head

# Start development server
uvicorn main:app --reload --port 8000
```

### Access
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t registry-backend:latest .
```

### Run Container
```bash
docker run -d \
  --name registry-backend \
  -p 8000:8000 \
  --env-file .env \
  registry-backend:latest
```

### View Logs
```bash
docker logs -f registry-backend
```

## 🌐 API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh token
- `POST /api/v1/auth/logout` - User logout

### Users
- `GET /api/v1/users` - List users
- `POST /api/v1/users` - Create user
- `GET /api/v1/users/{id}` - Get user
- `PUT /api/v1/users/{id}` - Update user
- `DELETE /api/v1/users/{id}` - Delete user

### Patients
- `GET /api/v1/patients` - List patients
- `POST /api/v1/patients` - Create patient
- `GET /api/v1/patients/{id}` - Get patient
- `PUT /api/v1/patients/{id}` - Update patient
- `DELETE /api/v1/patients/{id}` - Delete patient

### Research
- `GET /api/v1/research` - List research data
- `POST /api/v1/research/query` - Execute research query

See full API documentation at `/docs` endpoint.

## 🔐 Environment Variables

Required variables in `.env`:

```env
# Database
DATABASE_URL=postgresql://user:password@host:5432/dbname

# JWT
JWT_SECRET=your-secret-key-change-this
JWT_ISS=registry.api
JWT_AUD=registry.clients

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Optional
REDIS_URL=redis://localhost:6379/0
WHO_CLIENT_ID=your-who-client-id
WHO_CLIENT_SECRET=your-who-client-secret
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test
pytest tests/test_auth.py
```

## 📊 Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# View migration history
alembic history
```

## 🔍 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Container Stats
```bash
docker stats registry-backend
```

### View Logs
```bash
# Real-time logs
docker logs -f registry-backend

# Last 100 lines
docker logs --tail 100 registry-backend
```

## 🛠️ Troubleshooting

### Database Connection Issues
```bash
# Test connection
python -c "from sqlalchemy import create_engine; import os; engine = create_engine(os.getenv('DATABASE_URL')); conn = engine.connect(); print('Connected!'); conn.close()"
```

### Port Already in Use
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>
```

### Container Won't Start
```bash
# Check logs
docker logs registry-backend

# Inspect container
docker inspect registry-backend

# Remove and recreate
docker stop registry-backend
docker rm registry-backend
docker run -d --name registry-backend -p 8000:8000 --env-file .env registry-backend:latest
```

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc (Alternative docs)
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔒 Security

- JWT token authentication
- Password hashing with bcrypt
- SQL injection prevention (SQLAlchemy)
- CORS protection
- Rate limiting (via Nginx)
- Environment variable secrets
- Non-root Docker user

## 📈 Performance

- Uvicorn ASGI server
- Multiple worker processes
- Connection pooling
- Redis caching (optional)
- Async database queries

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Run tests
4. Submit pull request

## 📄 License

Proprietary - National Cancer Registry Platform

## 📞 Support

- Check logs: `docker logs registry-backend`
- View docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

**Version**: 1.0.0  
**Python**: 3.11+  
**Framework**: FastAPI 0.109.0
